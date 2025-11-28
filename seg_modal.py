import modal
import io
import base64
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

app = modal.App("grounding-segmentation-service")

# Model IDs
DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
SAM_MODEL_ID = "facebook/sam-vit-base"

# 在 Image 構建時下載模型（只會在第一次構建時執行）
def download_models():
    from transformers import (
        AutoProcessor, 
        AutoModelForZeroShotObjectDetection,
        AutoModelForMaskGeneration
    )
    
    def download_dino():
        print("📥 Downloading Grounding DINO model during image build...")
        AutoProcessor.from_pretrained(DINO_MODEL_ID)
        AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID)
        print("✅ DINO model cached!")
        
    def download_sam():
        print("📥 Downloading SAM model during image build...")
        AutoProcessor.from_pretrained(SAM_MODEL_ID)
        AutoModelForMaskGeneration.from_pretrained(SAM_MODEL_ID)
        print("✅ SAM model cached!")
    
    # 並發下載兩個模型
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(download_dino),
            executor.submit(download_sam)
        ]
        for future in as_completed(futures):
            future.result()  # 確保沒有異常
    
    print("✅ All models cached in image!")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "transformers",
        "Pillow",
        "numpy",
        "accelerate",
        "requests",
        "fastapi",
        "opencv-python-headless",
        "timm",
        "supervision"
    )
    .run_function(download_models)  # 構建時下載模型
)

# 使用类来持久化模型
@app.cls(
    image=image,
    gpu="T4", 
    timeout=600,
    scaledown_window=60,
    enable_memory_snapshot=True
)
class GroundingSegmentator:

    @modal.enter(snap=True)
    def load_models_snapshot(self):
        """冷啟動時執行：載入模型到 CPU（模型已在 Image 中）"""
        from transformers import (
            AutoProcessor, 
            AutoModelForZeroShotObjectDetection,
            AutoModelForMaskGeneration
        )
        
        def load_dino():
            print("🔄 Loading Grounding DINO model from cached image (cold start)...")
            dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
            dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                DINO_MODEL_ID,
                torch_dtype=torch.float32
            )
            print("✅ DINO model loaded to CPU!")
            return dino_processor, dino_model
        
        def load_sam():
            print("🔄 Loading SAM model from cached image (cold start)...")
            sam_processor = AutoProcessor.from_pretrained(SAM_MODEL_ID)
            sam_model = AutoModelForMaskGeneration.from_pretrained(
                SAM_MODEL_ID,
                torch_dtype=torch.float32
            )
            print("✅ SAM model loaded to CPU!")
            return sam_processor, sam_model
        
        # 並發載入兩個模型
        with ThreadPoolExecutor(max_workers=2) as executor:
            dino_future = executor.submit(load_dino)
            sam_future = executor.submit(load_sam)
            
            self.dino_processor, self.dino_model = dino_future.result()
            self.sam_processor, self.sam_model = sam_future.result()
        
        print("✅ All models loaded to CPU, snapshot will be taken!")

    @modal.enter(snap=False)
    def move_models_to_gpu(self):
        """從快照恢復後執行：將模型移到 GPU"""
        if torch.cuda.is_available():
            print("🔄 Moving models from snapshot to GPU...")
            self.dino_model = self.dino_model.to("cuda")
            self.sam_model = self.sam_model.to("cuda")
            self.device = torch.device("cuda")
            print("✅ Models moved to GPU!")
        else:
            self.device = torch.device("cpu")
            print("⚠️ GPU not available, using CPU")

    @modal.method()
    def segment(
        self, 
        image_bytes: bytes,
        text_prompt: str,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25
    ) -> dict:
        """
        接收圖片 bytes 和文本提示，回傳 boxes 和 masks
        
        Args:
            image_bytes: 圖片的 bytes
            text_prompt: 文本提示 (e.g., 'a cat. a dog.')
            box_threshold: Grounding DINO box detection 閾值
            text_threshold: Grounding DINO text detection 閾值
            
        Returns:
            {
                "boxes": [{"label": str, "score": float, "box": [x, y, x, y]}],
                "masks": [{"label": str, "mask_base64": str, "shape": [H, W]}]
            }
        """
        # 載入圖片
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 確保 text prompt 正確格式 (以 . 結尾)
        text_prompt = text_prompt.strip()
        if not text_prompt.endswith("."):
            text_prompt = text_prompt + "."
        
        # 1. Run Grounding DINO
        print(f"🔍 Running Grounding DINO with prompt: {text_prompt}")
        dino_inputs = self.dino_processor(
            images=pil_image, 
            text=text_prompt, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            dino_outputs = self.dino_model(**dino_inputs)
        
        # Post-process DINO outputs
        results = self.dino_processor.post_process_grounded_object_detection(
            dino_outputs,
            dino_inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[pil_image.size[::-1]]
        )[0]
        
        dino_boxes = results["boxes"]  # [N, 4]
        labels = results["labels"]
        scores = results["scores"]

        if len(dino_boxes) == 0:
            print("⚠️ No objects detected.")
            return {
                "boxes": [],
                "masks": [],
                "message": "No objects detected"
            }

        # 2. Run SAM (Segment Anything)
        print(f"✂️ Running SAM on {len(dino_boxes)} boxes...")
        sam_inputs = self.sam_processor(
            images=pil_image, 
            input_boxes=[dino_boxes.tolist()], 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            sam_outputs = self.sam_model(**sam_inputs)

        # Post-process SAM masks
        sam_masks = self.sam_processor.post_process_masks(
            masks=sam_outputs.pred_masks,
            original_sizes=sam_inputs.original_sizes,
            reshaped_input_sizes=sam_inputs.reshaped_input_sizes
        )[0]
        
        # Take the first mask for each box. Shape: [N, H, W]
        final_masks = sam_masks[:, 0, :, :]

        # 3. Format Output
        output_boxes = []
        output_masks = []
        
        for idx, (box, label, score, mask_tensor) in enumerate(
            zip(dino_boxes, labels, scores, final_masks)
        ):
            # Format Box
            box_list = box.cpu().tolist()
            output_boxes.append({
                "label": label,
                "score": float(score),
                "box": box_list  # [x_min, y_min, x_max, y_max]
            })
            
            # Format Mask - convert to numpy and encode as base64
            mask_np = mask_tensor.cpu().numpy().astype(np.uint8) * 255
            
            # Encode mask as PNG in base64
            mask_pil = Image.fromarray(mask_np, mode='L')
            with io.BytesIO() as buf:
                mask_pil.save(buf, format='PNG')
                buf.seek(0)
                mask_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            output_masks.append({
                "label": label,
                "mask_base64": mask_base64,
                "shape": list(mask_np.shape)
            })

        print(f"✅ Segmentation complete! Found {len(output_boxes)} objects")
        
        return {
            "boxes": output_boxes,
            "masks": output_masks,
            "original_size": list(pil_image.size)
        }


# 建立 Web Endpoint
@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def segment_api(item: dict):
    """
    HTTP endpoint: POST with {
        "image_base64": "...",
        "text_prompt": "a cat. a dog.",
        "box_threshold": 0.3,  # optional
        "text_threshold": 0.25  # optional
    }
    """
    # Parse input
    image_base64 = item.get("image_base64", "")
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]
    
    text_prompt = item.get("text_prompt", "")
    if not text_prompt:
        return {
            "status": "error",
            "message": "text_prompt is required"
        }
    
    box_threshold = item.get("box_threshold", 0.3)
    text_threshold = item.get("text_threshold", 0.25)
    
    # Decode image
    image_bytes = base64.b64decode(image_base64)
    
    # Run segmentation
    segmentator = GroundingSegmentator()
    result = segmentator.segment.remote(
        image_bytes=image_bytes,
        text_prompt=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    
    return {"status": "success", **result}

