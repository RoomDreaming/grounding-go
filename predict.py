# predict.py
import os
import torch
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Any
from cog import BasePredictor, Input, Path, BaseModel
from transformers import (
    AutoProcessor, 
    AutoModelForZeroShotObjectDetection,
    AutoModelForMaskGeneration
)

# 定義輸出結構
class Output(BaseModel):
    boxes: List[Dict[str, Any]]
    masks: List[Path]

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 模型 ID
        self.dino_id = "IDEA-Research/grounding-dino-base"
        self.sam_id = "facebook/sam-vit-base"
        
        print("Loading Grounding DINO...")
        self.dino_processor = AutoProcessor.from_pretrained(self.dino_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.dino_id).to(self.device)
        
        print("Loading SAM...")
        self.sam_processor = AutoProcessor.from_pretrained(self.sam_id)
        self.sam_model = AutoModelForMaskGeneration.from_pretrained(self.sam_id).to(self.device)
        
        print("Models loaded!")

    def predict(
        self,
        image: Path = Input(description="Input image"),
        text: str = Input(description="Text prompt for Grounding DINO (e.g., 'a cat. a dog.')"),
        box_threshold: float = Input(description="Threshold for Grounding DINO box detection", default=0.3),
        text_threshold: float = Input(description="Threshold for Grounding DINO text detection", default=0.25),
    ) -> Output:
        """Run a single prediction on the model"""
        
        # 1. Load and Preprocess Image
        pil_image = Image.open(image).convert("RGB")
        
        # 確保 text prompt 正確格式 (以 . 結尾)
        text = text.strip()
        if not text.endswith("."):
            text = text + "."
            
        # 2. Run Grounding DINO
        print(f"Running Grounding DINO with prompt: {text}")
        inputs = self.dino_processor(images=pil_image, text=text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
        
        # Post-process DINO outputs
        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[pil_image.size[::-1]]
        )[0]
        
        dino_boxes = results["boxes"] # [N, 4]
        labels = results["labels"]
        scores = results["scores"]

        if len(dino_boxes) == 0:
            print("No objects detected.")
            return Output(boxes=[], masks=[])

        # 3. Run SAM (Segment Anything)
        print(f"Running SAM on {len(dino_boxes)} boxes...")
        # SAM processor expects lists of lists for boxes usually, but transformers implementation handles batching
        # We need to format boxes for SAM: [x_min, y_min, x_max, y_max] -> transformers format
        
        # Prepare inputs for SAM
        # Note: transformers SAM processor expects 'input_boxes' as a list of lists of lists (batch, num_boxes, 4)
        # We are treating this as batch_size=1
        sam_inputs = self.sam_processor(
            images=pil_image, 
            input_boxes=[dino_boxes.tolist()], 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            sam_outputs = self.sam_model(**sam_inputs)

        # Post-process SAM masks
        # masks shape: (batch_size, num_boxes, height, width) -> (1, N, H, W)
        sam_masks = self.sam_processor.post_process_masks(
            masks=sam_outputs.pred_masks,
            original_sizes=sam_inputs.original_sizes,
            reshaped_input_sizes=sam_inputs.reshaped_input_sizes
        )[0] 
        
        # sam_masks is now [num_boxes, 1, H, W] usually (assuming 1 mask per box) or [num_boxes, 3, H, W]
        # We usually take the best score mask or the first one. 
        # The transformer output usually gives 3 masks per box (ambiguity). We take index 0 (highest score usually).
        # Actually post_process_masks returns Boolean Tensor.
        
        # Flatten masks
        final_masks = sam_masks[:, 0, :, :] # Take the first mask for each box. Shape: [N, H, W]

        # 4. Format Output
        output_boxes = []
        output_mask_paths = []
        
        # Clean temp directory
        out_dir = Path("output_masks")
        if os.path.exists(out_dir):
            import shutil
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        for idx, (box, label, score, mask_tensor) in enumerate(zip(dino_boxes, labels, scores, final_masks)):
            # Format Box JSON
            box_list = box.tolist() # [x, y, x, y]
            output_boxes.append({
                "label": label,
                "score": float(score),
                "box": box_list
            })
            
            # Format Mask Image
            # Mask is boolean tensor, convert to uint8 0-255
            mask_np = mask_tensor.cpu().numpy().astype(np.uint8) * 255
            
            # Create PIL image (Black background, White object)
            mask_img = Image.fromarray(mask_np, mode='L') # L = 8-bit pixels, black and white
            
            # Save
            mask_path = out_dir / f"mask_{idx}_{label}.png"
            mask_img.save(mask_path)
            output_mask_paths.append(Path(mask_path))

        return Output(boxes=output_boxes, masks=output_mask_paths)
