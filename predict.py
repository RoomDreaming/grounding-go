import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Iterator
import uuid

from cog import BasePredictor, Input, Path as CogPath

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything
from segment_anything import sam_model_registry, SamPredictor

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Paths defined in cog.yaml
        self.grounding_dino_config_path = "weights/GroundingDINO_SwinT_OGC.py"
        self.grounding_dino_checkpoint_path = "weights/groundingdino_swint_ogc.pth"
        self.sam_checkpoint_path = "weights/sam_vit_h_4b8939.pth"
        
        print("Loading GroundingDINO...")
        self.grounding_dino_model = self.load_grounding_dino(
            self.grounding_dino_config_path, 
            self.grounding_dino_checkpoint_path, 
            self.device
        )
        
        print("Loading SAM...")
        sam = sam_model_registry["vit_h"](checkpoint=self.sam_checkpoint_path)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)
        
    def load_grounding_dino(self, model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(f"GroundingDINO loaded: {load_res}")
        model.eval()
        return model.to(device)

    def transform_image(self, image_pil):
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(image_pil, None)
        return image

    def get_grounding_output(self, model, image, caption, box_threshold, text_threshold, device="cuda"):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
            
        model = model.to(device)
        image = image.to(device)
        
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
            
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        
        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]
        
        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            
        return boxes_filt, pred_phrases

    def adjust_mask(self, mask, adjustment_factor):
        """
        Adjust mask by eroding (negative factor) or dilating (positive factor).
        factor is roughly pixels.
        """
        if adjustment_factor == 0:
            return mask
            
        kernel_size = int(abs(adjustment_factor)) * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        if adjustment_factor > 0:
            # Dilation
            adjusted = cv2.dilate(mask_uint8, kernel, iterations=1)
        else:
            # Erosion
            adjusted = cv2.erode(mask_uint8, kernel, iterations=1)
            
        return adjusted > 127

    @torch.inference_mode()
    def predict(
            self,
            image: CogPath = Input(
                description="Image input",
                default=None,
            ),
            mask_prompt: str = Input(
                description="Positive mask prompt (what to segment)",
                default="clothes, shoes",
            ),
            negative_mask_prompt: str = Input(
                description="Negative mask prompt (objects to avoid/exclude in detection logic if applicable, primarily used for context logic here)",
                default=None,
            ),
            box_threshold: float = Input(
                description="Box threshold for Grounding DINO",
                default=0.3,
            ),
            text_threshold: float = Input(
                description="Text threshold for Grounding DINO",
                default=0.25,
            ),
            adjustment_factor: int = Input(
                description="Mask Adjustment Factor (-ve for erosion, +ve for dilation in pixels)",
                default=0,
            ),
    ) -> Iterator[CogPath]:
        """Run a single prediction on the model"""
        if image is None:
            raise ValueError("Please provide an input image.")

        predict_id = str(uuid.uuid4())
        print(f"Running prediction: {predict_id}...")
        
        # Load Image
        image_pil = Image.open(image).convert("RGB")
        image_cv = np.array(image_pil)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        
        # Run Grounding DINO
        transformed_image = self.transform_image(image_pil)
        boxes_filt, pred_phrases = self.get_grounding_output(
            self.grounding_dino_model, 
            transformed_image, 
            mask_prompt, 
            box_threshold, 
            text_threshold, 
            device=self.device
        )

        # Process boxes for SAM
        H, W, _ = image_cv.shape
        boxes_xyxy = boxes_filt * torch.Tensor([W, H, W, H])
        boxes_xyxy[:, :2] -= boxes_xyxy[:, 2:] / 2
        boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]

        boxes_xyxy = boxes_xyxy.cpu()
        
        # Run SAM
        self.sam_predictor.set_image(np.array(image_pil))
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy.to(self.device), (H, W))
        
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        
        # Output directory
        output_dir = f"/tmp/{predict_id}"
        os.makedirs(output_dir, exist_ok=True)

        bounding_boxes_output = []
        
        # Save results
        for idx, (mask, box, phrase) in enumerate(zip(masks, boxes_xyxy, pred_phrases)):
            # Convert mask to numpy (H, W)
            mask_np = mask[0].cpu().numpy()
            
            # Adjustment (Erosion/Dilation)
            if adjustment_factor != 0:
                mask_np = self.adjust_mask(mask_np, adjustment_factor)
            
            # Save Mask Image
            label_clean = phrase.split('(')[0].strip().replace(' ', '_')
            mask_filename = f"{output_dir}/mask_{idx}_{label_clean}.png"
            
            # Create a black and white mask image
            mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
            mask_img.save(mask_filename)
            yield CogPath(mask_filename)
            
            # Collect Box Info
            bounding_boxes_output.append({
                "id": idx,
                "label": phrase,
                "box": box.tolist(), # [x1, y1, x2, y2]
                "mask_file": os.path.basename(mask_filename)
            })

        # Save JSON
        json_filename = f"{output_dir}/bounding_boxes.json"
        with open(json_filename, 'w') as f:
            json.dump(bounding_boxes_output, f, indent=2)
        yield CogPath(json_filename)

        print("Done!")
