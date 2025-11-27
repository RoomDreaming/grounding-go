
from PIL import Image, ImageDraw, ImageFont
import sys
sys.path.insert(0, "weights/Grounded-Segment-Anything/GroundingDINO")
sys.path.insert(0, "weights/Grounded-Segment-Anything/segment_anything")

from groundingdino.util import box_ops
from groundingdino.util.inference import annotate, load_image, predict

import numpy as np
import torch
import cv2

def adjust_mask(mask, adjustment_factor):
    mask = mask.astype(np.uint8)

    if adjustment_factor == 0:  # Just return the mask as is if adjustment factor is 0
        return mask

    if adjustment_factor < 0:
        mask = cv2.erode(
            mask,
            np.ones((abs(adjustment_factor), abs(adjustment_factor)), np.uint8),
            iterations=1
        )

    if adjustment_factor > 0:
        mask = cv2.dilate(
            mask,
            np.ones((adjustment_factor, adjustment_factor), np.uint8),
            iterations=1
        )

    return mask

def detect(image,image_src, text_prompt, model, box_threshold = 0.3, text_threshold = 0.25):
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    annotated_frame = annotate(image_source=image_src, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB
    return annotated_frame, boxes

def segment(image, sam_model, boxes):
    sam_model.set_image(image)
    H, W, _ = image.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(device), image.shape[:2])
    masks, _, _ = sam_model.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes,
        multimask_output = False,
        )
    return masks.cpu()

def draw_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


def run_grounding_sam(local_image_path, positive_prompt, negative_prompt, groundingdino_model, sam_predictor,
                      adjustment_factor):
    image_source, image = load_image(local_image_path)

    annotated_frame, detected_boxes = detect(image, image_source, positive_prompt, groundingdino_model)

    segmented_frame_masks = segment(image_source, sam_predictor, boxes=detected_boxes)

    # Get box coordinates and labels
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(detected_boxes) * torch.Tensor([W, H, W, H])
    
    # Split prompt into individual labels
    labels = [label.strip() for label in positive_prompt.split(',')]
    
    # Create list to store individual masks and bounding box info
    individual_masks = []
    bounding_boxes = []
    
    for idx, (box, mask) in enumerate(zip(boxes_xyxy, segmented_frame_masks)):
        # Get individual mask
        single_mask = mask[0].cpu().numpy()
        
        # Apply adjustment factor
        mask_uint8 = (single_mask * 255).astype(np.uint8)
        adjusted_mask = adjust_mask(mask_uint8, adjustment_factor)
        
        # If negative_prompt is defined, subtract negative regions
        if negative_prompt:
            neg_annotated_frame, neg_detected_boxes = detect(image, image_source, negative_prompt, groundingdino_model)
            if len(neg_detected_boxes) > 0:
                neg_segmented_frame_masks = segment(image_source, sam_predictor, boxes=neg_detected_boxes)
                merged_neg_mask = np.logical_or.reduce(neg_segmented_frame_masks[:, 0])
                neg_mask = (merged_neg_mask.cpu().numpy() * 255).astype(np.uint8)
                adjusted_mask = adjusted_mask & ~neg_mask
        
        individual_masks.append(Image.fromarray(adjusted_mask))
        
        # Store bounding box info
        label = labels[idx % len(labels)]  # Cycle through labels if more boxes than labels
        bounding_boxes.append({
            'label': label,
            'box': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],  # [x1, y1, x2, y2]
            'index': idx
        })
    
    return individual_masks, bounding_boxes