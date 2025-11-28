# Grounded SAM - Object Detection & Segmentation

Object detection and segmentation API based on [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) and [Segment Anything](https://github.com/facebookresearch/segment-anything).

## Features

Input an image and text prompts to automatically detect and segment objects. Returns:

- **JSON formatted bounding boxes** - Contains coordinates and labels for each detected object
- **Individual mask images** - Each detected object has a corresponding black background with white mask

Supports positive and negative prompts for precise segmentation control.

## Credits

This project is based on and modified from [@schananas/grounded_sam](https://replicate.com/schananas/grounded_sam), with enhanced output format to support more flexible use cases.

Original papers:

- [Grounding DINO](https://arxiv.org/abs/2303.05499) - IDEA Research
- [Segment Anything](https://arxiv.org/abs/2304.02643) - Meta AI
