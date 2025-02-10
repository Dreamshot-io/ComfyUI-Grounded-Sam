# ComfyUI Grounded SAM Replicate API Node

This custom node for ComfyUI allows you to integrate the Grounded SAM (Segment Anything Model) through Replicate's API directly into your ComfyUI workflows.

## Description

The Grounded SAM Replicate API node enables semantic segmentation of images using text prompts. It connects to the Grounded SAM model hosted on Replicate's platform, allowing you to:

- Generate segmentation masks based on text prompts
- Specify both positive and negative text prompts for precise segmentation
- Adjust the segmentation sensitivity with an adjustment factor
- Get multiple mask outputs for further processing in ComfyUI

## Installation

1. Navigate to your ComfyUI custom nodes directory
2. Clone this repository or copy the files into a new directory
3. Restart ComfyUI

## Requirements

- A Replicate API token (get one at https://replicate.com)
- ComfyUI installation
- Required Python packages: `torch`, `numpy`, `Pillow`, `aiohttp`

## Usage

1. Add the "Grounded SAM (Replicate)" node to your workflow
2. Connect an image input
3. Provide your Replicate API token
4. Set your mask prompts:
   - `mask_prompt`: Words describing what you want to segment (e.g., "clothes,shoes")
   - `negative_mask_prompt`: Words describing what to exclude (e.g., "pants")
   - `adjustment_factor`: Fine-tune the segmentation (-50 to 50)

### Outputs

The node provides four output masks:

1. Negative segmentation
2. Positive segmentation
3. Image mask
4. Inverted mask

Add this to your ComfyUI config.json file:

```json
{
  "git_custom_nodes": {
    "https://github.com/Dreamshot-io/ComfyUI-Grounded-Sam": {
      "hash": "13292e95cf62ff925f59b6f038e816c5eb23298f",
      "disabled": false
    }
  }
}
```
