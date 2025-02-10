"""
@author: dreamshot.io
@title: Grounded SAM Replicate API
@nickname: Grounded SAM API
@description: This extension provides Grounded SAM segmentation through Replicate's API for ComfyUI.
"""

print("Loading Replicate API Node...")  # Add debug print

import os
import torch
import numpy as np
from PIL import Image
import io
import base64
import aiohttp
import json
import asyncio
from server import PromptServer
from aiohttp import web

class GroundedSamReplicateAPI:
    def __init__(self):
        self.api_url = "https://api.replicate.com/v1/predictions"
        self.model_version = "ee871c19efb1941f55f66a3d7d960428c8a5afcb77449547fe8e5a3ab9ebc21c"
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "api_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "mask_prompt": ("STRING", {
                    "default": "clothes,shoes",
                    "multiline": False,
                }), 
                "negative_mask_prompt": ("STRING", {
                    "default": "pants",
                    "multiline": False,
                }),
                "adjustment_factor": ("INT", {
                    "default": 0,
                    "min": -50,
                    "max": 50,
                    "step": 1,
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("negative_segmentation", "positive_segmentation", "image_mask", "inverted_mask")
    FUNCTION = "call_replicate_api"
    CATEGORY = "API/Replicate"

    def tensor_to_base64(self, image_tensor):
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]
        
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        if image_np.shape[-1] == 3:
            pil_image = Image.fromarray(image_np, 'RGB')
        else:
            pil_image = Image.fromarray(image_np[:, :, 0], 'L')
        
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def base64_to_tensor(self, base64_string):
        image_data = base64.b64decode(base64_string)
        pil_image = Image.open(io.BytesIO(image_data))
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        tensor_image = torch.from_numpy(np_image)[None,...]
        return tensor_image

    def call_replicate_api(self, images, api_token, mask_prompt, negative_mask_prompt, adjustment_factor):
        try:
            if not api_token:
                raise ValueError("Replicate API token is required")
            
            # Ensure adjustment_factor is an integer
            adjustment_factor = int(adjustment_factor)
            
            # Convert input image to base64
            image_base64 = self.tensor_to_base64(images)
            
            # Prepare request data
            data = {
                "version": self.model_version,
                "input": {
                    "image": f"data:image/png;base64,{image_base64}",
                    "mask_prompt": str(mask_prompt),
                    "negative_mask_prompt": str(negative_mask_prompt),
                    "adjustment_factor": adjustment_factor
                }
            }
            
            # Use synchronous requests for now (we'll make it async later if needed)
            import requests
            
            # Initial prediction request
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {api_token}",
                    "Content-Type": "application/json"
                },
                json=data
            )
            
            if response.status_code != 201:
                raise ValueError(f"Replicate API error: {response.text}")
            
            prediction = response.json()
            get_url = prediction['urls']['get']
            
            # Poll for completion
            while True:
                response = requests.get(
                    get_url,
                    headers={"Authorization": f"Bearer {api_token}"}
                )
                status_data = response.json()
                
                if status_data['status'] == 'succeeded':
                    output_urls = status_data['output']
                    if not isinstance(output_urls, list):
                        output_urls = [output_urls]
                    
                    # Process all output images
                    output_tensors = []
                    for url in output_urls:
                        img_response = requests.get(url)
                        img_base64 = base64.b64encode(img_response.content).decode()
                        output_tensors.append(self.base64_to_tensor(img_base64))
                    
                    # Ensure we have at least one output
                    if not output_tensors:
                        raise ValueError("No output images received from API")
                    
                    # Reorder outputs to get the properly segmented image first
                    if len(output_tensors) >= 2:
                        output_tensors[0], output_tensors[1] = output_tensors[1], output_tensors[0]
                        
                    # Pad remaining outputs if needed
                    while len(output_tensors) < 4:
                        output_tensors.append(output_tensors[0])
                        
                    return tuple(output_tensors[:4])
                
                elif status_data['status'] == 'failed':
                    raise ValueError(f"Prediction failed: {status_data.get('error', 'Unknown error')}")
                
                # Wait before polling again
                import time
                time.sleep(1)
                            
        except Exception as e:
            raise Exception(f"Error in Replicate API call: {str(e)}")

# Register nodes
NODE_CLASS_MAPPINGS = {
    "GroundedSamReplicateAPI": GroundedSamReplicateAPI
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GroundedSamReplicateAPI": "Grounded SAM (Replicate)"
}