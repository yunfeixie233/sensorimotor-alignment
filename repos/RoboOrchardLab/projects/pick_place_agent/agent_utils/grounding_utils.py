# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer


class Grounding:
    def __init__(self, model_path, gpu_id=0):
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.model, self.tokenizer = self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer from the specified path.

        Returns:
            model: The loaded model.
            tokenizer: The loaded tokenizer.
        """
        device = f"cuda:{self.gpu_id}"
        model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map={"": device},
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=False
        )
        return model, tokenizer

    def _generate_prompts(self, object_name):
        """Generate prompts for the specified object name.

        Args:
            object_name (str): The name of the object to generate prompts for.

        Returns:
            str: The generated prompt for the object.
        """
        object_prompts = f"<image>Please respond with segmentation mask for the {object_name}"  # noqa: E501
        return object_prompts

    def _apply_oriented_bbox_mask(
        self, rgb_image, depth_image, mask, padding=10
    ):
        """Apply an oriented bounding box mask to the RGB and depth images.

        Args:
            rgb_image (np.ndarray): The raw RGB image to apply the mask to.
            depth_image (np.ndarray): The raw depth image to apply the mask to.
            mask (np.ndarray): The binary mask to apply.
            padding (int): Padding around the bounding box.

        Returns:
            tuple:
                masked_rgb (np.ndarray):
                    The RGB image with the mask applied.
                masked_depth (np.ndarray):
                    The depth image with the mask applied.
        """
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return np.zeros_like(rgb_image), np.zeros_like(depth_image)

        points = np.column_stack((x_indices, y_indices)).astype(np.float32)
        rect = cv2.minAreaRect(points)
        center, (w, h), angle = rect
        w, h = w + padding * 2, h + padding * 2
        rect_padded = (center, (w, h), angle)
        box_padded = cv2.boxPoints(rect_padded).astype(np.int32)
        oriented_mask = np.zeros_like(mask)
        cv2.fillPoly(oriented_mask, [box_padded], (1,))

        masked_rgb = rgb_image * oriented_mask[..., np.newaxis]
        masked_depth = depth_image * oriented_mask

        return masked_rgb, masked_depth

    def get_object(self, rgb_img, depth_img, object_prompt):
        """Get the object mask from the RGB and depth images.

        Get the object mask from the RGB and depth images
        based on the object prompt.

        Args:
            rgb_img (np.ndarray):
                The RGB image to process, expected to be in the range [0, 1].
            depth_img (np.ndarray):
                The depth image to process.
            object_prompt (str):
                The prompt describing the object to be masked.

        Returns:
            tuple:
                answer (str):
                    The model's response to the object prompt.
                object_masked_rgb (np.ndarray):
                    The RGB image with the object mask applied.
                object_masked_depth (np.ndarray):
                    The depth image with the object mask applied.
        """
        full_object_prompt = self._generate_prompts(object_prompt)
        rgb_img = Image.fromarray((rgb_img * 255).astype(np.uint8)).convert(
            "RGB"
        )
        input_dict = {
            "image": rgb_img,
            "text": full_object_prompt,
            "past_text": "",
            "mask_prompts": None,
            "tokenizer": self.tokenizer,
        }
        return_dict = self.model.predict_forward(**input_dict)
        masks = return_dict["prediction_masks"]

        binary_mask = np.zeros((rgb_img.height, rgb_img.width), dtype=np.uint8)
        for mask in masks:
            binary_mask = np.maximum(
                binary_mask, (mask[0] > 0).astype(np.uint8) * 255
            )
        binary_mask[binary_mask > 0] = 1

        object_masked_rgb, object_masked_depth = (
            self._apply_oriented_bbox_mask(rgb_img, depth_img, binary_mask)
        )

        return (
            (object_masked_rgb * 255).astype(np.uint8),
            object_masked_depth.astype(np.uint16),
            binary_mask,
        )


if __name__ == "__main__":
    model_path = "thirdparty/Sa2VA-4B"
    part_grounder = Grounding(model_path)
    color_image_path = "data/example_color.png"
    depth_image_path = "data/example_depth.png"
    color_image = np.array(Image.open(color_image_path).convert("RGB"))
    depth_image = np.array(Image.open(depth_image_path))
    object_prompt = "cup"
    grounding_rgb, grounding_depth, mask = part_grounder.get_object(
        color_image, depth_image, object_prompt
    )
    Image.fromarray(grounding_rgb).save("data/grounding_rgb.png")
    Image.fromarray(grounding_depth).save("data/grounding_depth.png")
    Image.fromarray(mask * 255).save("data/grounding_mask.png")

    print("Prediction Done, Result saved to data")  # noqa: E501
