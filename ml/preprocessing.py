"""
Image preprocessing for the crop prediction model.

Handles:
- Aspect-ratio-preserving resize with padding
- Normalization using ImageNet stats
- Coordinate mapping between original and padded space
"""

from typing import Tuple

import numpy as np
import torch
from torchvision import transforms
from PIL import Image


# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AspectRatioResize:
    """
    Resize image preserving aspect ratio, pad to target size.
    
    Returns the transformed image AND the padding metadata needed
    to map coordinates back to the original image space.
    """

    def __init__(self, target_size: int = 224, fill: Tuple[int, ...] = (0, 0, 0)):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Tuple[Image.Image, dict]:
        """
        Resize and pad, return (padded_image, metadata).
        
        Metadata contains:
            scale: float — scale factor applied
            pad_left: int — left padding
            pad_top: int — top padding
            original_w: int
            original_h: int
            resized_w: int
            resized_h: int
        """
        orig_w, orig_h = img.size

        # Compute scale to fit within target_size
        scale = self.target_size / max(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Resize
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)

        # Pad to target_size × target_size
        pad_left = (self.target_size - new_w) // 2
        pad_top = (self.target_size - new_h) // 2

        padded = Image.new("RGB", (self.target_size, self.target_size), self.fill)
        padded.paste(img_resized, (pad_left, pad_top))

        metadata = {
            "scale": scale,
            "pad_left": pad_left,
            "pad_top": pad_top,
            "original_w": orig_w,
            "original_h": orig_h,
            "resized_w": new_w,
            "resized_h": new_h,
        }

        return padded, metadata


def get_train_transform(image_size: int = 224) -> transforms.Compose:
    """
    Training transform pipeline.
    
    Note: We do NOT apply random crops or flips that would change
    the spatial relationship — the crop coordinates must stay valid.
    We only apply safe augmentations (color jitter, etc).
    """
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform(image_size: int = 224) -> transforms.Compose:
    """Validation/inference transform pipeline (no augmentation)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_for_inference(
    image_path: str,
    image_size: int = 224,
) -> Tuple[torch.Tensor, dict]:
    """
    Preprocess a single image for inference.
    
    Returns (tensor [1, 3, H, W], metadata dict).
    """
    img = Image.open(image_path).convert("RGB")

    resizer = AspectRatioResize(target_size=image_size)
    padded, metadata = resizer(img)

    transform = get_val_transform(image_size)
    tensor = transform(padded).unsqueeze(0)  # Add batch dim

    return tensor, metadata
