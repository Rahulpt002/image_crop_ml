"""
Image preprocessing for the crop prediction model.

Handles:
- Aspect-ratio-preserving resize with padding
- Normalization using ImageNet stats
- Coordinate mapping between original and padded space
- EXIF orientation correction
"""

from typing import Tuple

import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ExifTags


# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def fix_exif_orientation(img: Image.Image) -> Image.Image:
    """
    Apply EXIF orientation tag and return correctly oriented image.

    This is critical: must be applied identically during upload,
    crop label extraction, training, and inference.
    """
    try:
        exif = img.getexif()
        if exif:
            orientation_key = None
            for key, val in ExifTags.TAGS.items():
                if val == "Orientation":
                    orientation_key = key
                    break

            if orientation_key and orientation_key in exif:
                orientation = exif[orientation_key]
                transform_map = {
                    2: Image.FLIP_LEFT_RIGHT,
                    3: Image.ROTATE_180,
                    4: Image.FLIP_TOP_BOTTOM,
                    5: lambda im: im.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90),
                    6: Image.ROTATE_270,
                    7: lambda im: im.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270),
                    8: Image.ROTATE_90,
                }
                if orientation in transform_map:
                    transform = transform_map[orientation]
                    if callable(transform) and not isinstance(transform, int):
                        img = transform(img)
                    else:
                        img = img.transpose(transform)
    except Exception:
        pass  # If EXIF fails, use image as-is

    return img


def load_image(path: str) -> Image.Image:
    """
    Load an image with EXIF correction applied.

    This is the SINGLE entry point for loading images throughout the
    ML pipeline. All code paths (training, inference, evaluation) must
    use this function to ensure EXIF is handled consistently.
    """
    img = Image.open(path)
    img = fix_exif_orientation(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


class AspectRatioResize:
    """
    Resize image preserving aspect ratio, pad to target size.

    Returns the transformed image AND the padding metadata needed
    to map coordinates back to the original image space.

    IMPORTANT: The crop labels (normalized 0–1 relative to original)
    are NOT transformed — they stay in the original coordinate space.
    The model learns the mapping: padded_image → original_coords.
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
    We only apply safe augmentations (color jitter) which modify
    pixel values but NOT spatial layout.
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

    Uses load_image() to ensure EXIF is handled consistently.

    Returns (tensor [1, 3, H, W], metadata dict).
    """
    img = load_image(image_path)

    resizer = AspectRatioResize(target_size=image_size)
    padded, metadata = resizer(img)

    transform = get_val_transform(image_size)
    tensor = transform(padded).unsqueeze(0)  # Add batch dim

    return tensor, metadata


def get_preprocessing_config(image_size: int = 224) -> dict:
    """
    Return a serializable dict describing the full preprocessing pipeline.

    This is saved alongside model checkpoints so the exact preprocessing
    can be reproduced at inference time.
    """
    return {
        "image_size": image_size,
        "resize_mode": "aspect_ratio_preserving_with_padding",
        "pad_fill": [0, 0, 0],
        "normalization": {
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        },
        "augmentation_training": {
            "color_jitter": {
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.1,
                "hue": 0.05,
            },
        },
        "augmentation_inference": None,
        "coordinate_system": "normalized_0_1_relative_to_original",
    }
