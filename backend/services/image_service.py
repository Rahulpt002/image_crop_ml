"""Image processing service: upload, thumbnail, EXIF handling."""

import hashlib
import logging
import shutil
import uuid
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ExifTags

from backend.config.settings import (
    UPLOAD_DIR, DATASET_IMAGES_DIR, THUMBNAIL_SIZE, ALLOWED_EXTENSIONS, MAX_IMAGE_SIZE,
    PORTRAIT_OUTPUT_WIDTH, PORTRAIT_OUTPUT_HEIGHT, PORTRAIT_OUTPUT_DPI,
)

logger = logging.getLogger(__name__)


def validate_image_file(filename: str, file_size: int) -> None:
    """Validate an uploaded image file."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}")
    if file_size > MAX_IMAGE_SIZE:
        raise ValueError(f"File too large: {file_size} bytes. Max: {MAX_IMAGE_SIZE}")


def fix_exif_orientation(img: Image.Image) -> Image.Image:
    """Apply EXIF orientation tag and return correctly oriented image."""
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
                transforms = {
                    2: Image.FLIP_LEFT_RIGHT,
                    3: Image.ROTATE_180,
                    4: Image.FLIP_TOP_BOTTOM,
                    5: lambda im: im.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90),
                    6: Image.ROTATE_270,
                    7: lambda im: im.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270),
                    8: Image.ROTATE_90,
                }
                if orientation in transforms:
                    transform = transforms[orientation]
                    if callable(transform) and not isinstance(transform, int):
                        img = transform(img)
                    else:
                        img = img.transpose(transform)
    except Exception as e:
        logger.warning(f"Could not process EXIF orientation: {e}")

    return img


def save_uploaded_image(
    file_content: bytes,
    original_filename: str,
    subfolder: str = "originals",
) -> Tuple[str, Path, int, int]:
    """
    Save an uploaded image to disk, fix EXIF, return (unique_filename, path, width, height).
    """
    ext = Path(original_filename).suffix.lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"

    target_dir = UPLOAD_DIR / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / unique_name

    # Save raw bytes first
    target_path.write_bytes(file_content)

    # Re-open, fix EXIF, get dimensions, re-save
    with Image.open(target_path) as img:
        img = fix_exif_orientation(img)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        width, height = img.size
        img.save(target_path, quality=95)

    return unique_name, target_path, width, height


def create_thumbnail(image_path: Path, size: Tuple[int, int] = THUMBNAIL_SIZE) -> Path:
    """Create a thumbnail for an image, return the thumbnail path."""
    thumb_dir = image_path.parent / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = thumb_dir / image_path.name

    with Image.open(image_path) as img:
        img.thumbnail(size, Image.LANCZOS)
        img.save(thumb_path, quality=85)

    return thumb_path


def load_image_as_array(image_path: Path) -> np.ndarray:
    """Load an image as a NumPy array (RGB)."""
    with Image.open(image_path) as img:
        img = fix_exif_orientation(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img)


def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
    """Return (width, height) of an image."""
    with Image.open(image_path) as img:
        img = fix_exif_orientation(img)
        return img.size


def crop_image(image_path: Path, x: int, y: int, width: int, height: int) -> Image.Image:
    """Crop an image and return the cropped PIL Image."""
    with Image.open(image_path) as img:
        img = fix_exif_orientation(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img.crop((x, y, x + width, y + height))


def copy_to_dataset(
    source_path: Path,
    example_id: str,
    image_type: str,  # "original" or "target"
) -> Path:
    """Copy an image into the dataset directory structure."""
    dest_dir = DATASET_IMAGES_DIR / example_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = source_path.suffix
    dest_path = dest_dir / f"{image_type}{ext}"
    shutil.copy2(source_path, dest_path)
    return dest_path


def export_portrait_crop(
    image_path: Path,
    x: int,
    y: int,
    width: int,
    height: int,
    output_path: Path,
    target_width: int = PORTRAIT_OUTPUT_WIDTH,
    target_height: int = PORTRAIT_OUTPUT_HEIGHT,
    dpi: Tuple[int, int] = PORTRAIT_OUTPUT_DPI,
) -> Path:
    """
    Extract crop from original image, resize to target dimensions (150x200),
    and save with specified DPI metadata (300x300 DPI).

    Args:
        image_path: Source image path
        x, y, width, height: Bounding box in absolute source pixels
        output_path: Destination file path
        target_width: Final output width (default 150 px)
        target_height: Final output height (default 200 px)
        dpi: Metadata resolution tuple (default (300, 300))

    Returns:
        The output_path
    """
    with Image.open(image_path) as img:
        img = fix_exif_orientation(img)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Clamp crop coordinates to image bounds
        img_w, img_h = img.size
        x1 = max(0, min(x, img_w - 1))
        y1 = max(0, min(y, img_h - 1))
        x2 = max(x1 + 1, min(x + width, img_w))
        y2 = max(y1 + 1, min(y + height, img_h))

        cropped = img.crop((x1, y1, x2, y2))
        resized = cropped.resize((target_width, target_height), Image.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resized.save(output_path, quality=95, dpi=dpi)

    return output_path
