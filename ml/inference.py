"""Inference module for crop prediction."""

import logging
from typing import Dict, Any, Optional
from PIL import Image

import torch

from ml.model import CropNet, build_model, get_device, BACKBONE_REGISTRY
from ml.preprocessing import fix_exif_orientation, preprocess_for_inference
from ml.face_detector import get_face_detector
from backend.config.settings import PORTRAIT_ASPECT_RATIO

logger = logging.getLogger(__name__)

# Cache for loaded models
_model_cache: Dict[str, CropNet] = {}


def predict_crop(
    image_path: str,
    model_path: str,
    image_size: int = 224,
    backbone: str = "mobilenet_v3_small",
    aspect_ratio: Optional[float] = None,
    image_aspect_ratio: Optional[float] = None,
) -> Dict[str, float]:
    """
    Predict crop rectangle for a single image.
    
    Args:
        image_path: Path to the input image
        model_path: Path to the model weights (.pt)
        image_size: Model input size
        backbone: Model backbone architecture
        aspect_ratio: Optional target physical crop aspect ratio (e.g., 0.75 for 3:4)
        image_aspect_ratio: Optional image width/height ratio for normalizing AR
    
    Returns:
        Dict with normalized crop coordinates:
            x, y, width, height (all 0–1)
    """
    device = get_device()

    # Load model (with caching)
    model = _get_model(model_path, backbone, device)

    # Preprocess image
    tensor, metadata = preprocess_for_inference(image_path, image_size)
    tensor = tensor.to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        pred = model.predict_with_validation(
            tensor,
            aspect_ratio=aspect_ratio,
            image_aspect_ratio=image_aspect_ratio,
        )

    # Extract values
    pred = pred.cpu().numpy()[0]

    result = {
        "x": float(pred[0]),
        "y": float(pred[1]),
        "width": float(pred[2]),
        "height": float(pred[3]),
    }

    logger.info(f"Prediction: x={result['x']:.3f}, y={result['y']:.3f}, "
                f"w={result['width']:.3f}, h={result['height']:.3f}")

    return result


def predict_portrait_crop(
    image_path: str,
    model_path: str,
    image_size: int = 224,
    backbone: str = "mobilenet_v3_small",
    aspect_ratio: float = PORTRAIT_ASPECT_RATIO,
    face_detector_backend: str = "mediapipe",
) -> Dict[str, Any]:
    """
    Run the full portrait crop pipeline:
    1. EXIF orientation correction
    2. Face detection
    3. CropNet prediction constrained to target aspect ratio (3:4)
    4. Coordinate validation
    5. Returns coordinates, face bounding box, and aspect ratio.

    Args:
        image_path: Path to input image
        model_path: Path to trained CropNet model weights (.pt)
        image_size: Model input resolution
        backbone: Backbone architecture
        aspect_ratio: Target crop aspect ratio (default 3:4 = 0.75)
        face_detector_backend: 'mediapipe' or 'opencv'

    Returns:
        Dict with keys:
            x, y, width, height (normalized 0–1)
            face_detected (bool)
            face_box (dict | None): x, y, width, height in absolute pixels
            face_confidence (float | None)
            aspect_ratio (float): actual resulting physical aspect ratio
    """
    # 1. Load image with EXIF correction to get true dimensions
    with Image.open(image_path) as img:
        img = fix_exif_orientation(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        orig_w, orig_h = img.size

        # 2. Run face detection
        face_detected = False
        face_box = None
        face_confidence = None
        try:
            detector = get_face_detector(backend=face_detector_backend)
            primary_face = detector.detect_primary(img)
            if primary_face:
                face_detected = True
                face_confidence = float(primary_face.confidence)
                face_box = {
                    "x": int(primary_face.x),
                    "y": int(primary_face.y),
                    "width": int(primary_face.width),
                    "height": int(primary_face.height),
                }
        except Exception as e:
            logger.warning(f"Face detection in portrait pipeline encountered an issue: {e}")

    # 3. Model prediction constrained to 3:4
    img_ar = orig_w / orig_h if orig_h > 0 else 1.0
    crop = predict_crop(
        image_path=image_path,
        model_path=model_path,
        image_size=image_size,
        backbone=backbone,
        aspect_ratio=aspect_ratio,
        image_aspect_ratio=img_ar,
    )

    # 4. Resulting aspect ratio
    phys_w = crop["width"] * orig_w
    phys_h = crop["height"] * orig_h
    actual_ar = phys_w / phys_h if phys_h > 0 else aspect_ratio

    return {
        "x": crop["x"],
        "y": crop["y"],
        "width": crop["width"],
        "height": crop["height"],
        "face_detected": face_detected,
        "face_box": face_box,
        "face_confidence": face_confidence,
        "aspect_ratio": float(actual_ar),
    }


def _get_model(model_path: str, backbone: str, device: torch.device) -> CropNet:
    """Load model from cache or disk."""
    cache_key = model_path

    if cache_key not in _model_cache:
        model = build_model(backbone=backbone, pretrained=False)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        _model_cache[cache_key] = model
        logger.info(f"Loaded model from {model_path} on {device}")

    return _model_cache[cache_key]


def clear_model_cache():
    """Clear the model cache (e.g., after retraining)."""
    _model_cache.clear()
