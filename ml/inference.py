"""Inference module for crop prediction."""

import logging
from typing import Dict

import torch

from ml.model import CropNet, build_model, get_device, BACKBONE_REGISTRY
from ml.preprocessing import preprocess_for_inference

logger = logging.getLogger(__name__)

# Cache for loaded models
_model_cache: Dict[str, CropNet] = {}


def predict_crop(
    image_path: str,
    model_path: str,
    image_size: int = 224,
    backbone: str = "mobilenet_v3_small",
) -> Dict[str, float]:
    """
    Predict crop rectangle for a single image.
    
    Args:
        image_path: Path to the input image
        model_path: Path to the model weights (.pt)
        image_size: Model input size
        backbone: Model backbone architecture
    
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
        pred = model.predict_with_validation(tensor)

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
