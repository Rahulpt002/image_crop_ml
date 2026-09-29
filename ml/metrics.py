"""Evaluation metrics for crop prediction."""

from typing import List, Dict

import torch
import numpy as np


def compute_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute IoU between predicted and target crop rectangles.
    
    Both inputs are [x, y, width, height] normalized.
    """
    # Convert to [x1, y1, x2, y2]
    pred_x1, pred_y1 = pred[0], pred[1]
    pred_x2, pred_y2 = pred[0] + pred[2], pred[1] + pred[3]
    
    tgt_x1, tgt_y1 = target[0], target[1]
    tgt_x2, tgt_y2 = target[0] + target[2], target[1] + target[3]

    # Intersection
    inter_x1 = max(pred_x1, tgt_x1)
    inter_y1 = max(pred_y1, tgt_y1)
    inter_x2 = min(pred_x2, tgt_x2)
    inter_y2 = min(pred_y2, tgt_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Union
    pred_area = pred[2] * pred[3]
    tgt_area = target[2] * target[3]
    union_area = pred_area + tgt_area - inter_area

    if union_area <= 0:
        return 0.0

    return float(inter_area / union_area)


def compute_mae(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Compute Mean Absolute Error per coordinate.
    
    Args:
        predictions: [N, 4] array of predicted crops
        targets: [N, 4] array of target crops
    
    Returns:
        Dict with mae_x, mae_y, mae_width, mae_height, mae_overall
    """
    errors = np.abs(predictions - targets)
    return {
        "mae_x": float(np.mean(errors[:, 0])),
        "mae_y": float(np.mean(errors[:, 1])),
        "mae_width": float(np.mean(errors[:, 2])),
        "mae_height": float(np.mean(errors[:, 3])),
        "mae_overall": float(np.mean(errors)),
    }


def evaluate_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    Full evaluation: IoU + MAE metrics.
    
    Args:
        predictions: [N, 4] normalized predicted crops
        targets: [N, 4] normalized target crops
    
    Returns:
        Dict with all metrics
    """
    n = len(predictions)
    if n == 0:
        return {
            "mean_iou": 0.0,
            "median_iou": 0.0,
            "mae_x": 0.0,
            "mae_y": 0.0,
            "mae_width": 0.0,
            "mae_height": 0.0,
            "mae_overall": 0.0,
            "num_samples": 0,
        }

    # IoU for each sample
    ious = [compute_iou(predictions[i], targets[i]) for i in range(n)]

    # MAE
    mae_metrics = compute_mae(predictions, targets)

    return {
        "mean_iou": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
        **mae_metrics,
        "num_samples": n,
    }
