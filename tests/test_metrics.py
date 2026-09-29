"""Tests for crop coordinate utilities and metrics."""

import pytest
import numpy as np

from ml.metrics import compute_iou, compute_mae, evaluate_predictions


class TestIoU:
    """Test IoU computation."""

    def test_perfect_overlap(self):
        """Identical boxes should have IoU = 1.0."""
        pred = np.array([0.1, 0.1, 0.5, 0.5])
        target = np.array([0.1, 0.1, 0.5, 0.5])
        assert compute_iou(pred, target) == pytest.approx(1.0)

    def test_no_overlap(self):
        """Non-overlapping boxes should have IoU = 0.0."""
        pred = np.array([0.0, 0.0, 0.3, 0.3])
        target = np.array([0.5, 0.5, 0.3, 0.3])
        assert compute_iou(pred, target) == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Partially overlapping boxes should have 0 < IoU < 1."""
        pred = np.array([0.0, 0.0, 0.5, 0.5])
        target = np.array([0.25, 0.25, 0.5, 0.5])
        iou = compute_iou(pred, target)
        assert 0.0 < iou < 1.0
        # Intersection: 0.25*0.25 = 0.0625
        # Union: 0.25 + 0.25 - 0.0625 = 0.4375
        assert iou == pytest.approx(0.0625 / 0.4375, abs=1e-6)

    def test_contained(self):
        """When one box contains another."""
        pred = np.array([0.0, 0.0, 1.0, 1.0])
        target = np.array([0.2, 0.2, 0.3, 0.3])
        iou = compute_iou(pred, target)
        # Intersection = 0.09, Union = 1.0
        assert iou == pytest.approx(0.09, abs=1e-6)


class TestMAE:
    """Test MAE computation."""

    def test_perfect_predictions(self):
        """Zero error for identical predictions."""
        preds = np.array([[0.1, 0.2, 0.5, 0.5]])
        targets = np.array([[0.1, 0.2, 0.5, 0.5]])
        mae = compute_mae(preds, targets)
        assert mae["mae_x"] == pytest.approx(0.0)
        assert mae["mae_overall"] == pytest.approx(0.0)

    def test_known_error(self):
        """Known error values."""
        preds = np.array([[0.1, 0.2, 0.5, 0.5]])
        targets = np.array([[0.2, 0.3, 0.6, 0.7]])
        mae = compute_mae(preds, targets)
        assert mae["mae_x"] == pytest.approx(0.1)
        assert mae["mae_y"] == pytest.approx(0.1)
        assert mae["mae_width"] == pytest.approx(0.1)
        assert mae["mae_height"] == pytest.approx(0.2)


class TestCropBoundaryValidation:
    """Test that crop coordinates stay within valid bounds."""

    def test_normalized_bounds(self):
        """Normalized values must be in [0, 1]."""
        crops = [
            (100, 200, 500, 500, 1024, 1536),
            (0, 0, 1024, 1536, 1024, 1536),
        ]
        for x, y, w, h, img_w, img_h in crops:
            nx = x / img_w
            ny = y / img_h
            nw = w / img_w
            nh = h / img_h
            assert 0 <= nx <= 1
            assert 0 <= ny <= 1
            assert 0 < nw <= 1
            assert 0 < nh <= 1
            assert nx + nw <= 1.0 + 1e-6
            assert ny + nh <= 1.0 + 1e-6

    def test_crop_does_not_exceed_image(self):
        """Crop rectangle must stay within image bounds."""
        img_w, img_h = 4000, 6000
        crop_x, crop_y = 100, 200
        crop_w, crop_h = 3000, 3000

        assert crop_x >= 0
        assert crop_y >= 0
        assert crop_x + crop_w <= img_w
        assert crop_y + crop_h <= img_h


class TestEvaluatePredictions:
    """Test full evaluation pipeline."""

    def test_empty(self):
        """Empty input should return zeros."""
        result = evaluate_predictions(np.array([]).reshape(0, 4), np.array([]).reshape(0, 4))
        assert result["num_samples"] == 0
        assert result["mean_iou"] == 0.0

    def test_multiple_samples(self):
        """Evaluation with multiple samples."""
        preds = np.array([
            [0.1, 0.1, 0.5, 0.5],
            [0.2, 0.2, 0.4, 0.4],
        ])
        targets = np.array([
            [0.1, 0.1, 0.5, 0.5],
            [0.2, 0.2, 0.4, 0.4],
        ])
        result = evaluate_predictions(preds, targets)
        assert result["mean_iou"] == pytest.approx(1.0)
        assert result["mae_overall"] == pytest.approx(0.0)
        assert result["num_samples"] == 2
