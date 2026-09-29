"""
Comprehensive ML pipeline audit tests.

Covers:
1. Coordinate transformation correctness
2. Aspect-ratio resize + padding validation
3. Model output → original coordinate mapping
4. EXIF consistency
5. Augmentation safety
6. Sigmoid + boundary validation
7. IoU coordinate system consistency
8. Split integrity (no data leakage)
9. Checkpoint completeness
10. Synthetic dataset end-to-end test
"""

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch
from PIL import Image

from ml.preprocessing import (
    AspectRatioResize,
    get_train_transform,
    get_val_transform,
    preprocess_for_inference,
    load_image,
    fix_exif_orientation,
    get_preprocessing_config,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from ml.model import CropNet, build_model, get_device
from ml.dataset import CropDataset
from ml.metrics import compute_iou, compute_mae, evaluate_predictions


# ─── Helpers ──────────────────────────────────────────────────────────

def create_synthetic_image(
    width: int, height: int,
    crop_x: float, crop_y: float,
    crop_w: float, crop_h: float,
    tmpdir: str,
) -> Tuple[str, dict]:
    """
    Create a synthetic image with a known crop region.

    The crop region is drawn as a bright colored rectangle on a dark background,
    so it's visually distinguishable.

    Returns (path, crop_dict).
    """
    img = Image.new("RGB", (width, height), (30, 30, 40))

    # Draw the crop region with a distinctive pattern
    px_x = int(crop_x * width)
    px_y = int(crop_y * height)
    px_w = int(crop_w * width)
    px_h = int(crop_h * height)

    pixels = img.load()
    for y in range(px_y, min(px_y + px_h, height)):
        for x in range(px_x, min(px_x + px_w, width)):
            # Create a gradient pattern inside the crop region
            r = int(100 + 155 * (x - px_x) / max(px_w, 1))
            g = int(80 + 175 * (y - px_y) / max(px_h, 1))
            b = 120
            pixels[x, y] = (r, g, b)

    path = os.path.join(tmpdir, f"synth_{width}x{height}_{crop_x:.2f}.jpg")
    img.save(path, quality=95)

    return path, {
        "original_path": path,
        "norm_x": crop_x,
        "norm_y": crop_y,
        "norm_width": crop_w,
        "norm_height": crop_h,
    }


# ═══════════════════════════════════════════════════════════════════════
# 1. COORDINATE TRANSFORMATION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestCoordinateTransformations:
    """Verify preprocessing doesn't destroy the image↔coordinate relationship."""

    def test_aspect_ratio_resize_preserves_ratio(self):
        """Aspect ratio must be preserved after resize."""
        img = Image.new("RGB", (800, 600))
        resizer = AspectRatioResize(target_size=224)
        padded, meta = resizer(img)

        assert padded.size == (224, 224)
        # Original is 4:3, resized should maintain that
        assert abs(meta["resized_w"] / meta["resized_h"] - 800 / 600) < 0.02

    def test_resize_landscape(self):
        """Landscape image: width is the constraining dimension."""
        img = Image.new("RGB", (1000, 500))
        resizer = AspectRatioResize(target_size=224)
        padded, meta = resizer(img)

        # Width should be 224, height = 224 * 500/1000 = 112
        assert meta["resized_w"] == 224
        assert meta["resized_h"] == 112
        assert meta["pad_left"] == 0
        assert meta["pad_top"] == (224 - 112) // 2

    def test_resize_portrait(self):
        """Portrait image: height is the constraining dimension."""
        img = Image.new("RGB", (500, 1000))
        resizer = AspectRatioResize(target_size=224)
        padded, meta = resizer(img)

        # Height should be 224, width = 224 * 500/1000 = 112
        assert meta["resized_h"] == 224
        assert meta["resized_w"] == 112
        assert meta["pad_top"] == 0
        assert meta["pad_left"] == (224 - 112) // 2

    def test_resize_square(self):
        """Square image: no padding needed."""
        img = Image.new("RGB", (1000, 1000))
        resizer = AspectRatioResize(target_size=224)
        padded, meta = resizer(img)

        assert meta["resized_w"] == 224
        assert meta["resized_h"] == 224
        assert meta["pad_left"] == 0
        assert meta["pad_top"] == 0

    def test_crop_labels_not_transformed(self):
        """
        CRITICAL: Crop labels must remain in original normalized space.

        The model learns: padded_image → original_normalized_coords.
        Labels should NOT be adjusted for padding.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a non-square image
            path, crop_dict = create_synthetic_image(
                800, 600, 0.1, 0.1, 0.8, 0.8, tmpdir
            )

            dataset = CropDataset([crop_dict], image_size=224, is_training=False)
            tensor, label = dataset[0]

            # Label should be the original normalized coords, NOT adjusted for padding
            assert label[0].item() == pytest.approx(0.1, abs=1e-6)
            assert label[1].item() == pytest.approx(0.1, abs=1e-6)
            assert label[2].item() == pytest.approx(0.8, abs=1e-6)
            assert label[3].item() == pytest.approx(0.8, abs=1e-6)

    def test_tensor_shape_always_square(self):
        """Output tensor must always be [3, image_size, image_size]."""
        for w, h in [(800, 600), (600, 800), (500, 500), (1920, 1080), (100, 3000)]:
            img = Image.new("RGB", (w, h))
            resizer = AspectRatioResize(target_size=224)
            padded, meta = resizer(img)
            transform = get_val_transform(224)
            tensor = transform(padded)
            assert tensor.shape == (3, 224, 224), f"Failed for {w}x{h}: got {tensor.shape}"


# ═══════════════════════════════════════════════════════════════════════
# 2. INFERENCE COORDINATE MAPPING
# ═══════════════════════════════════════════════════════════════════════

class TestInferenceMapping:
    """Verify model output maps correctly back to original dimensions."""

    def test_normalized_to_absolute_conversion(self):
        """Model outputs (normalized) → absolute pixel coords must be correct."""
        # Simulate model output
        pred_norm = {"x": 0.1, "y": 0.2, "width": 0.6, "height": 0.5}
        img_w, img_h = 1024, 768

        abs_x = int(pred_norm["x"] * img_w)
        abs_y = int(pred_norm["y"] * img_h)
        abs_w = int(pred_norm["width"] * img_w)
        abs_h = int(pred_norm["height"] * img_h)

        assert abs_x == 102
        assert abs_y == 153
        assert abs_w == 614
        assert abs_h == 384

        # Verify within bounds
        assert abs_x >= 0
        assert abs_y >= 0
        assert abs_x + abs_w <= img_w
        assert abs_y + abs_h <= img_h

    def test_inference_preprocessing_roundtrip(self):
        """Image → preprocess → model input → model output → original coords."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create image
            img = Image.new("RGB", (1200, 800), (100, 150, 200))
            path = os.path.join(tmpdir, "test.jpg")
            img.save(path)

            tensor, meta = preprocess_for_inference(path, image_size=224)

            assert tensor.shape == (1, 3, 224, 224)
            assert meta["original_w"] == 1200
            assert meta["original_h"] == 800

            # Simulated model output (normalized to original space)
            pred = np.array([0.1, 0.2, 0.6, 0.5])

            # Convert to absolute
            abs_crop = (
                int(pred[0] * meta["original_w"]),
                int(pred[1] * meta["original_h"]),
                int(pred[2] * meta["original_w"]),
                int(pred[3] * meta["original_h"]),
            )

            assert abs_crop == (120, 160, 720, 400)
            assert abs_crop[0] + abs_crop[2] <= meta["original_w"]
            assert abs_crop[1] + abs_crop[3] <= meta["original_h"]


# ═══════════════════════════════════════════════════════════════════════
# 3. EXIF CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════

class TestEXIFConsistency:
    """Verify EXIF rotation is applied consistently everywhere."""

    def test_load_image_applies_exif(self):
        """load_image() must apply EXIF and return RGB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img = Image.new("RGB", (200, 300), (100, 150, 200))
            path = os.path.join(tmpdir, "test.jpg")
            img.save(path)

            loaded = load_image(path)
            assert loaded.mode == "RGB"
            assert loaded.size == (200, 300)

    def test_load_image_rgba_conversion(self):
        """RGBA images must be converted to RGB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img = Image.new("RGBA", (200, 300), (100, 150, 200, 255))
            path = os.path.join(tmpdir, "test.png")
            img.save(path)

            loaded = load_image(path)
            assert loaded.mode == "RGB"

    def test_dataset_uses_load_image(self):
        """CropDataset must use load_image (which includes EXIF correction)."""
        import inspect
        source = inspect.getsource(CropDataset.__getitem__)
        assert "load_image" in source, "CropDataset.__getitem__ must use load_image()"

    def test_inference_uses_load_image(self):
        """preprocess_for_inference must use load_image (which includes EXIF)."""
        import inspect
        source = inspect.getsource(preprocess_for_inference)
        assert "load_image" in source, "preprocess_for_inference must use load_image()"


# ═══════════════════════════════════════════════════════════════════════
# 4. AUGMENTATION SAFETY
# ═══════════════════════════════════════════════════════════════════════

class TestAugmentationSafety:
    """Verify augmentation does not invalidate crop labels."""

    def test_color_jitter_preserves_spatial_layout(self):
        """ColorJitter changes pixel values but not spatial arrangement."""
        img = Image.new("RGB", (224, 224))
        # Draw a known pattern
        pixels = img.load()
        for y in range(224):
            for x in range(224):
                pixels[x, y] = (x % 256, y % 256, (x + y) % 256)

        transform = get_train_transform(224)

        # Apply multiple times — size should never change
        for _ in range(10):
            tensor = transform(img)
            assert tensor.shape == (3, 224, 224)

    def test_no_spatial_augmentation(self):
        """Training transforms must NOT include flips, rotations, or crops."""
        import inspect
        source = inspect.getsource(get_train_transform)

        forbidden = ["RandomHorizontalFlip", "RandomVerticalFlip",
                      "RandomRotation", "RandomCrop", "RandomResizedCrop",
                      "RandomAffine", "RandomPerspective"]
        for aug in forbidden:
            assert aug not in source, f"Training transform contains forbidden {aug}"

    def test_train_val_same_tensor_shape(self):
        """Train and val transforms must produce identical tensor shapes."""
        img = Image.new("RGB", (224, 224), (100, 150, 200))
        train_t = get_train_transform(224)
        val_t = get_val_transform(224)

        train_tensor = train_t(img)
        val_tensor = val_t(img)

        assert train_tensor.shape == val_tensor.shape


# ═══════════════════════════════════════════════════════════════════════
# 5. MODEL OUTPUT VALIDATION
# ═══════════════════════════════════════════════════════════════════════

class TestModelOutputValidation:
    """Verify sigmoid + boundary constraints produce valid boxes."""

    def test_sigmoid_constrains_output_0_1(self):
        """Raw model output must be in [0, 1] due to sigmoid."""
        model = build_model(backbone="mobilenet_v3_small", pretrained=False)
        model.eval()

        # Random input
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (4, 4)
        assert (out >= 0.0).all(), f"Output has values < 0: {out.min()}"
        assert (out <= 1.0).all(), f"Output has values > 1: {out.max()}"

    def test_predict_with_validation_enforces_boundaries(self):
        """predict_with_validation must ensure x+w≤1 and y+h≤1."""
        model = build_model(backbone="mobilenet_v3_small", pretrained=False)
        model.eval()

        # Test with many random inputs
        for _ in range(20):
            x = torch.randn(8, 3, 224, 224)
            with torch.no_grad():
                out = model.predict_with_validation(x)

            x_coord = out[:, 0]
            y_coord = out[:, 1]
            width = out[:, 2]
            height = out[:, 3]

            # x + width ≤ 1
            assert ((x_coord + width) <= 1.0 + 1e-6).all(), \
                f"x+w > 1: max = {(x_coord + width).max()}"

            # y + height ≤ 1
            assert ((y_coord + height) <= 1.0 + 1e-6).all(), \
                f"y+h > 1: max = {(y_coord + height).max()}"

            # All values non-negative
            assert (out >= 0.0).all()

    def test_extreme_inputs_still_valid(self):
        """Extreme pixel values should still produce valid crops."""
        model = build_model(backbone="mobilenet_v3_small", pretrained=False)
        model.eval()

        # All zeros
        x_zero = torch.zeros(1, 3, 224, 224)
        # All ones
        x_ones = torch.ones(1, 3, 224, 224)
        # Very large values
        x_large = torch.full((1, 3, 224, 224), 100.0)

        for x in [x_zero, x_ones, x_large]:
            with torch.no_grad():
                out = model.predict_with_validation(x)
            assert (out >= 0.0).all()
            assert (out[:, 0] + out[:, 2] <= 1.0 + 1e-6).all()
            assert (out[:, 1] + out[:, 3] <= 1.0 + 1e-6).all()

    def test_model_output_width_height_positive(self):
        """Width and height must be > 0 (sigmoid never outputs exactly 0)."""
        model = build_model(backbone="mobilenet_v3_small", pretrained=False)
        model.eval()

        x = torch.randn(16, 3, 224, 224)
        with torch.no_grad():
            out = model(x)

        # Sigmoid output is always strictly > 0 (approaches 0 but never reaches it)
        assert (out[:, 2] > 0.0).all(), "Width should be > 0"
        assert (out[:, 3] > 0.0).all(), "Height should be > 0"


# ═══════════════════════════════════════════════════════════════════════
# 6. IoU COORDINATE SYSTEM CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════

class TestIoUCoordinateConsistency:
    """Verify IoU is calculated in the same coordinate system for both inputs."""

    def test_iou_symmetric(self):
        """IoU(A, B) == IoU(B, A)."""
        a = np.array([0.1, 0.1, 0.5, 0.5])
        b = np.array([0.2, 0.2, 0.6, 0.6])
        assert compute_iou(a, b) == pytest.approx(compute_iou(b, a))

    def test_iou_uses_xywh_format(self):
        """Both pred and target must use [x, y, width, height] format."""
        # If format were [x1, y1, x2, y2], this would give wrong IoU
        pred = np.array([0.0, 0.0, 0.5, 0.5])  # xywh
        target = np.array([0.0, 0.0, 0.5, 0.5])
        assert compute_iou(pred, target) == pytest.approx(1.0)

    def test_evaluation_uses_same_format_as_model(self):
        """
        evaluate_predictions input format must match model output format.

        Both must be [x, y, width, height] normalized.
        """
        preds = np.array([[0.1, 0.1, 0.5, 0.5]])
        targets = np.array([[0.1, 0.1, 0.5, 0.5]])
        result = evaluate_predictions(preds, targets)
        assert result["mean_iou"] == pytest.approx(1.0)
        assert result["mae_overall"] == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════
# 7. SPLIT INTEGRITY
# ═══════════════════════════════════════════════════════════════════════

class TestSplitIntegrity:
    """Verify splitting happens before augmentation and prevents leakage."""

    def test_splits_are_disjoint(self):
        """No example should appear in multiple splits."""
        # Simulate the split logic
        import random
        ids = [f"id_{i}" for i in range(100)]
        rng = random.Random(42)
        rng.shuffle(ids)

        n = len(ids)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)

        train_ids = set(ids[:n_train])
        val_ids = set(ids[n_train:n_train + n_val])
        test_ids = set(ids[n_train + n_val:])

        # No overlap
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

        # All accounted for
        assert len(train_ids | val_ids | test_ids) == n

    def test_splits_deterministic(self):
        """Same seed must produce same splits."""
        import random

        ids = [f"id_{i}" for i in range(50)]

        for seed in [42, 123, 999]:
            rng1 = random.Random(seed)
            ids1 = ids.copy()
            rng1.shuffle(ids1)

            rng2 = random.Random(seed)
            ids2 = ids.copy()
            rng2.shuffle(ids2)

            assert ids1 == ids2

    def test_augmentation_is_training_time_only(self):
        """Augmentation only happens during training, not during split assignment."""
        # CropDataset with is_training=True uses ColorJitter
        # CropDataset with is_training=False does not
        # Splits are assigned in dataset_service.py which has no augmentation
        train_transform = get_train_transform()
        val_transform = get_val_transform()

        # Verify val has no augmentation
        import inspect
        val_source = inspect.getsource(get_val_transform)
        assert "ColorJitter" not in val_source


# ═══════════════════════════════════════════════════════════════════════
# 8. CHECKPOINT COMPLETENESS
# ═══════════════════════════════════════════════════════════════════════

class TestCheckpointCompleteness:
    """Verify checkpoints contain all required information."""

    def test_preprocessing_config_complete(self):
        """Preprocessing config must contain all reproduction info."""
        config = get_preprocessing_config(224)

        assert config["image_size"] == 224
        assert config["resize_mode"] == "aspect_ratio_preserving_with_padding"
        assert config["normalization"]["mean"] == IMAGENET_MEAN
        assert config["normalization"]["std"] == IMAGENET_STD
        assert config["pad_fill"] == [0, 0, 0]
        assert config["coordinate_system"] == "normalized_0_1_relative_to_original"

    def test_preprocessing_config_serializable(self):
        """Config must be JSON serializable."""
        config = get_preprocessing_config(224)
        json_str = json.dumps(config)
        parsed = json.loads(json_str)
        assert parsed == config

    def test_model_state_dict_savable(self):
        """Model weights must be savable and reloadable."""
        model = build_model(backbone="mobilenet_v3_small", pretrained=False)
        model.eval()

        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out1 = model(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            torch.save(model.state_dict(), path)

            model2 = build_model(backbone="mobilenet_v3_small", pretrained=False)
            model2.load_state_dict(torch.load(path, weights_only=True))
            model2.eval()

            with torch.no_grad():
                out2 = model2(x)

            assert torch.allclose(out1, out2, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# 9. SYNTHETIC DATASET END-TO-END TEST
# ═══════════════════════════════════════════════════════════════════════

class TestSyntheticDataset:
    """
    End-to-end test with synthetic images where correct crop coords are known exactly.
    Verifies: original → preprocessing → model coordinate system → original coords.
    """

    SYNTHETIC_EXAMPLES = [
        # (width, height, crop_x, crop_y, crop_w, crop_h)
        (800, 600, 0.10, 0.10, 0.80, 0.80),  # Example 1
        (1024, 768, 0.20, 0.05, 0.60, 0.90),  # Example 2
        (640, 480, 0.05, 0.20, 0.90, 0.60),  # Example 3
        (1920, 1080, 0.15, 0.10, 0.70, 0.80),  # Landscape HD
        (1080, 1920, 0.10, 0.15, 0.80, 0.70),  # Portrait HD
        (500, 500, 0.00, 0.00, 1.00, 1.00),  # Full image
    ]

    def test_dataset_loading_preserves_labels(self):
        """Labels must pass through dataset loading unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            examples = []
            for w, h, cx, cy, cw, ch in self.SYNTHETIC_EXAMPLES:
                _, example = create_synthetic_image(w, h, cx, cy, cw, ch, tmpdir)
                examples.append(example)

            dataset = CropDataset(examples, image_size=224, is_training=False)

            for i, (w, h, cx, cy, cw, ch) in enumerate(self.SYNTHETIC_EXAMPLES):
                tensor, label = dataset[i]

                assert tensor.shape == (3, 224, 224)
                assert label[0].item() == pytest.approx(cx, abs=1e-6), f"Example {i}: x mismatch"
                assert label[1].item() == pytest.approx(cy, abs=1e-6), f"Example {i}: y mismatch"
                assert label[2].item() == pytest.approx(cw, abs=1e-6), f"Example {i}: w mismatch"
                assert label[3].item() == pytest.approx(ch, abs=1e-6), f"Example {i}: h mismatch"

    def test_inference_preprocessing_preserves_metadata(self):
        """Preprocessing must preserve original dimensions in metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for w, h, cx, cy, cw, ch in self.SYNTHETIC_EXAMPLES:
                path, _ = create_synthetic_image(w, h, cx, cy, cw, ch, tmpdir)
                tensor, meta = preprocess_for_inference(path, image_size=224)

                assert tensor.shape == (1, 3, 224, 224)
                assert meta["original_w"] == w, f"original_w mismatch for {w}x{h}"
                assert meta["original_h"] == h, f"original_h mismatch for {w}x{h}"

    def test_normalized_to_absolute_roundtrip(self):
        """normalized → absolute → normalized must be identity."""
        for w, h, cx, cy, cw, ch in self.SYNTHETIC_EXAMPLES:
            # Normalize → absolute
            abs_x = int(cx * w)
            abs_y = int(cy * h)
            abs_w = int(cw * w)
            abs_h = int(ch * h)

            # Absolute → normalize
            norm_x2 = abs_x / w
            norm_y2 = abs_y / h
            norm_w2 = abs_w / w
            norm_h2 = abs_h / h

            # Should be close (int truncation introduces small error)
            assert norm_x2 == pytest.approx(cx, abs=1.0 / w)
            assert norm_y2 == pytest.approx(cy, abs=1.0 / h)
            assert norm_w2 == pytest.approx(cw, abs=1.0 / w)
            assert norm_h2 == pytest.approx(ch, abs=1.0 / h)

    def test_absolute_crop_within_bounds(self):
        """Absolute crop must stay within image boundaries."""
        for w, h, cx, cy, cw, ch in self.SYNTHETIC_EXAMPLES:
            abs_x = int(cx * w)
            abs_y = int(cy * h)
            abs_w = int(cw * w)
            abs_h = int(ch * h)

            assert abs_x >= 0
            assert abs_y >= 0
            assert abs_x + abs_w <= w, f"x+w={abs_x + abs_w} > w={w}"
            assert abs_y + abs_h <= h, f"y+h={abs_y + abs_h} > h={h}"

    def test_model_forward_with_synthetic_data(self):
        """Model must produce valid output for synthetic data."""
        model = build_model(backbone="mobilenet_v3_small", pretrained=False)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            examples = []
            for w, h, cx, cy, cw, ch in self.SYNTHETIC_EXAMPLES:
                _, example = create_synthetic_image(w, h, cx, cy, cw, ch, tmpdir)
                examples.append(example)

            dataset = CropDataset(examples, image_size=224, is_training=False)
            loader = torch.utils.data.DataLoader(dataset, batch_size=len(examples))

            images, labels = next(iter(loader))

            with torch.no_grad():
                predictions = model.predict_with_validation(images)

            # Check shapes
            assert predictions.shape == labels.shape == (len(examples), 4)

            # Check all predictions are valid
            for i in range(len(examples)):
                pred = predictions[i]
                assert pred[0] >= 0.0 and pred[0] <= 1.0  # x
                assert pred[1] >= 0.0 and pred[1] <= 1.0  # y
                assert pred[2] > 0.0 and pred[2] <= 1.0   # width
                assert pred[3] > 0.0 and pred[3] <= 1.0   # height
                assert pred[0] + pred[2] <= 1.0 + 1e-6    # x + width
                assert pred[1] + pred[3] <= 1.0 + 1e-6    # y + height


# ═══════════════════════════════════════════════════════════════════════
# 10. MPS DEVICE TEST
# ═══════════════════════════════════════════════════════════════════════

class TestMPSDevice:
    """Verify MPS training works on Apple Silicon."""

    def test_device_detection(self):
        """Device detection should return a valid device."""
        device = get_device()
        assert str(device) in ("mps", "cuda", "cpu")

    def test_model_to_device(self):
        """Model must be movable to the detected device."""
        device = get_device()
        model = build_model(backbone="mobilenet_v3_small", pretrained=False)
        model = model.to(device)

        x = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (2, 4)
        assert out.device.type == device.type

    def test_loss_backward_on_device(self):
        """Training step must work on detected device."""
        device = get_device()
        model = build_model(backbone="mobilenet_v3_small", pretrained=False)
        model = model.to(device)
        model.train()

        criterion = torch.nn.SmoothL1Loss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        x = torch.randn(2, 3, 224, 224).to(device)
        target = torch.tensor([[0.1, 0.1, 0.8, 0.8], [0.2, 0.2, 0.6, 0.6]]).to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0  # Should have some loss
        assert not torch.isnan(loss)  # No NaN
