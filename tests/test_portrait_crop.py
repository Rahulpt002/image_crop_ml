"""
Tests for portrait crop system:
- 3:4 aspect ratio constraint
- Replaceable face detector interface
- Portrait export (150x200 px, 300 DPI)
- Training label aspect ratio validation
- Inference pipeline integration
"""

import os
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from ml.model import CropNet, constrain_to_aspect_ratio
from ml.face_detector import (
    FaceDetection,
    FaceDetector,
    MediaPipeFaceDetector,
    OpenCVHaarFaceDetector,
    get_face_detector,
)
from ml.inference import predict_portrait_crop
from backend.services.image_service import export_portrait_crop
from backend.api.dataset import validate_aspect_ratio
from backend.config.settings import (
    PORTRAIT_ASPECT_RATIO,
    PORTRAIT_OUTPUT_WIDTH,
    PORTRAIT_OUTPUT_HEIGHT,
    PORTRAIT_OUTPUT_DPI,
    ASPECT_RATIO_TOLERANCE,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. ASPECT RATIO CONSTRAINT TESTS
# ═══════════════════════════════════════════════════════════════════════

def test_aspect_ratio_constraint_exact():
    """constrain_to_aspect_ratio produces width/height == 0.75 for square space."""
    # Input box: x=0.1, y=0.1, w=0.6, h=0.5 -> AR = 1.2 (not 0.75)
    pred = torch.tensor([[0.1, 0.1, 0.6, 0.5]], dtype=torch.float32)
    constrained = constrain_to_aspect_ratio(pred, aspect_ratio=0.75)

    w = constrained[0, 2].item()
    h = constrained[0, 3].item()
    assert abs((w / h) - 0.75) < 1e-4, f"Expected 0.75, got {w / h}"


def test_aspect_ratio_constraint_bounds():
    """Constrained box stays within [0, 1] for arbitrary random inputs."""
    torch.manual_seed(42)
    for _ in range(100):
        # Generate random boxes: x, y in [0, 0.8], w, h in [0.05, 0.9]
        x = torch.rand(1) * 0.7
        y = torch.rand(1) * 0.7
        w = torch.rand(1) * 0.8 + 0.05
        h = torch.rand(1) * 0.8 + 0.05
        # Ensure x+w <= 1 and y+h <= 1
        w = torch.clamp(w, max=1.0 - x)
        h = torch.clamp(h, max=1.0 - y)

        pred = torch.stack([x, y, w, h], dim=1)
        constrained = constrain_to_aspect_ratio(pred, aspect_ratio=0.75)

        cx, cy, cw, ch = constrained[0].tolist()

        assert cx >= 0.0, f"cx={cx} < 0"
        assert cy >= 0.0, f"cy={cy} < 0"
        assert cw > 0.0, f"cw={cw} <= 0"
        assert ch > 0.0, f"ch={ch} <= 0"
        assert cx + cw <= 1.001, f"cx+cw={cx+cw} > 1.0"
        assert cy + ch <= 1.001, f"cy+ch={cy+ch} > 1.0"
        assert abs((cw / ch) - 0.75) < 1e-4, f"AR={cw/ch} != 0.75"


def test_aspect_ratio_constraint_batch():
    """Works on batches of predictions [B, 4]."""
    preds = torch.tensor([
        [0.1, 0.1, 0.4, 0.4],
        [0.2, 0.3, 0.5, 0.6],
        [0.0, 0.0, 0.8, 0.8],
    ], dtype=torch.float32)

    constrained = constrain_to_aspect_ratio(preds, aspect_ratio=0.75)
    assert constrained.shape == (3, 4)

    for i in range(3):
        w = constrained[i, 2].item()
        h = constrained[i, 3].item()
        assert abs((w / h) - 0.75) < 1e-4


def test_aspect_ratio_non_square_image():
    """Handles non-square original image dimensions via image_aspect_ratio."""
    # Suppose image is 1200x800 (w=1200, h=800, AR=1.5)
    img_ar = 1200 / 800  # 1.5
    pred = torch.tensor([[0.2, 0.2, 0.5, 0.5]], dtype=torch.float32)

    constrained = constrain_to_aspect_ratio(
        pred, aspect_ratio=0.75, image_aspect_ratio=img_ar
    )

    norm_w = constrained[0, 2].item()
    norm_h = constrained[0, 3].item()

    # Physical dimensions:
    phys_w = norm_w * 1200
    phys_h = norm_h * 800

    physical_ar = phys_w / phys_h
    assert abs(physical_ar - 0.75) < 1e-3, f"Expected physical AR 0.75, got {physical_ar}"


# ═══════════════════════════════════════════════════════════════════════
# 2. FACE DETECTOR INTERFACE TESTS
# ═══════════════════════════════════════════════════════════════════════

def test_face_detector_interface():
    """FaceDetector ABC requires detect() implementation."""
    class DummyDetector(FaceDetector):
        def detect(self, image: Image.Image):
            return [
                FaceDetection(x=10, y=20, width=50, height=60, confidence=0.95)
            ]

    detector = DummyDetector()
    img = Image.new("RGB", (100, 100), color="white")
    detections = detector.detect(img)

    assert len(detections) == 1
    assert detections[0].x == 10
    assert detections[0].confidence == 0.95
    primary = detector.detect_primary(img)
    assert primary is not None
    assert primary.width == 50


def test_face_detection_no_face_blank():
    """Haar detector returns empty list for blank white image."""
    detector = OpenCVHaarFaceDetector()
    blank = Image.new("RGB", (300, 300), color="white")
    detections = detector.detect(blank)
    assert isinstance(detections, list)
    assert len(detections) == 0
    assert detector.detect_primary(blank) is None


def test_get_face_detector_factory():
    """get_face_detector returns singleton instance."""
    d1 = get_face_detector("opencv")
    d2 = get_face_detector("opencv")
    assert d1 is d2
    assert isinstance(d1, FaceDetector)


# ═══════════════════════════════════════════════════════════════════════
# 3. EXPORT PIPELINE TESTS (150x200 px, 300 DPI)
# ═══════════════════════════════════════════════════════════════════════

def test_export_dimensions():
    """export_portrait_crop produces an image of exactly 150x200 px."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        src_path = tmp_path / "orig.jpg"
        out_path = tmp_path / "exported.jpg"

        # Create 800x1000 test image
        img = Image.new("RGB", (800, 1000), color=(120, 150, 180))
        img.save(src_path, quality=95)

        export_portrait_crop(
            image_path=src_path,
            x=100,
            y=100,
            width=300,
            height=400,
            output_path=out_path,
            target_width=PORTRAIT_OUTPUT_WIDTH,
            target_height=PORTRAIT_OUTPUT_HEIGHT,
            dpi=PORTRAIT_OUTPUT_DPI,
        )

        assert out_path.exists()
        with Image.open(out_path) as exported:
            assert exported.size == (150, 200), f"Expected (150, 200), got {exported.size}"


def test_export_dpi():
    """export_portrait_crop embeds 300x300 DPI metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        src_path = tmp_path / "orig.jpg"
        out_path = tmp_path / "exported.jpg"

        img = Image.new("RGB", (600, 800), color="blue")
        img.save(src_path, quality=95)

        export_portrait_crop(
            image_path=src_path,
            x=50,
            y=50,
            width=300,
            height=400,
            output_path=out_path,
            dpi=(300, 300),
        )

        with Image.open(out_path) as exported:
            # Pillow stores dpi in info dict
            dpi = exported.info.get("dpi")
            assert dpi is not None, "DPI metadata missing"
            assert round(dpi[0]) == 300 and round(dpi[1]) == 300, f"Expected 300 DPI, got {dpi}"


def test_export_no_distortion():
    """When input crop is 3:4 and output is 150x200 (3:4), there is no distortion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        src_path = tmp_path / "orig.jpg"
        out_path = tmp_path / "exported.jpg"

        # Draw a pattern with a known 3:4 feature
        img = Image.new("RGB", (600, 800), color="white")
        # Draw a rectangle inside the crop region
        arr = np.array(img)
        arr[100:500, 100:400] = [200, 50, 50]  # width 300, height 400 (AR = 0.75)
        Image.fromarray(arr).save(src_path, quality=95)

        export_portrait_crop(
            image_path=src_path,
            x=100,
            y=100,
            width=300,
            height=400,
            output_path=out_path,
        )

        with Image.open(out_path) as exported:
            assert exported.size == (150, 200)
            assert exported.size[0] / exported.size[1] == 0.75


# ═══════════════════════════════════════════════════════════════════════
# 4. TRAINING LABEL ASPECT RATIO VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def test_training_label_ar_validation():
    """validate_aspect_ratio correctly validates 3:4 targets."""
    # Perfect 3:4: 150x200, 300x400, 450x600, 750x1000
    for w, h in [(150, 200), (300, 400), (450, 600), (750, 1000)]:
        is_valid, ar = validate_aspect_ratio(w, h)
        assert is_valid, f"{w}x{h} should be valid 3:4"
        assert abs(ar - 0.75) < 1e-4


def test_training_label_bad_ar():
    """validate_aspect_ratio flags non-3:4 targets."""
    # 1:1 square
    is_valid, ar = validate_aspect_ratio(500, 500)
    assert not is_valid
    assert abs(ar - 1.0) < 1e-4

    # 16:9 landscape
    is_valid, ar = validate_aspect_ratio(1920, 1080)
    assert not is_valid

    # 9:16 vertical video
    is_valid, ar = validate_aspect_ratio(1080, 1920)
    assert not is_valid


# ═══════════════════════════════════════════════════════════════════════
# 5. INFERENCE PIPELINE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════

def test_inference_pipeline_portrait():
    """Full portrait crop pipeline runs and returns 3:4 crop coords."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_path = tmp_path / "test_portrait.jpg"
        model_path = tmp_path / "cropnet.pt"

        # Create test image
        img = Image.new("RGB", (600, 800), color=(200, 180, 160))
        img.save(img_path, quality=95)

        # Create and save a small CropNet model
        model = CropNet(backbone_name="mobilenet_v3_small", pretrained=False)
        torch.save(model.state_dict(), model_path)

        result = predict_portrait_crop(
            image_path=str(img_path),
            model_path=str(model_path),
            image_size=224,
            backbone="mobilenet_v3_small",
            aspect_ratio=PORTRAIT_ASPECT_RATIO,
            face_detector_backend="opencv",  # use opencv for fast unit test
        )

        assert "x" in result
        assert "y" in result
        assert "width" in result
        assert "height" in result
        assert "face_detected" in result
        assert "face_box" in result
        assert "aspect_ratio" in result

        # Physical aspect ratio must be ≈ 0.75
        assert abs(result["aspect_ratio"] - 0.75) < 0.05, f"AR: {result['aspect_ratio']}"


def test_fallback_no_face():
    """Inference continues and returns valid 3:4 crop even if no face is found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_path = tmp_path / "blank.jpg"
        model_path = tmp_path / "cropnet.pt"

        # Solid color image with no face
        img = Image.new("RGB", (400, 600), color=(50, 50, 50))
        img.save(img_path, quality=95)

        model = CropNet(backbone_name="mobilenet_v3_small", pretrained=False)
        torch.save(model.state_dict(), model_path)

        result = predict_portrait_crop(
            image_path=str(img_path),
            model_path=str(model_path),
            image_size=224,
            backbone="mobilenet_v3_small",
            face_detector_backend="opencv",
        )

        assert result["face_detected"] is False
        assert result["face_box"] is None
        assert abs(result["aspect_ratio"] - 0.75) < 0.05
