"""
Dataset sanity-check visualization.

Generates comparison images showing:
- Original image
- Ground-truth crop rectangle (green)
- Predicted crop rectangle (red, if model is available)

Usage:
    python -m tools.sanity_check [--output-dir ./sanity_check_output] [--max-images 10]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.preprocessing import load_image

logger = logging.getLogger(__name__)


def draw_crop_rect(
    draw: ImageDraw.ImageDraw,
    x: int, y: int, w: int, h: int,
    color: str,
    label: str,
    line_width: int = 3,
):
    """Draw a labeled rectangle on the image."""
    draw.rectangle(
        [x, y, x + w, y + h],
        outline=color,
        width=line_width,
    )
    # Label
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    # Background for text
    draw.rectangle(
        [x, y - text_h - 4, x + text_w + 8, y],
        fill=color,
    )
    draw.text((x + 4, y - text_h - 2), label, fill="white", font=font)


def generate_sanity_check_image(
    image_path: str,
    image_width: int,
    image_height: int,
    gt_norm: dict,
    pred_norm: dict = None,
    filename: str = "sanity_check.jpg",
) -> Image.Image:
    """
    Generate a single sanity-check comparison image.

    Args:
        image_path: Path to the original image
        image_width: Original image width
        image_height: Original image height
        gt_norm: Ground truth dict with x, y, width, height (normalized)
        pred_norm: Prediction dict with x, y, width, height (normalized), or None
        filename: Label for the image

    Returns:
        PIL Image with rectangles drawn
    """
    img = load_image(image_path).copy()

    # Scale to reasonable display size
    max_display = 800
    scale = min(max_display / image_width, max_display / image_height, 1.0)
    if scale < 1.0:
        display_w = int(image_width * scale)
        display_h = int(image_height * scale)
        img = img.resize((display_w, display_h), Image.LANCZOS)
    else:
        display_w, display_h = image_width, image_height

    draw = ImageDraw.Draw(img)

    # Ground truth (green)
    gt_x = int(gt_norm["x"] * display_w)
    gt_y = int(gt_norm["y"] * display_h)
    gt_w = int(gt_norm["width"] * display_w)
    gt_h = int(gt_norm["height"] * display_h)
    draw_crop_rect(draw, gt_x, gt_y, gt_w, gt_h, "lime", "GT")

    # Prediction (red)
    if pred_norm:
        pred_x = int(pred_norm["x"] * display_w)
        pred_y = int(pred_norm["y"] * display_h)
        pred_w = int(pred_norm["width"] * display_w)
        pred_h = int(pred_norm["height"] * display_h)
        draw_crop_rect(draw, pred_x, pred_y, pred_w, pred_h, "red", "PRED")

    # Add filename label at bottom
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        font = ImageFont.load_default()
    draw.text((5, display_h - 18), filename, fill="white", font=font)

    return img


def run_sanity_check(output_dir: str = "./sanity_check_output", max_images: int = 20):
    """Run sanity check on all labeled examples in the database."""
    from backend.database.connection import SessionLocal
    from backend.models.db_models import TrainingExample, ModelVersion

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()

    try:
        examples = (
            db.query(TrainingExample)
            .filter(TrainingExample.status == "labeled")
            .limit(max_images)
            .all()
        )

        if not examples:
            print("No labeled examples found in database.")
            return

        # Check for active model
        active_model = db.query(ModelVersion).filter(ModelVersion.is_active == 1).first()
        has_model = active_model is not None

        if has_model:
            from ml.inference import predict_crop
            print(f"Using model: {active_model.version} ({active_model.backbone})")
        else:
            print("No trained model available — showing ground truth only.")

        results = []
        for i, ex in enumerate(examples):
            print(f"Processing {i + 1}/{len(examples)}: {ex.original_filename}")

            gt = {
                "x": ex.norm_x,
                "y": ex.norm_y,
                "width": ex.norm_width,
                "height": ex.norm_height,
            }

            pred = None
            if has_model:
                try:
                    pred = predict_crop(
                        ex.original_path,
                        active_model.model_path,
                        active_model.image_size,
                        backbone=active_model.backbone,
                    )
                except Exception as e:
                    print(f"  Prediction failed: {e}")

            # Generate comparison image
            comp_img = generate_sanity_check_image(
                ex.original_path,
                ex.original_width,
                ex.original_height,
                gt, pred,
                filename=ex.original_filename,
            )

            out_file = output_path / f"check_{i:03d}_{ex.original_filename}"
            comp_img.save(str(out_file), quality=90)

            # Compute IoU if we have a prediction
            if pred:
                from ml.metrics import compute_iou
                gt_arr = np.array([gt["x"], gt["y"], gt["width"], gt["height"]])
                pred_arr = np.array([pred["x"], pred["y"], pred["width"], pred["height"]])
                iou = compute_iou(pred_arr, gt_arr)
                results.append({"file": ex.original_filename, "iou": iou})
                print(f"  IoU: {iou:.4f}")

        # Summary
        if results:
            ious = [r["iou"] for r in results]
            print(f"\n--- Summary ---")
            print(f"Mean IoU: {np.mean(ious):.4f}")
            print(f"Median IoU: {np.median(ious):.4f}")
            print(f"Min IoU: {np.min(ious):.4f}")
            print(f"Max IoU: {np.max(ious):.4f}")

            # Save results JSON
            (output_path / "results.json").write_text(json.dumps(results, indent=2))

        print(f"\nSanity check images saved to: {output_path}")

    finally:
        db.close()


def run_synthetic_sanity_check(output_dir: str = "./sanity_check_output"):
    """
    Run sanity check with synthetic images (no database needed).
    Useful for verifying the coordinate pipeline is correct.
    """
    import tempfile

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    synthetic_cases = [
        {"name": "example_1", "w": 800, "h": 600, "cx": 0.10, "cy": 0.10, "cw": 0.80, "ch": 0.80},
        {"name": "example_2", "w": 1024, "h": 768, "cx": 0.20, "cy": 0.05, "cw": 0.60, "ch": 0.90},
        {"name": "example_3", "w": 640, "h": 480, "cx": 0.05, "cy": 0.20, "cw": 0.90, "ch": 0.60},
        {"name": "landscape_hd", "w": 1920, "h": 1080, "cx": 0.15, "cy": 0.10, "cw": 0.70, "ch": 0.80},
        {"name": "portrait_hd", "w": 1080, "h": 1920, "cx": 0.10, "cy": 0.15, "cw": 0.80, "ch": 0.70},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        for case in synthetic_cases:
            # Create synthetic image
            img = Image.new("RGB", (case["w"], case["h"]), (30, 30, 40))
            pixels = img.load()
            px_x = int(case["cx"] * case["w"])
            px_y = int(case["cy"] * case["h"])
            px_w = int(case["cw"] * case["w"])
            px_h = int(case["ch"] * case["h"])

            for y in range(px_y, min(px_y + px_h, case["h"])):
                for x in range(px_x, min(px_x + px_w, case["w"])):
                    r = int(80 + 175 * (x - px_x) / max(px_w, 1))
                    g = int(100 + 155 * (y - px_y) / max(px_h, 1))
                    pixels[x, y] = (r, g, 120)

            path = Path(tmpdir) / f"{case['name']}.jpg"
            img.save(str(path), quality=95)

            gt = {"x": case["cx"], "y": case["cy"], "width": case["cw"], "height": case["ch"]}

            comp_img = generate_sanity_check_image(
                str(path), case["w"], case["h"],
                gt, pred_norm=None,
                filename=case["name"],
            )

            out_file = output_path / f"synthetic_{case['name']}.jpg"
            comp_img.save(str(out_file), quality=90)
            print(f"Saved: {out_file}")

    print(f"\nSynthetic sanity check images saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset sanity check visualization")
    parser.add_argument("--output-dir", default="./sanity_check_output", help="Output directory")
    parser.add_argument("--max-images", type=int, default=20, help="Max images to process")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic images only")
    args = parser.parse_args()

    if args.synthetic:
        run_synthetic_sanity_check(args.output_dir)
    else:
        run_sanity_check(args.output_dir, args.max_images)
