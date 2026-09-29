"""Dataset API routes."""

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.config.settings import UPLOAD_DIR, CROP_CONFIDENCE_THRESHOLD
from backend.database.connection import get_db
from backend.models.db_models import TrainingExample
from backend.schemas.api_schemas import (
    TrainingExampleResponse,
    TrainingExampleUpdate,
    DatasetStats,
    DatasetSplitConfig,
    CropExtractionResult as CropExtractionSchema,
)
from backend.services.image_service import save_uploaded_image, validate_image_file, copy_to_dataset
from backend.services.crop_extractor import CropLabelExtractor
from backend.services.dataset_service import get_dataset_stats, assign_splits

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/dataset", tags=["dataset"])

# Shared crop extractor
crop_extractor = CropLabelExtractor()


@router.post("/upload", response_model=TrainingExampleResponse)
async def upload_training_pair(
    original: UploadFile = File(...),
    target: UploadFile = File(None),
    db: Session = Depends(get_db),
):
    """
    Upload an original image and optionally a target crop image.
    If target is provided, attempts automatic crop label extraction.
    """
    # Validate & save original
    orig_content = await original.read()
    validate_image_file(original.filename, len(orig_content))
    orig_name, orig_path, orig_w, orig_h = save_uploaded_image(
        orig_content, original.filename, subfolder="originals"
    )

    # Create DB record
    example = TrainingExample(
        original_filename=original.filename,
        original_path=str(orig_path),
        original_width=orig_w,
        original_height=orig_h,
        status="pending",
    )

    # Process target if provided
    if target:
        tgt_content = await target.read()
        validate_image_file(target.filename, len(tgt_content))
        tgt_name, tgt_path, tgt_w, tgt_h = save_uploaded_image(
            tgt_content, target.filename, subfolder="targets"
        )

        example.target_filename = target.filename
        example.target_path = str(tgt_path)
        example.target_width = tgt_w
        example.target_height = tgt_h

        # Attempt automatic crop extraction
        try:
            result = crop_extractor.extract_crop(orig_path, tgt_path)
            example.crop_x = result.x
            example.crop_y = result.y
            example.crop_width = result.width
            example.crop_height = result.height

            norm = result.to_normalized(orig_w, orig_h)
            example.norm_x = norm["x"]
            example.norm_y = norm["y"]
            example.norm_width = norm["width"]
            example.norm_height = norm["height"]

            example.confidence = result.confidence
            example.extraction_method = result.method
            example.status = "labeled" if result.confidence >= CROP_CONFIDENCE_THRESHOLD else "pending"

        except Exception as e:
            logger.warning(f"Crop extraction failed: {e}")
            example.confidence = 0.0
            example.extraction_method = "failed"
            example.status = "pending"

    db.add(example)
    db.commit()
    db.refresh(example)

    return example


@router.get("", response_model=dict)
async def get_dataset(
    skip: int = 0,
    limit: int = 50,
    split: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Get dataset examples with pagination and filtering."""
    query = db.query(TrainingExample)

    if split:
        query = query.filter(TrainingExample.split == split)
    if status:
        query = query.filter(TrainingExample.status == status)

    total = query.count()
    examples = query.order_by(TrainingExample.created_at.desc()).offset(skip).limit(limit).all()

    return {
        "total": total,
        "examples": [TrainingExampleResponse.model_validate(e) for e in examples],
        "stats": get_dataset_stats(db).model_dump(),
    }


@router.get("/stats", response_model=DatasetStats)
async def get_stats(db: Session = Depends(get_db)):
    """Get dataset statistics."""
    return get_dataset_stats(db)


@router.get("/{example_id}", response_model=TrainingExampleResponse)
async def get_example(example_id: str, db: Session = Depends(get_db)):
    """Get a single training example."""
    example = db.query(TrainingExample).filter(TrainingExample.id == example_id).first()
    if not example:
        raise HTTPException(status_code=404, detail="Example not found")
    return example


@router.put("/{example_id}", response_model=TrainingExampleResponse)
async def update_example(
    example_id: str,
    update: TrainingExampleUpdate,
    db: Session = Depends(get_db),
):
    """Update a training example's crop label or metadata."""
    example = db.query(TrainingExample).filter(TrainingExample.id == example_id).first()
    if not example:
        raise HTTPException(status_code=404, detail="Example not found")

    update_data = update.model_dump(exclude_unset=True)

    # If crop coordinates are being updated, recalculate normalized values
    if any(k in update_data for k in ("crop_x", "crop_y", "crop_width", "crop_height")):
        cx = update_data.get("crop_x", example.crop_x)
        cy = update_data.get("crop_y", example.crop_y)
        cw = update_data.get("crop_width", example.crop_width)
        ch = update_data.get("crop_height", example.crop_height)

        if cx is not None and cy is not None and cw is not None and ch is not None:
            update_data["norm_x"] = cx / example.original_width
            update_data["norm_y"] = cy / example.original_height
            update_data["norm_width"] = cw / example.original_width
            update_data["norm_height"] = ch / example.original_height

    for key, value in update_data.items():
        setattr(example, key, value)

    db.commit()
    db.refresh(example)

    return example


@router.post("/{example_id}/label", response_model=TrainingExampleResponse)
async def set_manual_label(
    example_id: str,
    crop_x: int = Form(...),
    crop_y: int = Form(...),
    crop_width: int = Form(...),
    crop_height: int = Form(...),
    db: Session = Depends(get_db),
):
    """Manually set the crop label for a training example."""
    example = db.query(TrainingExample).filter(TrainingExample.id == example_id).first()
    if not example:
        raise HTTPException(status_code=404, detail="Example not found")

    # Validate bounds
    if crop_x + crop_width > example.original_width:
        raise HTTPException(status_code=400, detail="Crop exceeds image width")
    if crop_y + crop_height > example.original_height:
        raise HTTPException(status_code=400, detail="Crop exceeds image height")

    example.crop_x = crop_x
    example.crop_y = crop_y
    example.crop_width = crop_width
    example.crop_height = crop_height
    example.norm_x = crop_x / example.original_width
    example.norm_y = crop_y / example.original_height
    example.norm_width = crop_width / example.original_width
    example.norm_height = crop_height / example.original_height
    example.extraction_method = "manual"
    example.confidence = 1.0
    example.status = "labeled"

    db.commit()
    db.refresh(example)

    return example


@router.delete("/{example_id}")
async def delete_example(example_id: str, db: Session = Depends(get_db)):
    """Delete a training example."""
    example = db.query(TrainingExample).filter(TrainingExample.id == example_id).first()
    if not example:
        raise HTTPException(status_code=404, detail="Example not found")

    # Clean up files
    for path_str in [example.original_path, example.target_path]:
        if path_str:
            path = Path(path_str)
            if path.exists():
                path.unlink()

    db.delete(example)
    db.commit()

    return {"status": "deleted", "id": example_id}


@router.post("/split", response_model=DatasetStats)
async def split_dataset(
    config: DatasetSplitConfig = DatasetSplitConfig(),
    db: Session = Depends(get_db),
):
    """Assign train/validation/test splits to all labeled examples."""
    return assign_splits(db, config)
