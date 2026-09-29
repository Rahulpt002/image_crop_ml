"""Model & prediction API routes."""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.models.db_models import ModelVersion, Prediction, TrainingExample
from backend.schemas.api_schemas import (
    ModelStatus,
    ModelInfo,
    PredictionResponse,
    CorrectionRequest,
)
from backend.services.image_service import save_uploaded_image, validate_image_file, crop_image
from backend.config.settings import EXPORTS_DIR

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/model", tags=["model"])


@router.get("/status", response_model=ModelStatus)
async def get_model_status(db: Session = Depends(get_db)):
    """Get available models and active model info."""
    versions = (
        db.query(ModelVersion)
        .order_by(ModelVersion.created_at.desc())
        .all()
    )

    active = next((v for v in versions if v.is_active), None)

    # Detect device
    device = "cpu"
    try:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
    except Exception:
        pass

    return ModelStatus(
        has_model=len(versions) > 0,
        active_version=active.version if active else None,
        versions=[
            ModelInfo(
                id=v.id,
                version=v.version,
                created_at=v.created_at,
                backbone=v.backbone,
                image_size=v.image_size,
                val_loss=v.val_loss,
                mean_iou=v.mean_iou,
                is_active=bool(v.is_active),
                dataset_size=v.dataset_size,
            )
            for v in versions
        ],
        device=device,
    )


@router.post("/activate/{version}")
async def activate_model(version: str, db: Session = Depends(get_db)):
    """Set a model version as the active one for inference."""
    model = db.query(ModelVersion).filter(ModelVersion.version == version).first()
    if not model:
        raise HTTPException(status_code=404, detail=f"Model version {version} not found")

    # Deactivate all, activate the selected one
    db.query(ModelVersion).update({ModelVersion.is_active: 0})
    model.is_active = 1
    db.commit()

    return {"status": "activated", "version": version}


@router.post("/predict", response_model=PredictionResponse)
async def predict_crop(
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Predict crop rectangle for a new image using the active model."""
    # Validate and save
    content = await image.read()
    validate_image_file(image.filename, len(content))
    name, path, width, height = save_uploaded_image(content, image.filename, subfolder="predictions")

    # Get active model
    active_model = db.query(ModelVersion).filter(ModelVersion.is_active == 1).first()
    if not active_model:
        raise HTTPException(status_code=400, detail="No active model. Train a model first.")

    # Run inference
    try:
        from ml.inference import predict_crop as ml_predict
        pred = ml_predict(
            str(path),
            active_model.model_path,
            active_model.image_size,
            backbone=active_model.backbone,
        )
    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Create prediction record
    prediction = Prediction(
        image_filename=image.filename,
        image_path=str(path),
        image_width=width,
        image_height=height,
        model_version=active_model.version,
        pred_x=pred["x"],
        pred_y=pred["y"],
        pred_width=pred["width"],
        pred_height=pred["height"],
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    return PredictionResponse(
        id=prediction.id,
        image_filename=prediction.image_filename,
        image_path=prediction.image_path,
        image_width=prediction.image_width,
        image_height=prediction.image_height,
        model_version=prediction.model_version,
        pred_x=prediction.pred_x,
        pred_y=prediction.pred_y,
        pred_width=prediction.pred_width,
        pred_height=prediction.pred_height,
        pred_crop_x=int(prediction.pred_x * width),
        pred_crop_y=int(prediction.pred_y * height),
        pred_crop_width=int(prediction.pred_width * width),
        pred_crop_height=int(prediction.pred_height * height),
    )


@router.post("/correct")
async def save_correction(
    correction: CorrectionRequest,
    db: Session = Depends(get_db),
):
    """Save a user correction of an AI prediction."""
    prediction = db.query(Prediction).filter(Prediction.id == correction.prediction_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    prediction.corrected_x = correction.corrected_x
    prediction.corrected_y = correction.corrected_y
    prediction.corrected_width = correction.corrected_width
    prediction.corrected_height = correction.corrected_height

    # Optionally add to training dataset
    if correction.add_to_dataset:
        example = TrainingExample(
            original_filename=prediction.image_filename,
            original_path=prediction.image_path,
            original_width=prediction.image_width,
            original_height=prediction.image_height,
            crop_x=int(correction.corrected_x * prediction.image_width),
            crop_y=int(correction.corrected_y * prediction.image_height),
            crop_width=int(correction.corrected_width * prediction.image_width),
            crop_height=int(correction.corrected_height * prediction.image_height),
            norm_x=correction.corrected_x,
            norm_y=correction.corrected_y,
            norm_width=correction.corrected_width,
            norm_height=correction.corrected_height,
            extraction_method="correction",
            confidence=1.0,
            status="labeled",
            notes=f"Correction of prediction {prediction.id}",
        )
        db.add(example)
        prediction.added_to_dataset = 1
        prediction.training_example_id = example.id

    db.commit()

    return {"status": "saved", "added_to_dataset": correction.add_to_dataset}


@router.post("/export/{prediction_id}")
async def export_crop(prediction_id: str, db: Session = Depends(get_db)):
    """Export a cropped image based on a prediction."""
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    # Use corrected values if available, otherwise prediction
    nx = prediction.corrected_x if prediction.corrected_x is not None else prediction.pred_x
    ny = prediction.corrected_y if prediction.corrected_y is not None else prediction.pred_y
    nw = prediction.corrected_width if prediction.corrected_width is not None else prediction.pred_width
    nh = prediction.corrected_height if prediction.corrected_height is not None else prediction.pred_height

    if nx is None:
        raise HTTPException(status_code=400, detail="No crop coordinates available")

    # Convert to absolute
    x = int(nx * prediction.image_width)
    y = int(ny * prediction.image_height)
    w = int(nw * prediction.image_width)
    h = int(nh * prediction.image_height)

    # Crop
    source = Path(prediction.image_path)
    cropped = crop_image(source, x, y, w, h)

    export_name = f"crop_{prediction.id[:8]}.jpg"
    export_path = EXPORTS_DIR / export_name
    cropped.save(export_path, quality=95)

    prediction.export_path = str(export_path)
    db.commit()

    return {"status": "exported", "path": str(export_path), "filename": export_name}
