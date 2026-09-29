"""Training API routes."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.connection import get_db, SessionLocal
from backend.models.db_models import TrainingRun
from backend.schemas.api_schemas import TrainingConfig, TrainingStatus, TrainingRunResponse
from backend.services.training_service import training_orchestrator
from backend.services.dataset_service import get_dataset_stats

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/training", tags=["training"])


@router.post("/start")
async def start_training(
    config: TrainingConfig = TrainingConfig(),
    db: Session = Depends(get_db),
):
    """Start a new training run in the background."""
    # Check we have data
    stats = get_dataset_stats(db)
    if stats.labeled == 0:
        raise HTTPException(status_code=400, detail="No labeled examples in dataset")
    if stats.train == 0:
        raise HTTPException(
            status_code=400,
            detail="No training examples assigned. Run dataset split first."
        )

    try:
        version = training_orchestrator.start_training(config, SessionLocal)
        return {
            "status": "started",
            "model_version": version,
            "config": config.model_dump(),
        }
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/stop")
async def stop_training():
    """Request graceful stop of current training."""
    if not training_orchestrator.is_running:
        raise HTTPException(status_code=400, detail="No training job is running")

    training_orchestrator.stop_training()
    return {"status": "stop_requested"}


@router.get("/status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training status."""
    return training_orchestrator.status


@router.get("/history", response_model=list[TrainingRunResponse])
async def get_training_history(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
):
    """Get training run history."""
    runs = (
        db.query(TrainingRun)
        .order_by(TrainingRun.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return runs
