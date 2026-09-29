"""Training loop for CropNet."""

import json
import logging
import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from backend.config.settings import MODELS_DIR
from backend.schemas.api_schemas import TrainingConfig
from ml.model import CropNet, build_model, get_device
from ml.dataset import CropDataset, build_dataloaders
from ml.metrics import evaluate_predictions
from ml.preprocessing import get_preprocessing_config

logger = logging.getLogger(__name__)


def run_training_job(
    config: TrainingConfig,
    version: str,
    db_session_factory,
    orchestrator,
    stop_check: Callable[[], bool],
):
    """
    Full training job. Called from a background thread.
    
    Args:
        config: Training hyperparameters
        version: Model version string (e.g. "v001")
        db_session_factory: SQLAlchemy session factory
        orchestrator: TrainingOrchestrator for progress updates
        stop_check: Callable that returns True if stop is requested
    """
    from backend.models.db_models import TrainingExample, TrainingRun, ModelVersion

    device = get_device()
    logger.info(f"Training on device: {device}")

    # Create a new DB session for this thread
    db = db_session_factory()

    try:
        # Gather training examples
        train_examples = _load_examples(db, "train")
        val_examples = _load_examples(db, "validation")

        if len(train_examples) == 0:
            raise ValueError("No training examples found")

        # Create training run record
        run = TrainingRun(
            model_version=version,
            backbone=config.backbone,
            image_size=config.image_size,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            epochs=config.epochs,
            optimizer=config.optimizer,
            loss_function=config.loss_function,
            train_samples=len(train_examples),
            val_samples=len(val_examples),
            status="training",
            device=str(device),
        )
        db.add(run)
        db.commit()
        db.refresh(run)

        # Build model
        model = build_model(backbone=config.backbone, pretrained=True)
        model = model.to(device)

        # Build data loaders
        train_loader, val_loader = build_dataloaders(
            train_examples, val_examples,
            image_size=config.image_size,
            batch_size=config.batch_size,
        )

        # Loss and optimizer
        criterion = nn.SmoothL1Loss()
        if config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.01
        )

        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0
        version_dir = MODELS_DIR / version
        version_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, config.epochs + 1):
            if stop_check():
                logger.info("Training stopped by user")
                run.status = "stopped"
                break

            # Train
            train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)

            # Validate
            val_loss = float("inf")
            if len(val_loader) > 0:
                val_loss = _validate_epoch(model, val_loader, criterion, device)

            scheduler.step()

            # Track best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), version_dir / "best_model.pt")
            else:
                patience_counter += 1

            # Save latest
            torch.save(model.state_dict(), version_dir / "latest_model.pt")

            # Update progress
            orchestrator.update_progress(
                epoch=epoch,
                total_epochs=config.epochs,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                device=str(device),
                run_id=run.id,
            )

            # Update DB
            run.current_epoch = epoch
            run.train_loss = train_loss
            run.val_loss = val_loss
            run.best_val_loss = best_val_loss
            db.commit()

            logger.info(
                f"Epoch {epoch}/{config.epochs} — "
                f"train_loss: {train_loss:.5f}, val_loss: {val_loss:.5f}, "
                f"best: {best_val_loss:.5f}"
            )

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Final evaluation
        if val_examples:
            model.load_state_dict(torch.load(version_dir / "best_model.pt", weights_only=True))
            metrics = _evaluate_model(model, val_loader, device)
            run.mean_iou = metrics["mean_iou"]
            run.mae_x = metrics["mae_x"]
            run.mae_y = metrics["mae_y"]
            run.mae_width = metrics["mae_width"]
            run.mae_height = metrics["mae_height"]

        config_data = config.model_dump()
        config_data["device"] = str(device)
        config_data["version"] = version
        config_data["preprocessing"] = get_preprocessing_config(config.image_size)
        (version_dir / "config.json").write_text(json.dumps(config_data, indent=2))

        if val_examples:
            (version_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        # Create model version record
        model_version = ModelVersion(
            version=version,
            training_run_id=run.id,
            backbone=config.backbone,
            image_size=config.image_size,
            model_path=str(version_dir / "best_model.pt"),
            config_path=str(version_dir / "config.json"),
            metrics_path=str(version_dir / "metrics.json") if val_examples else None,
            val_loss=best_val_loss if best_val_loss != float("inf") else None,
            mean_iou=metrics.get("mean_iou") if val_examples else None,
            is_active=1,  # New model becomes active
            dataset_size=len(train_examples) + len(val_examples),
        )

        # Deactivate previous models
        db.query(ModelVersion).update({ModelVersion.is_active: 0})
        db.add(model_version)

        if run.status != "stopped":
            run.status = "completed"
        run.completed_at = datetime.datetime.utcnow()
        run.model_path = str(version_dir / "best_model.pt")
        db.commit()

        logger.info(f"Training completed: version={version}, best_val_loss={best_val_loss:.5f}")

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        try:
            run.status = "failed"
            run.error_message = str(e)
            db.commit()
        except Exception:
            pass
        raise

    finally:
        db.close()


def _load_examples(db, split: str):
    """Load examples for a given split as a list of dicts."""
    from backend.models.db_models import TrainingExample

    examples = (
        db.query(TrainingExample)
        .filter(
            TrainingExample.status == "labeled",
            TrainingExample.split == split,
        )
        .all()
    )

    return [
        {
            "original_path": e.original_path,
            "norm_x": e.norm_x,
            "norm_y": e.norm_y,
            "norm_width": e.norm_width,
            "norm_height": e.norm_height,
        }
        for e in examples
        if e.norm_x is not None
    ]


def _train_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch, return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def _validate_epoch(model, loader, criterion, device):
    """Run validation, return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            loss = criterion(predictions, labels)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def _evaluate_model(model, loader, device):
    """Evaluate model and return metrics dict."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            predictions = model.predict_with_validation(images)

            all_preds.append(predictions.cpu().numpy())
            all_targets.append(labels.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return evaluate_predictions(preds, targets)
