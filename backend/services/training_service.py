"""Training orchestrator — manages background training jobs."""

import json
import logging
import threading
import datetime
from pathlib import Path
from typing import Optional

from backend.config.settings import MODELS_DIR
from backend.models.db_models import TrainingRun, ModelVersion
from backend.schemas.api_schemas import TrainingConfig, TrainingStatus

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """
    Manages the background training lifecycle.
    
    Only one training job can run at a time.
    Training runs in a background thread so FastAPI stays responsive.
    """

    def __init__(self):
        self._current_status = TrainingStatus(status="idle")
        self._thread: Optional[threading.Thread] = None
        self._stop_requested = False
        self._lock = threading.Lock()

    @property
    def status(self) -> TrainingStatus:
        with self._lock:
            return self._current_status.model_copy()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start_training(self, config: TrainingConfig, db_session_factory) -> str:
        """Start a background training job. Returns the run ID."""
        if self.is_running:
            raise RuntimeError("A training job is already running.")

        self._stop_requested = False

        # Determine model version
        version = self._next_version()

        with self._lock:
            self._current_status = TrainingStatus(
                status="training",
                run_id=None,  # Will be set once DB record is created
                epoch=0,
                total_epochs=config.epochs,
                device=None,
                progress=0.0,
            )

        self._thread = threading.Thread(
            target=self._run_training,
            args=(config, version, db_session_factory),
            daemon=True,
        )
        self._thread.start()

        return version

    def stop_training(self) -> None:
        """Request graceful stop of current training."""
        self._stop_requested = True

    def update_progress(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: float,
        best_val_loss: float,
        device: str,
        run_id: str,
    ):
        """Called by the training loop to report progress."""
        with self._lock:
            self._current_status = TrainingStatus(
                status="training",
                run_id=run_id,
                epoch=epoch,
                total_epochs=total_epochs,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                device=device,
                progress=round((epoch / total_epochs) * 100, 1) if total_epochs > 0 else 0,
            )

    def _run_training(self, config: TrainingConfig, version: str, db_session_factory):
        """The actual training loop, runs in a background thread."""
        try:
            # Import here to avoid circular deps and keep ML separate
            from ml.train import run_training_job

            run_training_job(
                config=config,
                version=version,
                db_session_factory=db_session_factory,
                orchestrator=self,
                stop_check=lambda: self._stop_requested,
            )

            with self._lock:
                self._current_status = TrainingStatus(
                    status="stopped" if self._stop_requested else "completed",
                    run_id=self._current_status.run_id,
                    epoch=self._current_status.epoch,
                    total_epochs=self._current_status.total_epochs,
                    train_loss=self._current_status.train_loss,
                    val_loss=self._current_status.val_loss,
                    best_val_loss=self._current_status.best_val_loss,
                    device=self._current_status.device,
                    progress=100.0 if not self._stop_requested else self._current_status.progress,
                )

        except Exception as e:
            logger.exception(f"Training failed: {e}")
            with self._lock:
                self._current_status = TrainingStatus(
                    status="failed",
                    run_id=self._current_status.run_id,
                )

    def _next_version(self) -> str:
        """Determine the next model version string."""
        existing = sorted(MODELS_DIR.iterdir()) if MODELS_DIR.exists() else []
        version_nums = []
        for d in existing:
            if d.is_dir() and d.name.startswith("v"):
                try:
                    version_nums.append(int(d.name[1:]))
                except ValueError:
                    pass
        next_num = (max(version_nums) + 1) if version_nums else 1
        return f"v{next_num:03d}"


# Singleton instance
training_orchestrator = TrainingOrchestrator()
