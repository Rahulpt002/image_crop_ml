"""SQLAlchemy ORM models."""

import datetime
import uuid

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text, ForeignKey, Enum as SAEnum
)
from sqlalchemy.orm import relationship

from backend.database.connection import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class TrainingExample(Base):
    """A single training example: original + target crop label."""
    __tablename__ = "training_examples"

    id = Column(String, primary_key=True, default=generate_uuid)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Original image
    original_filename = Column(String, nullable=False)
    original_path = Column(String, nullable=False)
    original_width = Column(Integer, nullable=False)
    original_height = Column(Integer, nullable=False)

    # Target crop image (user's manually cropped version)
    target_filename = Column(String, nullable=True)
    target_path = Column(String, nullable=True)
    target_width = Column(Integer, nullable=True)
    target_height = Column(Integer, nullable=True)

    # Crop rectangle (absolute pixels relative to original)
    crop_x = Column(Integer, nullable=True)
    crop_y = Column(Integer, nullable=True)
    crop_width = Column(Integer, nullable=True)
    crop_height = Column(Integer, nullable=True)

    # Normalized crop rectangle (0–1)
    norm_x = Column(Float, nullable=True)
    norm_y = Column(Float, nullable=True)
    norm_width = Column(Float, nullable=True)
    norm_height = Column(Float, nullable=True)

    # Extraction metadata
    extraction_method = Column(String, default="auto")  # auto | manual
    confidence = Column(Float, nullable=True)

    # Dataset split assignment
    split = Column(String, default="unassigned")  # train | validation | test | unassigned

    # Status
    status = Column(String, default="pending")  # pending | labeled | error

    # Optional notes
    notes = Column(Text, nullable=True)


class TrainingRun(Base):
    """Record of a training run."""
    __tablename__ = "training_runs"

    id = Column(String, primary_key=True, default=generate_uuid)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Configuration
    model_version = Column(String, nullable=False)
    backbone = Column(String, default="mobilenet_v3_small")
    image_size = Column(Integer, default=224)
    batch_size = Column(Integer, default=8)
    learning_rate = Column(Float, default=0.0001)
    epochs = Column(Integer, default=30)
    optimizer = Column(String, default="adamw")
    loss_function = Column(String, default="smooth_l1")

    # Dataset info
    train_samples = Column(Integer, default=0)
    val_samples = Column(Integer, default=0)
    test_samples = Column(Integer, default=0)

    # Results
    status = Column(String, default="pending")  # pending | training | completed | failed | stopped
    current_epoch = Column(Integer, default=0)
    train_loss = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)
    best_val_loss = Column(Float, nullable=True)
    device = Column(String, nullable=True)

    # Metrics
    mean_iou = Column(Float, nullable=True)
    mae_x = Column(Float, nullable=True)
    mae_y = Column(Float, nullable=True)
    mae_width = Column(Float, nullable=True)
    mae_height = Column(Float, nullable=True)

    # Paths
    model_path = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)


class ModelVersion(Base):
    """A saved model version."""
    __tablename__ = "model_versions"

    id = Column(String, primary_key=True, default=generate_uuid)
    version = Column(String, nullable=False, unique=True)  # e.g. "v001"
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Training info
    training_run_id = Column(String, ForeignKey("training_runs.id"), nullable=True)
    backbone = Column(String, nullable=False)
    image_size = Column(Integer, nullable=False)

    # Paths
    model_path = Column(String, nullable=False)
    config_path = Column(String, nullable=True)
    metrics_path = Column(String, nullable=True)

    # Metrics
    val_loss = Column(Float, nullable=True)
    mean_iou = Column(Float, nullable=True)

    # Status
    is_active = Column(Integer, default=0)  # 1 = active model for inference

    # Dataset stats
    dataset_size = Column(Integer, default=0)

    training_run = relationship("TrainingRun", backref="saved_model")


class Prediction(Base):
    """A prediction made by the model, optionally with user correction."""
    __tablename__ = "predictions"

    id = Column(String, primary_key=True, default=generate_uuid)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Input image
    image_filename = Column(String, nullable=False)
    image_path = Column(String, nullable=False)
    image_width = Column(Integer, nullable=False)
    image_height = Column(Integer, nullable=False)

    # Model used
    model_version = Column(String, nullable=True)

    # AI prediction (normalized)
    pred_x = Column(Float, nullable=True)
    pred_y = Column(Float, nullable=True)
    pred_width = Column(Float, nullable=True)
    pred_height = Column(Float, nullable=True)

    # User correction (normalized), if any
    corrected_x = Column(Float, nullable=True)
    corrected_y = Column(Float, nullable=True)
    corrected_width = Column(Float, nullable=True)
    corrected_height = Column(Float, nullable=True)

    # Whether this was added back to the training dataset
    added_to_dataset = Column(Integer, default=0)
    training_example_id = Column(String, ForeignKey("training_examples.id"), nullable=True)

    # Export path
    export_path = Column(String, nullable=True)
