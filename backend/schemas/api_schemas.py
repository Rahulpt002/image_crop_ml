"""Pydantic schemas for API request/response validation."""

from __future__ import annotations

import datetime
from typing import Optional, List

from pydantic import BaseModel, Field, field_validator


# ─── Crop Coordinates ────────────────────────────────────────────────

class CropRect(BaseModel):
    """Absolute pixel crop rectangle."""
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class NormalizedCropRect(BaseModel):
    """Normalized crop rectangle (values 0–1)."""
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)

    @field_validator("width")
    @classmethod
    def validate_width(cls, v, info):
        x = info.data.get("x", 0.0)
        if x + v > 1.0 + 1e-6:
            raise ValueError(f"x + width = {x + v} exceeds 1.0")
        return v

    @field_validator("height")
    @classmethod
    def validate_height(cls, v, info):
        y = info.data.get("y", 0.0)
        if y + v > 1.0 + 1e-6:
            raise ValueError(f"y + height = {y + v} exceeds 1.0")
        return v


class CropExtractionResult(BaseModel):
    """Result of automatic crop label extraction."""
    crop: CropRect
    normalized: NormalizedCropRect
    confidence: float = Field(ge=0.0, le=1.0)
    method: str = "auto"


# ─── Image Info ───────────────────────────────────────────────────────

class ImageInfo(BaseModel):
    """Basic image metadata."""
    filename: str
    path: str
    width: int
    height: int


# ─── Training Example ────────────────────────────────────────────────

class TrainingExampleCreate(BaseModel):
    """Request to create a training example with a manual label."""
    crop_x: int = Field(ge=0)
    crop_y: int = Field(ge=0)
    crop_width: int = Field(gt=0)
    crop_height: int = Field(gt=0)
    extraction_method: str = "manual"
    confidence: float = 1.0
    notes: Optional[str] = None


class TrainingExampleResponse(BaseModel):
    """Full training example in API responses."""
    id: str
    created_at: datetime.datetime
    updated_at: datetime.datetime

    original_filename: str
    original_path: str
    original_width: int
    original_height: int

    target_filename: Optional[str] = None
    target_path: Optional[str] = None
    target_width: Optional[int] = None
    target_height: Optional[int] = None

    crop_x: Optional[int] = None
    crop_y: Optional[int] = None
    crop_width: Optional[int] = None
    crop_height: Optional[int] = None

    norm_x: Optional[float] = None
    norm_y: Optional[float] = None
    norm_width: Optional[float] = None
    norm_height: Optional[float] = None

    extraction_method: str = "auto"
    confidence: Optional[float] = None
    split: str = "unassigned"
    status: str = "pending"
    batch_id: Optional[str] = None
    file_hash: Optional[str] = None
    notes: Optional[str] = None

    class Config:
        from_attributes = True


class TrainingExampleUpdate(BaseModel):
    """Update a training example's crop label."""
    crop_x: Optional[int] = None
    crop_y: Optional[int] = None
    crop_width: Optional[int] = None
    crop_height: Optional[int] = None
    extraction_method: Optional[str] = None
    confidence: Optional[float] = None
    split: Optional[str] = None
    status: Optional[str] = None
    notes: Optional[str] = None


# ─── Dataset ─────────────────────────────────────────────────────────

class DatasetStats(BaseModel):
    """Dataset statistics."""
    total: int = 0
    labeled: int = 0
    pending: int = 0
    train: int = 0
    validation: int = 0
    test: int = 0
    unassigned: int = 0


class DatasetSplitConfig(BaseModel):
    """Dataset split configuration."""
    train_ratio: float = Field(default=0.8, ge=0.0, le=1.0)
    val_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    test_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    seed: int = 42

    @field_validator("test_ratio")
    @classmethod
    def validate_ratios(cls, v, info):
        train = info.data.get("train_ratio", 0.8)
        val = info.data.get("val_ratio", 0.1)
        if abs(train + val + v - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {train + val + v}")
        return v


# ─── Training ────────────────────────────────────────────────────────

class TrainingConfig(BaseModel):
    """Training configuration."""
    backbone: str = "mobilenet_v3_small"
    image_size: int = 224
    batch_size: int = 8
    learning_rate: float = 0.0001
    epochs: int = 30
    optimizer: str = "adamw"
    loss_function: str = "smooth_l1"


class TrainingStatus(BaseModel):
    """Current training status."""
    status: str = "idle"  # idle | training | completed | failed | stopped
    run_id: Optional[str] = None
    epoch: int = 0
    total_epochs: int = 0
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    best_val_loss: Optional[float] = None
    device: Optional[str] = None
    progress: float = 0.0  # 0-100


class TrainingRunResponse(BaseModel):
    """Training run history entry."""
    id: str
    created_at: datetime.datetime
    completed_at: Optional[datetime.datetime] = None
    model_version: str
    backbone: str
    image_size: int
    batch_size: int
    learning_rate: float
    epochs: int
    train_samples: int
    val_samples: int
    test_samples: int
    status: str
    current_epoch: int
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    best_val_loss: Optional[float] = None
    device: Optional[str] = None
    mean_iou: Optional[float] = None

    class Config:
        from_attributes = True


# ─── Model ────────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    """Information about a model version."""
    id: str
    version: str
    created_at: datetime.datetime
    backbone: str
    image_size: int
    val_loss: Optional[float] = None
    mean_iou: Optional[float] = None
    is_active: bool = False
    dataset_size: int = 0

    class Config:
        from_attributes = True


class ModelStatus(BaseModel):
    """Current model status."""
    has_model: bool = False
    active_version: Optional[str] = None
    versions: List[ModelInfo] = []
    device: Optional[str] = None


# ─── Prediction ───────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """Request to predict crop for an image (by upload, not JSON body)."""
    pass


class PredictionResponse(BaseModel):
    """Prediction result."""
    id: str
    image_filename: str
    image_path: str
    image_width: int
    image_height: int
    model_version: Optional[str] = None

    # Predicted normalized crop
    pred_x: Optional[float] = None
    pred_y: Optional[float] = None
    pred_width: Optional[float] = None
    pred_height: Optional[float] = None

    # Absolute pixel crop
    pred_crop_x: Optional[int] = None
    pred_crop_y: Optional[int] = None
    pred_crop_width: Optional[int] = None
    pred_crop_height: Optional[int] = None

    class Config:
        from_attributes = True


class CorrectionRequest(BaseModel):
    """User correction of a prediction."""
    prediction_id: str
    corrected_x: float = Field(ge=0.0, le=1.0)
    corrected_y: float = Field(ge=0.0, le=1.0)
    corrected_width: float = Field(gt=0.0, le=1.0)
    corrected_height: float = Field(gt=0.0, le=1.0)
    add_to_dataset: bool = False
