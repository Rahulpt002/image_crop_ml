"""Application configuration."""

import os
from pathlib import Path


# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Storage paths
UPLOAD_DIR = BASE_DIR / "uploads"
DATASET_DIR = BASE_DIR / "dataset"
DATASET_IMAGES_DIR = DATASET_DIR / "images"
MODELS_DIR = BASE_DIR / "models"
EXPORTS_DIR = BASE_DIR / "exports"

# Ensure directories exist
for d in [UPLOAD_DIR, DATASET_IMAGES_DIR, MODELS_DIR, EXPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Database
DATABASE_URL = f"sqlite:///{BASE_DIR / 'ai_crop_trainer.db'}"

# Image processing
MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
THUMBNAIL_SIZE = (256, 256)

# Crop label extraction
CROP_CONFIDENCE_THRESHOLD = float(os.getenv("CROP_CONFIDENCE_THRESHOLD", "0.5"))

# ML defaults
DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_EPOCHS = 30
DEFAULT_OPTIMIZER = "adamw"
DEFAULT_LOSS = "smooth_l1"
DEFAULT_BACKBONE = "mobilenet_v3_small"

# Dataset split
DEFAULT_TRAIN_SPLIT = 0.8
DEFAULT_VAL_SPLIT = 0.1
DEFAULT_TEST_SPLIT = 0.1
SPLIT_RANDOM_SEED = 42

# Portrait crop output
PORTRAIT_ASPECT_RATIO = 3 / 4          # width / height = 0.75
PORTRAIT_OUTPUT_WIDTH = 150            # px
PORTRAIT_OUTPUT_HEIGHT = 200           # px
PORTRAIT_OUTPUT_DPI = (300, 300)       # metadata only
ASPECT_RATIO_TOLERANCE = 0.03         # ±3% tolerance for validation

# Server
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8001"))
CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
