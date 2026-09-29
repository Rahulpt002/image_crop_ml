"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config.settings import CORS_ORIGINS, UPLOAD_DIR, EXPORTS_DIR
from backend.database.connection import init_db
from backend.api.dataset import router as dataset_router
from backend.api.training import router as training_router
from backend.api.model import router as model_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized.")

    # Log device info
    try:
        import torch
        if torch.backends.mps.is_available():
            logger.info("🍎 Apple MPS acceleration available")
        elif torch.cuda.is_available():
            logger.info(f"🎮 CUDA available: {torch.cuda.get_device_name()}")
        else:
            logger.info("💻 Running on CPU")
    except ImportError:
        logger.info("⚠️ PyTorch not installed — ML features disabled")

    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="AI Crop Trainer",
    description="Local-first AI image cropping training studio",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving for uploaded images
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/exports", StaticFiles(directory=str(EXPORTS_DIR)), name="exports")

# API routes
app.include_router(dataset_router)
app.include_router(training_router)
app.include_router(model_router)


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    device = "cpu"
    try:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass

    return {
        "status": "ok",
        "device": device,
        "version": "0.1.0",
    }
