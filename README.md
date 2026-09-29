# AI Crop Trainer

A local-first AI image cropping training studio that learns **your** preferred cropping style.

## Concept

You provide pairs of images:
- **Original image** — the full, uncropped photograph
- **Target crop** — your manually cropped version

The system learns the relationship between originals and your crops, then predicts crop rectangles for new images.

```
ORIGINAL IMAGE → AI MODEL → PREDICTED CROP RECTANGLE → AUTO-CROPPED RESULT
```

## Architecture

| Layer | Technology |
|-------|-----------|
| Frontend | React + TypeScript + Vite + Tailwind CSS |
| Backend | Python 3.11+ / FastAPI |
| Database | SQLite |
| ML | PyTorch with pretrained vision backbone |
| Acceleration | Apple MPS → CUDA → CPU |

## Project Structure

```
├── backend/          # FastAPI server
│   ├── api/          # Route handlers
│   ├── services/     # Business logic
│   ├── models/       # SQLAlchemy models
│   ├── schemas/      # Pydantic schemas
│   ├── database/     # DB connection & migrations
│   └── config/       # Configuration
├── frontend/         # React + Vite app
├── ml/               # ML pipeline (separate from API)
│   ├── model.py      # Model architecture
│   ├── train.py      # Training loop
│   ├── inference.py  # Prediction
│   ├── dataset.py    # PyTorch dataset
│   └── evaluate.py   # Metrics (IoU, MAE)
├── dataset/          # Training data storage
├── models/           # Saved model checkpoints
├── uploads/          # Uploaded images
├── exports/          # Exported crops
├── scripts/          # Utility scripts
├── tests/            # Test suite
└── docs/             # Documentation
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Apple Silicon Mac (recommended) or any machine with CUDA/CPU

### Setup & Run

```bash
# One-command start:
./run.sh

# Or manually:

# Backend
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (in another terminal)
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

## How It Works

### 1. Create Training Examples

Go to `/trainer`:
1. Upload an **original image**
2. Upload your **manually cropped version**
3. The system auto-detects the crop rectangle using feature matching
4. Review and adjust the crop rectangle if needed
5. Save the training example

### 2. Build Your Dataset

Go to `/dataset`:
- View all training examples with thumbnails
- Edit or delete examples
- Configure train/validation/test split (default: 80/10/10)

### 3. Train the Model

Go to `/train`:
- Configure hyperparameters (learning rate, epochs, batch size)
- Click "Train Model"
- Watch live training progress with loss curves
- Model trains on MPS (Apple Silicon), CUDA, or CPU automatically

### 4. Test & Use

Go to `/test` or `/crop`:
- Upload a new image
- See the AI-predicted crop rectangle
- Adjust manually if needed
- Download the cropped result
- Optionally save corrections as new training examples

## ML Model

- **Backbone**: MobileNetV3-Small (lightweight, M2-friendly)
- **Head**: Regression head outputting `[x, y, width, height]` (normalized 0–1)
- **Loss**: Smooth L1 Loss
- **Optimizer**: AdamW
- **Output constraint**: Sigmoid activation ensures valid normalized coordinates

The model does NOT perform generic face detection — it learns YOUR specific cropping preferences from YOUR examples.

## Apple Silicon / MPS

The system automatically detects and uses Apple's Metal Performance Shaders:

```python
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
```

### MPS Troubleshooting

- Ensure macOS 12.3+ and PyTorch 2.0+
- If MPS causes issues, set `PYTORCH_DEVICE=cpu` to force CPU
- Some operations may fall back to CPU automatically

## Model Versioning

Models are saved as:
```
models/v001/
  ├── model.pt       # Weights
  ├── config.json    # Training config
  └── metrics.json   # Evaluation results
```

## Future Export

The model architecture is designed to be exportable to:
- **ONNX** — for cross-platform inference
- **Core ML** — for on-device iOS/macOS inference

## License

Private / Internal Use
