# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational aneurysm detection system for brain CT images using a hybrid Python/C++ architecture. Includes a FastAPI web application for managing analyses. **Not for clinical use.**

## Commands

### Run Detection (CLI)
```bash
# Single image
python main.py --image data/raw/image.jpg

# Directory of images
python main.py --dir data/raw/

# With visualization
python main.py --image data/raw/image.jpg --visualize --save-viz

# Benchmark
python main.py --benchmark --image data/raw/image.jpg
```

### Web Application
```bash
# Install dependencies
pip install -r webapp_requirements.txt

# Start development server
uvicorn webapp.main:app --reload --port 8000

# Access at http://localhost:8000
```

### Run Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api/test_auth_routes.py -v

# Run single test
pytest tests/test_api/test_auth_routes.py::TestLogin::test_login_success -v

# Generate test reports
pytest tests/ --junitxml=test-report.xml --html=test-report.html

# With coverage
pytest tests/ --cov=webapp --cov-report=html
```

### Build C++ Module (Optional)
```bash
# Requires Visual Studio 2022 Build Tools and OpenCV 4.9.0 at C:\opencv
build_opencv.bat
```

### Training (Requires Python 3.11/3.12)
```bash
cd training
training_env\Scripts\activate
python train_model.py --data_dir ../data/processed --epochs 50
python export_onnx.py --checkpoint models/best_model.pth
```

## Architecture

### Detection Pipeline
```
Input Image
    ↓
┌─────────────────────────────────────┐
│ Preprocessing (Python+OpenCV)       │  ~4ms
│ - Grayscale, CLAHE, Resize 224x224  │
│ - Normalize to (1,3,224,224) tensor │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Inference Engine (auto-selected)    │  ~20ms
│ 1. ONNX Runtime (preferred)         │
│ 2. OpenCV DNN (Python 3.14+)        │
│ 3. Heuristic fallback (no model)    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Post-processing (Python)            │
│ - Softmax, NMS, Detection filtering │
└─────────────────────────────────────┘
    ↓
Output: {has_aneurysm, confidence, timings}
```

### Web Application Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Jinja2 + HTMX)                 │
│  Templates: login, dashboard, upload, results, history, admin   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                          │
│  ├─ auth/        → Login, logout, register (session cookies)    │
│  ├─ api/         → REST endpoints (analyses, sessions, admin)   │
│  └─ services/    → AnalysisService (wraps DetectionPipeline)    │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ SQLite Database │  │ DetectionPipeline│  │ File Storage    │
│ (users, analyses│  │ (src/inference/) │  │ (webapp/uploads)│
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Key Files

### CLI/Core
| File | Purpose |
|------|---------|
| `main.py` | CLI entry point |
| `src/inference/pipeline.py` | Main `DetectionPipeline` class |
| `src/inference/onnx_inference.py` | `create_inference_engine()` factory |
| `src/visualization/overlay.py` | Draw detections, generate reports |
| `config.yaml` | All configuration (preprocessing, model, thresholds) |

### Web Application
| File | Purpose |
|------|---------|
| `webapp/main.py` | FastAPI app entry point, page routes |
| `webapp/database/models.py` | SQLAlchemy models (User, Analysis, AnalysisSession) |
| `webapp/auth/security.py` | Password hashing (PBKDF2-SHA256), session tokens |
| `webapp/auth/dependencies.py` | `get_current_user` dependency for auth |
| `webapp/services/analysis_service.py` | Wraps DetectionPipeline for web use |
| `webapp/api/admin.py` | Admin endpoints (tests runner, database explorer) |

### Testing
| File | Purpose |
|------|---------|
| `tests/conftest.py` | Pytest fixtures (test_db, client, authenticated_client) |
| `tests/test_api/` | API route tests |
| `tests/test_auth/` | Authentication/security tests |
| `tests/test_database/` | Model and relationship tests |

## Database Schema

Four tables: `users`, `analyses`, `analysis_sessions`, `user_sessions`
- Users have many Analyses and AnalysisSessions (cascade delete)
- AnalysisSessions group multiple Analyses
- UserSessions store HTTP session tokens for cookie-based auth

## Two-Environment Setup

- **Main environment (Python 3.14)**: Runs inference with OpenCV DNN, webapp with FastAPI
- **Training environment (Python 3.11)**: Located at `training/training_env/`, uses PyTorch/timm

## Model

- Architecture: MobileNetV3 (2 classes: normal, aneurysm)
- Input: `(1, 3, 224, 224)` float32 tensor
- Location: `models/onnx/mobilenetv3_aneurysm.onnx`
- Current model is ImageNet-pretrained; train on ADAM Challenge dataset for real detection

## Notes

- Password hashing uses PBKDF2-SHA256 (bcrypt incompatible with Python 3.14)
- Admin users (`is_admin=True`) can access `/admin/tests` and `/admin/database`
- Tests use in-memory SQLite database via pytest fixtures
