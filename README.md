# MulTiCheat — AI-Based Exam Cheating Detection System

> State-of-the-art multimodal video analysis system that detects suspicious exam cheating behavior inside dense 60+ person classrooms. It produces time-stamped flags with annotated evidence, geometric spatial mapping, and presents results through an interactive video timeline.

## Features

- **Dense Multi-Person Tracking (BoT-SORT)**: Utilizes deep Visual Re-Identification (ReID) to maintain persistent student tracking IDs (`track_id`) even during severe occlusion or overlapping camera angles.
- **High-Fidelity Model Inference**: Employs `yolov8m` (Medium) architectures overriding default compression matrices with native 1080p retention (`imgsz=1280`), guaranteeing detection of tiny unauthorized objects (phones/books) for students in the furthest back rows.
- **Geometric Interaction Engine**: Calculates multi-person spatial interactions instead of isolated bounding boxes. Automatically detects relativistic behaviors such as:
  - **Paper Copying**: Projects dynamic 2D head-yaw vectors casting spatial rays to intersect neighboring desk planes.
  - **Note Passing**: Tracks synchronized anomalous wrist proximity dropping below extreme mathematical thresholds between distinct entity IDs.
- **Interactive Proctor Timeline**: A React-based diagnostic UI featuring an HTML5 Canvas skeleton overlay mimicking the native ML matrices seamlessly over the tracking feed.
- **Automated Evidence Bundling**: Zero-click automated exporting of suspicious timestamps with max-confidence visual evidence into proctor-ready print PDFs, CSVs, and localized Zips.

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm
- Git

### Backend Setup
```bash
# Clone / navigate to the project
cd "Major Project"

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Copy environment config
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux

# Run the backend
python -m uvicorn backend.main:app --reload
```

The API will be available at **http://localhost:8000**.  
Interactive docs at **http://localhost:8000/docs**.

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at **http://localhost:5173**.

### Run Tests
```bash
# From project root
pip install pytest-asyncio
python -m pytest tests/ -v
```

## Project Structure
```
Major Project/
├── backend/            # FastAPI backend
│   ├── main.py         # App entry point
│   ├── config.py       # Configuration management
│   └── routers/        # API route modules
├── frontend/           # React + TypeScript UI
├── tests/              # pytest test suite
├── docs/report/        # IEEE project report
├── uploads/            # Uploaded videos (runtime)
├── evidence/           # Generated evidence (runtime)
├── config.yaml         # Default configuration
├── .env.example        # Environment variable template
└── requirements.txt    # Python dependencies
```

## API Endpoints
| Method | Path      | Description          |
|--------|-----------|----------------------|
| GET    | `/`       | API info             |
| GET    | `/health` | Health check         |

## Configuration
Settings are loaded with the following priority:
1. Environment variables (`MULTICHEAT_*`)
2. `.env` file
3. `config.yaml`
4. Built-in defaults

## License
This project is for academic and research purposes.
