# MulTiCheat — AI-Based Exam Cheating Detection System

> Multimodal video analysis system that detects suspicious exam cheating behavior, produces time-stamped flags with annotated evidence, and presents results through an interactive video timeline.

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
