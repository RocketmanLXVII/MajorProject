# How to Run MulTiCheat Pro on Any PC / Laptop

This guide provides step-by-step instructions to clone, set up, and run **MulTiCheat Pro** on any operating system (**macOS, Windows, or Linux**).

---

## Prerequisites

Before starting, ensure you have the following installed on your machine:

1. **Python**: Python `3.10` or higher (`3.11`, `3.12`, or `3.13` recommended). Check version with:
   ```bash
   python3 --version
   ```
2. **Node.js & npm**: Node.js `v18.0.0` or higher. Check version with:
   ```bash
   node -v
   npm -v
   ```
3. **Git**: Installed and available in terminal.

---

## 🚀 Quick Start Guide (Standard Local Setup)

### Step 1: Clone the Repository

Open your terminal or command prompt and clone the repository:

```bash
git clone https://github.com/RocketmanLXVII/MajorProject.git
cd MajorProject
```

---

### Step 2: Set Up the Backend Environment

1. **Create a Python Virtual Environment**:
   - **macOS / Linux**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt)**:
     ```cmd
     python -m venv .venv
     .venv\Scripts\activate.bat
     ```

2. **Install Backend Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Pre-download Vision Models** *(Optional - models auto-download on first run)*:
   ```bash
   python download_models.py
   ```

---

### Step 3: Set Up the Frontend Environment

Open a new terminal window or navigate to the `frontend/` directory:

```bash
cd frontend
npm install
cd ..
```

---

### Step 4: Launch the Application

You will need **two open terminal windows** (one for backend, one for frontend):

#### Terminal 1: Backend API Server
*Make sure your virtual environment (`.venv`) is activated in the project root directory.*

```bash
# macOS / Linux
.venv/bin/uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Windows
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
- **Backend API**: `http://localhost:8000`
- **Interactive Swagger API Docs**: `http://localhost:8000/docs`

#### Terminal 2: Frontend React Web App
*Navigate to the `frontend/` directory.*

```bash
cd frontend
npm run dev
```
- **Frontend App**: `http://localhost:5173`

---

## 🐳 Docker Deployment (Alternative Setup)

If you prefer using Docker and Docker Compose:

```bash
# Build and launch all services in detached mode
docker-compose up --build -d

# View service logs
docker-compose logs -f
```

- **Frontend App**: `http://localhost:5173`
- **Backend API**: `http://localhost:8000`

---

## 🧪 Running the Automated Test Suite

To verify that all perception, tracking, 3D gaze, and API endpoints are working properly:

```bash
# Ensure .venv is activated from project root
pytest tests/ -v
```

Expected Output:
```
======================== 78 passed in 7.25s ========================
```

---

## 📂 Testing Video File

A sample video (`Test Vid.mp4`) is included in the project root directory:
- You can upload `Test Vid.mp4` directly via the web UI at `http://localhost:5173`.
- The system will process the video, perform 3D gaze estimation, extract 17-point pose keypoints, and highlight any malpractice alerts in real-time.

---

## 🛠 Troubleshooting

1. **OpenCV GUI / Headless Error on Linux**:
   If running on Ubuntu/Debian server and OpenCV throws a display error:
   ```bash
   sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
   ```
2. **Port 8000 or 5173 already in use**:
   Change the port in command:
   ```bash
   uvicorn backend.main:app --reload --port 8080
   ```
   and update Vite proxy if needed.
