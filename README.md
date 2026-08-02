# 🐀 RATTUS AI — Leptospirosis Environmental Exposure-Risk Engine

**RATTUS AI** is a real-time computer vision dashboard and risk intelligence engine designed to assess hyperlocal leptospirosis exposure risks. It combines live YOLOv8 object detection, pre-recorded video analysis, Open-Meteo weather API data, and field observation parameters to compute automated risk scores and issue municipal/citizen guidance.

---

## ✨ Key Features

- **⚡ Decoupled Asynchronous Vision Engine (30 FPS)**: Multithreaded architecture separating camera frame capture, YOLO inference, and web video streaming to guarantee fluid 30 FPS live feedback without stutter.
- **🐀 Fine-Tuned YOLOv8 Rodent Detection**: Defaults to fine-tuned rodent model weights (`runs/detect/runs/rat_yolov8/weights/best.pt`) with real-time detection counting and bounding box HUD overlays.
- **🛡️ Dual-Model Human False-Positive Suppression**: Employs a secondary COCO detector (`yolov8n.pt`) and bounding box aspect-ratio filters to prevent humans, faces, and clothing from being falsely flagged as rodents.
- **📹 Pre-Recorded Video Upload Analysis**: Upload `.mp4`, `.avi`, or `.mov` videos simulating rats running in drains or urban environments to test the AI detection pipeline.
- **⏻ Manual Camera Power Control**: Integrated ON/OFF toggle switch to pause vision capture and release camera hardware resources.
- **📊 Dynamic Leptospirosis Risk Matrix**:
  - **Rodent Activity Index** (35% weight)
  - **Recent Rainfall Index** (30% weight via live Open-Meteo API)
  - **Drain-Flow Risk** (25% weight via field slider)
  - **Historical Baseline Risk** (10% weight)
  - **Synergy Elevation**: Automatic risk score boost when heavy rain, drain blockage, and high rodent density overlap.
- **🏛️ Municipal & Citizen Guidance**: Outputs actionable response steps for city workers and residents based on the active risk level (*Low, Moderate, High, Critical*).

---

## 🛠️ Architecture Overview

- **Backend**: Python 3.10+, [FastAPI](https://fastapi.tiangolo.com/), [Ultralytics YOLOv8](https://docs.ultralytics.com/), [OpenCV](https://opencv.org/), PyTorch.
- **Frontend**: HTML5, Modern CSS3 (Dark Mode aesthetic with Outfit & JetBrains Mono typography), Asynchronous JavaScript.
- **API Protocol**: RESTful JSON endpoints + MJPEG (`multipart/x-mixed-replace`) video streaming.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10 or higher
- Webcam or IP Camera (optional, can use video upload feature)

### 2. Installation

Clone the repository and set up a Python virtual environment:

```powershell
# Clone repo
git clone https://github.com/Dhaneesh05/RATTUS_AI.git
cd RATTUS_AI

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Server

Start the FastAPI application server:

```powershell
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Open your web browser and navigate to:
👉 **`http://localhost:8000`**

---

## 📡 API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Serves the web dashboard UI |
| `/api/stream` | `GET` | Live MJPEG video stream with bounding boxes |
| `/api/stats` | `GET` | Returns live rodent count, FPS, and session metrics |
| `/api/risk` | `POST` | Calculates risk score, risk level, and action plans |
| `/api/config` | `GET / POST` | Reads or updates AI confidence threshold, camera source, and model weights |
| `/api/upload_video` | `POST` | Uploads a video file (`.mp4`) and sets it as the active stream source |
| `/api/weather` | `GET` | Fetches 24h rainfall data from Open-Meteo weather forecast API |

---

## 📂 Project Structure

```text
ActionHackathonYoloV8/
├── server.py               # FastAPI web server & API router
├── vision_engine.py        # Multithreaded YOLOv8 vision engine & camera manager
├── static/
│   ├── index.html          # Web dashboard layout
│   ├── style.css           # Dark mode styling & risk gauges
│   └── app.js             # Async frontend client script
├── runs/
│   └── detect/runs/rat_yolov8/weights/best.pt  # Fine-tuned rat weights
├── requirements.txt        # Python dependency manifest
├── .gitignore              # Git ignore configuration
└── README.md               # Documentation
```

---

## 📜 License

Distributed under the MIT License.
