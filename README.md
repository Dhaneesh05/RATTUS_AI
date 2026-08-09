# 🐀 RATTUS AI — Hyperlocal Leptospirosis Exposure-Risk Intelligence Engine

**RATTUS AI** is a real-time computer vision dashboard, geospatial intelligence, and environmental risk assessment engine designed to mitigate leptospirosis transmission risks. By combining fine-tuned YOLOv8 object detection, pre-recorded video simulation, live Open-Meteo weather API rainfall tracking, interactive geospatial drain monitoring, and dynamic risk scoring, RATTUS AI delivers automated exposure metrics alongside actionable municipal and citizen guidance.

---

## ✨ Key Features

- **⚡ Decoupled Asynchronous Vision Engine (30 FPS)**: Multithreaded architecture separating camera frame capture, YOLO inference, and web video streaming to guarantee fluid, stutter-free 30 FPS live feedback.
- **🐀 Fine-Tuned YOLOv8 Rodent Detection**: Utilizes specialized rodent model weights (`runs/detect/runs/rat_yolov8/weights/best.pt`) with real-time frame-by-frame counting and HUD bounding box overlays.
- **🛡️ Human False-Positive Suppression**: Dual-model aspect-ratio filtering and secondary COCO detector (`yolov8n.pt`) validation to prevent humans, hands, or clothing from triggering false rodent alarms.
- **📹 Video Upload Analysis**: Upload `.mp4`, `.avi`, or `.mov` field footage to test urban rat detection without requiring live physical cameras.
- **🗺️ Interactive Geospatial Drain Map**: Built-in Leaflet.js interactive map monitoring citywide drain nodes (e.g. Kuala Lumpur zone) with color-coded risk markers, popup telemetry, and a synchronized node table directory.
- **🌦️ Multi-Location Weather Integration**: Syncs 24-hour accumulated rainfall and ambient temperature via the Open-Meteo API across preset Malaysian urban hubs (Kuala Lumpur, Penang, Johor Bahru, Shah Alam, Ipoh) or custom coordinates.
- **📊 Dynamic Leptospirosis Risk Matrix**:
  - **Rodent Activity Index** (35% weight)
  - **Recent Rainfall Index** (30% weight via Open-Meteo API)
  - **Drain-Flow Water Level** (25% weight via field slider/sensor indicator)
  - **Historical Area Baseline Risk** (10% weight)
  - **Synergy Risk Elevation**: Automatic +10 point risk boost when heavy rain, high water level, and active rodent populations coincide.
- **🏛️ Actionable Municipal & Citizen Guidance**: Generates tailored operational instructions for city council sanitation crews and safety alerts for residents (*Low, Moderate, High, Critical*).
- **📄 Printable PDF Incident Report Generator**: Client-side PDF generator producing printable, timestamped exposure-risk incident reports.
- **☁️ Supabase Audit Trail (Optional)**: Connects to Supabase to persist risk assessment records in cloud database tables.

---

## 🛠️ Architecture & Tech Stack

- **Backend**: Python 3.10+, [FastAPI](https://fastapi.tiangolo.com/), [Ultralytics YOLOv8](https://docs.ultralytics.com/), [OpenCV](https://opencv.org/), PyTorch, [Supabase Python SDK](https://supabase.com/).
- **Frontend**: Responsive HTML5, Vanilla CSS3 (Dark-mode theme with Outfit & JetBrains Mono typography), Asynchronous JavaScript, Leaflet.js.
- **Streamlit App Alternative**: Includes `app.py` for standalone Streamlit dashboard deployment.
- **API Protocol**: RESTful JSON endpoints + MJPEG (`multipart/x-mixed-replace`) streaming.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10 or higher
- Webcam, IP Camera URL, or pre-recorded MP4 video file

### 2. Installation

Clone the repository and set up a Python virtual environment:

```powershell
# Clone repo
git clone https://github.com/Dhaneesh05/RATTUS_AI.git
cd RATTUS_AI

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Windows PowerShell
# source .venv/bin/activate    # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. (Optional) Configuration
Copy `.env` and fill in Supabase credentials if database audit logging is desired:
```env
SUPABASE_URL=https://your-supabase-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key
```

### 4. Running the Web Application Server

Launch the FastAPI application server:

```powershell
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser and navigate to:
👉 **`http://localhost:8000`**

---

## 📡 API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Serves the web dashboard interface |
| `/api/stream` | `GET` | Live MJPEG annotated video stream |
| `/api/stats` | `GET` | Returns live detection FPS, current count, max count, and average confidence |
| `/api/risk` | `POST` | Calculates risk score, risk level band, synergy boost, and municipal/citizen action guidance |
| `/api/config` | `GET / POST` | Reads or updates AI confidence threshold, false-positive suppression, input source, and model weights |
| `/api/upload_video` | `POST` | Uploads a video file (`.mp4`) and switches vision source to the uploaded video |
| `/api/weather` | `GET` | Fetches 24h rainfall & ambient weather data from Open-Meteo for selected city presets or lat/lon |
| `/api/nodes` | `GET` | Returns geospatial drain monitoring nodes with computed LERS risk scores for interactive mapping |

---

## 📂 Project Structure

```text
ActionHackathonYoloV8/
├── server.py               # FastAPI web server, API routing & Supabase integration
├── vision_engine.py        # Multithreaded YOLOv8 vision engine & camera frame pipeline
├── app.py                  # Alternative Streamlit dashboard UI
├── static/
│   ├── index.html          # Web dashboard layout & multi-tab UI
│   ├── style.css           # Modern dark-mode styling, gauges & cards
│   ├── app.js              # Asynchronous frontend client script & map controller
│   └── pdf_report.js       # Client-side PDF incident report export script
├── runs/
│   └── detect/runs/rat_yolov8/weights/best.pt  # Fine-tuned rat model weights
├── uploads/                # Uploaded sample video storage
├── requirements.txt        # Python dependency manifest
├── .env                    # Environment variables (Supabase URL & Key)
├── .gitignore              # Git ignore rules
└── README.md               # Documentation
```

---

## 📜 License

Distributed under the MIT License.
