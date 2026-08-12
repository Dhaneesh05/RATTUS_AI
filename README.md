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
| `/api/config` | `GET / POST` | Reads or updates AI confidence threshold, false-positive suppression (`suppress_human_fp`, `suppress_void_fp`, `suppress_static_fp`, `static_seconds`), input source, and model weights |
| `/api/upload_video` | `POST` | Uploads a video file (`.mp4`) and switches vision source to the uploaded video |
| `/api/weather` | `GET` | Fetches 24h rainfall & ambient weather data from Open-Meteo for selected city presets or lat/lon |
| `/api/nodes` | `GET` | Returns geospatial drain monitoring nodes with computed LERS risk scores for interactive mapping |

---

## 🎯 False-Positive Control & Retraining

The detector fires on empty drain openings: a dark circular void reads as a rodent,
often with high confidence, so raising `conf_threshold` cannot fix it without also
deleting real rats in dark pipes. Two independent controls handle it at runtime, and
a retraining pipeline fixes it at the source.

### Runtime gates (`vision_engine.py`)

| Gate | Config key | Default | What it does |
| :--- | :--- | :--- | :--- |
| Appearance | `suppress_void_fp` | **on** | Drops boxes whose centre is dark, flat, textureless and ringed by a brighter rim — the signature of a pipe mouth. Calibrated on `rat-dataset`: rejects the drain hole, drops 0 of 400 labelled rats. |
| Temporal | `suppress_static_fp` | **off** | Drops boxes pinned to the same pixels for `static_seconds`. Scenery never moves — but neither does a rodent watching from a burrow, so this is opt-in. Enable only for a locked-off camera where false positives cost more than a missed animal. |

Inference resolution is read from the weights themselves, so it always matches the
size the model was trained at (416 here). Running 640 against 416-trained weights was
a measurable source of confident nonsense.

```bash
curl -X POST localhost:8000/api/config -H "Content-Type: application/json" \
     -d '{"suppress_static_fp": true, "static_seconds": 2.5}'
```

### Retraining with hard negatives (`tools/`)

YOLO learns "not a rat" from images with **empty label files**. This dataset shipped
with zero of them, which is why the model had never been told what an empty pipe is.

```bash
python tools/mine_hard_negatives.py       # 1. propose candidates from uploads/*.mp4
                                          # 2. REVIEW hard_negatives/contact_sheet_*.jpg
                                          #    delete any frame containing an animal
python tools/apply_hard_negatives.py      # 3. file survivors as background images
python tools/retrain.py --epochs 40       # 4. retrain + before/after comparison
```

**The review step is not optional.** Rodents freeze for seconds at a time, so no
automatic check reliably separates "empty hole" from "motionless mouse in hole" — on
this project's own footage, the first mining pass proposed 24 frames of which 11
contained a visible rodent. A mislabelled negative teaches the model to ignore rats.

For footage you already know is rat-free, skip the guessing entirely:

```bash
python tools/mine_hard_negatives.py --videos \
       --rat-free "uploads/empty_drain.mp4:0-30"
```

`apply_hard_negatives.py` records every file it adds and holds back 20% in `valid/`
so the fix can be measured honestly; `--undo` removes exactly those files again.
Keep backgrounds under ~10% of the dataset or the model turns timid.

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
├── tools/
│   ├── mine_hard_negatives.py   # Proposes empty-pipe frames from footage for review
│   ├── apply_hard_negatives.py  # Files reviewed frames as background images (--undo)
│   └── retrain.py               # Retrains and reports the before/after FP change
├── runs/
│   └── detect/runs/rat_yolov8/weights/best.pt  # Fine-tuned rat model weights
├── hard_negatives/         # Mining output & contact sheets awaiting review
├── uploads/                # Uploaded sample video storage
├── requirements.txt        # Python dependency manifest
├── .env                    # Environment variables (Supabase URL & Key)
├── .gitignore              # Git ignore rules
└── README.md               # Documentation
```

---

## 📜 License

Distributed under the MIT License.
