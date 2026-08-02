from __future__ import annotations

import math
import os
import shutil
from typing import Dict, Any, Optional

import requests
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vision_engine import VisionEngine, DEFAULT_RAT_WEIGHTS, FALLBACK_COCO_WEIGHTS

app = FastAPI(title="RATTUS AI Exposure Risk API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure uploads directory exists
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize Vision Engine
vision_engine = VisionEngine()

DEFAULT_ACTIONS = {
    "Low": {
        "municipal": "Continue routine monitoring. No urgent field action required.",
        "citizen": "Normal hygiene precautions. Avoid contact with drain water.",
    },
    "Moderate": {
        "municipal": "Schedule drain inspection and targeted cleaning within 48 hours.",
        "citizen": "Avoid walking through stagnant water. Cover cuts and wash after outdoor exposure.",
    },
    "High": {
        "municipal": "Prioritize drain clearing, sanitation, and rodent control at this location today.",
        "citizen": "Avoid floodwater and drain runoff. Use boots/gloves if exposure is unavoidable.",
    },
    "Critical": {
        "municipal": "Dispatch urgent response for drain clearing, rodent control, and public warning.",
        "citizen": "Avoid the area and floodwater. Seek medical care for fever after water exposure.",
    },
}

class RiskRequest(BaseModel):
    rodent_count: int = Field(0, ge=0)
    rainfall_mm: float = Field(0.0, ge=0.0)
    drain_risk: float = Field(50.0, ge=0.0, le=100.0)
    historical_risk: float = Field(45.0, ge=0.0, le=100.0)
    manual_override: bool = False
    manual_count: int = Field(0, ge=0)


class ConfigRequest(BaseModel):
    conf_threshold: Optional[float] = None
    suppress_human_fp: Optional[bool] = None
    weights_path: Optional[str] = None
    source_type: Optional[str] = None
    camera_index: Optional[int] = None
    camera_url: Optional[str] = None
    video_file_path: Optional[str] = None


def risk_band(score: float) -> str:
    if score >= 80:
        return "Critical"
    if score >= 60:
        return "High"
    if score >= 35:
        return "Moderate"
    return "Low"


@app.get("/api/stream")
def video_stream():
    """Returns MJPEG video stream with live YOLOv8 annotations & rat counts."""
    return StreamingResponse(
        vision_engine.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/stats")
def get_stats():
    """Returns live detection statistics."""
    stats = vision_engine.stats
    return {
        "current_count": stats.current_count,
        "max_session_count": stats.max_session_count,
        "total_frames_processed": stats.total_frames_processed,
        "fps": stats.fps,
        "last_detection_time": stats.last_detection_time,
        "avg_confidence": stats.avg_confidence,
    }


@app.get("/api/config")
def get_config():
    """Returns current model & camera configuration."""
    return {
        "weights_path": vision_engine.weights_path,
        "conf_threshold": vision_engine.conf_threshold,
        "suppress_human_fp": vision_engine.suppress_human_fp,
        "source_type": vision_engine.source_type,
        "camera_index": vision_engine.camera_index,
        "camera_url": vision_engine.camera_url,
        "video_file_path": vision_engine.video_file_path,
        "available_weights": [
            DEFAULT_RAT_WEIGHTS if os.path.exists(DEFAULT_RAT_WEIGHTS) else None,
            FALLBACK_COCO_WEIGHTS if os.path.exists(FALLBACK_COCO_WEIGHTS) else None,
        ]
    }


@app.post("/api/config")
def update_config(req: ConfigRequest):
    """Updates model or camera settings dynamically."""
    vision_engine.set_config(
        conf_threshold=req.conf_threshold,
        suppress_human_fp=req.suppress_human_fp,
        weights_path=req.weights_path,
        source_type=req.source_type,
        camera_index=req.camera_index,
        camera_url=req.camera_url,
        video_file_path=req.video_file_path,
    )
    return {"status": "updated", "config": get_config()}


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Uploads a video file and sets it as the active vision source."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    clean_name = os.path.basename(file.filename)
    dest_path = os.path.join(UPLOAD_DIR, clean_name)
    
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    print(f"[Server] Video uploaded to: {dest_path}")
    
    # Switch vision engine to this video file
    vision_engine.set_config(
        source_type="video_file",
        video_file_path=dest_path
    )
    
    return {
        "status": "success",
        "filename": clean_name,
        "video_path": dest_path,
        "config": get_config()
    }


@app.post("/api/risk")
def calculate_risk_endpoint(req: RiskRequest):
    """Calculates exposure risk score and outputs guidance."""
    rodent_count = req.manual_count if req.manual_override else req.rodent_count
    
    rodent_index = min(100, rodent_count * 18)
    rainfall_index = min(100, req.rainfall_mm * 2.5)
    drain_index = req.drain_risk
    historical_index = req.historical_risk

    score = (
        rodent_index * 0.35
        + rainfall_index * 0.30
        + drain_index * 0.25
        + historical_index * 0.10
    )

    explanation = [
        f"Rodent activity index: {rodent_index:.0f}/100 from {rodent_count} detections.",
        f"Rainfall index: {rainfall_index:.0f}/100 from {req.rainfall_mm:.1f} mm recent rainfall.",
        f"Drain-flow risk: {drain_index:.0f}/100 based on field observation.",
        f"Historical area risk: {historical_index:.0f}/100 based on local prior baseline.",
    ]

    synergy_applied = False
    if rodent_count > 0 and req.rainfall_mm >= 20 and req.drain_risk >= 60:
        score = min(100, score + 10)
        synergy_applied = True
        explanation.append("Synergy boost applied: Overlapping rodent presence, heavy rainfall, and drain blockage.")

    final_score = int(round(min(100, max(0, score))))
    band = risk_band(final_score)
    actions = DEFAULT_ACTIONS.get(band, DEFAULT_ACTIONS["Low"])

    return {
        "score": final_score,
        "level": band,
        "rodent_count": rodent_count,
        "rodent_index": round(rodent_index, 1),
        "rainfall_index": round(rainfall_index, 1),
        "drain_index": round(drain_index, 1),
        "historical_index": round(historical_index, 1),
        "synergy_boost": synergy_applied,
        "explanation": explanation,
        "municipal_action": actions["municipal"],
        "citizen_action": actions["citizen"],
    }


@app.get("/api/weather")
def get_weather(lat: float = 3.1390, lon: float = 101.6869):
    """Fetches 24h rainfall data from Open-Meteo weather API."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "rain",
        "past_days": 1,
        "forecast_days": 1,
        "timezone": "auto",
    }
    try:
        resp = requests.get(url, params=params, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        rain_values = data.get("hourly", {}).get("rain", [])
        recent = [float(v or 0) for v in rain_values[-24:]]
        total_rain = round(sum(recent), 1)
        return {"success": True, "rainfall_mm": total_rain, "location": {"lat": lat, "lon": lon}}
    except Exception as e:
        return {"success": False, "rainfall_mm": 18.0, "error": str(e)}


# Mount static directory for Frontend Dashboard UI
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def read_root():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "RATTUS AI FastAPI backend is running."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
