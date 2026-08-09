from __future__ import annotations

import math
import os
import shutil
from typing import Dict, Any, Optional

import requests
from dotenv import load_dotenv
from supabase import create_client, Client
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vision_engine import VisionEngine, DEFAULT_RAT_WEIGHTS, FALLBACK_COCO_WEIGHTS

# Load environment variables from .env
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY and "your-project-ref" not in SUPABASE_URL:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[Supabase] Connected to database.")
    except Exception as err:
        print(f"[Supabase Warning] Could not initialize client: {err}")
else:
    print("[Supabase] URL/Key missing or placeholder in .env")

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
    water_level: float = Field(50.0, ge=0.0, le=100.0)
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
    water_level_index = req.water_level
    historical_index = req.historical_risk

    score = (
        rodent_index * 0.35
        + rainfall_index * 0.30
        + water_level_index * 0.25
        + historical_index * 0.10
    )

    explanation = [
        f"Rodent activity index: {rodent_index:.0f}/100 from {rodent_count} detections.",
        f"Rainfall index: {rainfall_index:.0f}/100 from {req.rainfall_mm:.1f} mm recent rainfall.",
        f"Water level index: {water_level_index:.0f}/100 (suggested/manual override).",
        f"Historical area risk: {historical_index:.0f}/100 based on local prior baseline.",
    ]

    synergy_applied = False
    if rodent_count > 0 and req.rainfall_mm >= 20 and req.water_level >= 60:
        score = min(100, score + 10)
        synergy_applied = True
        explanation.append("Synergy boost applied: Overlapping rodent presence, heavy rainfall, and high water levels.")

    final_score = int(round(min(100, max(0, score))))
    band = risk_band(final_score)
    actions = DEFAULT_ACTIONS.get(band, DEFAULT_ACTIONS["Low"])

    # Log risk assessment to Supabase if configured
    if supabase:
        try:
            supabase.table("risk_assessments").insert({
                "risk_score": final_score,
                "risk_level": band,
                "synergy_boost": synergy_applied,
                "municipal_action": actions["municipal"],
                "citizen_action": actions["citizen"],
            }).execute()
        except Exception as err:
            print(f"[Supabase Log Error] {err}")

    return {
        "score": final_score,
        "level": band,
        "rodent_count": rodent_count,
        "rodent_index": round(rodent_index, 1),
        "rainfall_index": round(rainfall_index, 1),
        "water_level_index": round(water_level_index, 1),
        "historical_index": round(historical_index, 1),
        "synergy_boost": synergy_applied,
        "explanation": explanation,
        "municipal_action": actions["municipal"],
        "citizen_action": actions["citizen"],
    }


LOCATION_PRESETS = {
    "kuala_lumpur": {"name": "Kuala Lumpur", "lat": 3.1390, "lon": 101.6869},
    "penang": {"name": "George Town, Penang", "lat": 5.4164, "lon": 100.3327},
    "johor_bahru": {"name": "Johor Bahru", "lat": 1.4927, "lon": 103.7414},
    "shah_alam": {"name": "Shah Alam, Selangor", "lat": 3.0738, "lon": 101.5183},
    "ipoh": {"name": "Ipoh, Perak", "lat": 4.5975, "lon": 101.0901},
}


@app.get("/api/weather")
def get_weather(location: str = Query("kuala_lumpur"), lat: Optional[float] = None, lon: Optional[float] = None):
    """Fetches real-time weather & 24h rainfall data from Open-Meteo API."""
    target_lat = lat
    target_lon = lon
    city_name = "Custom Location"

    if location in LOCATION_PRESETS:
        target_lat = LOCATION_PRESETS[location]["lat"]
        target_lon = LOCATION_PRESETS[location]["lon"]
        city_name = LOCATION_PRESETS[location]["name"]
    elif lat is None or lon is None:
        target_lat = LOCATION_PRESETS["kuala_lumpur"]["lat"]
        target_lon = LOCATION_PRESETS["kuala_lumpur"]["lon"]
        city_name = LOCATION_PRESETS["kuala_lumpur"]["name"]

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": target_lat,
        "longitude": target_lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation,rain",
        "hourly": "rain",
        "past_days": 1,
        "forecast_days": 1,
        "timezone": "auto",
    }
    try:
        resp = requests.get(url, params=params, timeout=6)
        resp.raise_for_status()
        data = resp.json()

        current = data.get("current", {})
        rain_values = data.get("hourly", {}).get("rain", [])
        recent = [float(v or 0) for v in rain_values[-24:]]
        total_rain = round(sum(recent), 1)

        temp_c = current.get("temperature_2m", 28.0)
        humidity = current.get("relative_humidity_2m", 80)
        current_rain = current.get("precipitation", 0.0)

        return {
            "success": True,
            "location_key": location,
            "city": city_name,
            "latitude": target_lat,
            "longitude": target_lon,
            "temperature_c": temp_c,
            "humidity_pct": humidity,
            "current_rain_mm": current_rain,
            "rainfall_mm": total_rain,
            "presets": {k: v["name"] for k, v in LOCATION_PRESETS.items()}
        }
    except Exception as e:
        print(f"[Weather API Error] {e}")
        return {
            "success": False,
            "fallback": True,
            "location_key": location,
            "city": city_name,
            "latitude": target_lat,
            "longitude": target_lon,
            "temperature_c": 28.0,
            "humidity_pct": 80,
            "current_rain_mm": 0.0,
            "rainfall_mm": 18.0,
            "presets": {k: v["name"] for k, v in LOCATION_PRESETS.items()},
            "error": str(e)
        }


@app.get("/api/nodes")
def get_drain_nodes():
    """Returns geospatial monitoring nodes across Kuala Lumpur with computed LERS risk scores."""
    nodes_raw = [
        {
            "id": "KL-DN-01",
            "name": "Chow Kit Drain Station",
            "lat": 3.1638,
            "lon": 101.6980,
            "rodent_count": 6,
            "rainfall_mm": 24.5,
            "water_level_pct": 85.0,
            "historical_risk": 75.0,
            "status": "active",
        },
        {
            "id": "KL-DN-02",
            "name": "Bukit Bintang Commercial Drain",
            "lat": 3.1466,
            "lon": 101.7115,
            "rodent_count": 3,
            "rainfall_mm": 18.0,
            "water_level_pct": 60.0,
            "historical_risk": 50.0,
            "status": "active",
        },
        {
            "id": "KL-DN-03",
            "name": "Brickfields Underground Culvert",
            "lat": 3.1305,
            "lon": 101.6862,
            "rodent_count": 4,
            "rainfall_mm": 22.0,
            "water_level_pct": 75.0,
            "historical_risk": 65.0,
            "status": "active",
        },
        {
            "id": "KL-DN-04",
            "name": "Kampung Baru Main Channel",
            "lat": 3.1610,
            "lon": 101.7065,
            "rodent_count": 1,
            "rainfall_mm": 12.0,
            "water_level_pct": 40.0,
            "historical_risk": 40.0,
            "status": "active",
        },
        {
            "id": "KL-DN-05",
            "name": "Bangsar South Stormwater Outlet",
            "lat": 3.1110,
            "lon": 101.6660,
            "rodent_count": 5,
            "rainfall_mm": 26.0,
            "water_level_pct": 80.0,
            "historical_risk": 70.0,
            "status": "active",
        },
    ]

    processed_nodes = []
    for item in nodes_raw:
        rodent_index = min(100, item["rodent_count"] * 18)
        rainfall_index = min(100, item["rainfall_mm"] * 2.5)
        water_level_index = item["water_level_pct"]
        historical_index = item["historical_risk"]

        score = (
            rodent_index * 0.35
            + rainfall_index * 0.30
            + water_level_index * 0.25
            + historical_index * 0.10
        )

        synergy = False
        if item["rodent_count"] > 0 and item["rainfall_mm"] >= 20 and item["water_level_pct"] >= 60:
            score = min(100, score + 10)
            synergy = True

        final_score = int(round(min(100, max(0, score))))
        band = risk_band(final_score)
        actions = DEFAULT_ACTIONS.get(band, DEFAULT_ACTIONS["Low"])

        processed_nodes.append({
            "id": item["id"],
            "name": item["name"],
            "latitude": item["lat"],
            "longitude": item["lon"],
            "rodent_count": item["rodent_count"],
            "rodent_index": round(rodent_index, 1),
            "rainfall_mm": item["rainfall_mm"],
            "rainfall_index": round(rainfall_index, 1),
            "water_level_pct": item["water_level_pct"],
            "historical_risk": item["historical_risk"],
            "lers_score": final_score,
            "risk_level": band,
            "synergy_boost": synergy,
            "municipal_action": actions["municipal"],
            "citizen_action": actions["citizen"],
            "status": item["status"],
        })

    return {"success": True, "count": len(processed_nodes), "nodes": processed_nodes}



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
