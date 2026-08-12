from __future__ import annotations

import math
import os
import random
import shutil
from typing import Dict, Any, Optional, Tuple

import requests
from dotenv import load_dotenv
from supabase import create_client, Client
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, Response
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
    suppress_void_fp: Optional[bool] = None
    suppress_static_fp: Optional[bool] = None
    static_seconds: Optional[float] = None


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
        "suppress_void_fp": vision_engine.suppress_void_fp,
        "suppress_static_fp": vision_engine.suppress_static_fp,
        "static_seconds": vision_engine.static_tracker.static_seconds,
        "infer_imgsz": vision_engine.infer_imgsz,
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
        suppress_void_fp=req.suppress_void_fp,
        suppress_static_fp=req.suppress_static_fp,
        static_seconds=req.static_seconds,
    )
    return {"status": "updated", "config": get_config()}


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Uploads a video file and sets it as the active vision source."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    try:
        clean_name = os.path.basename(file.filename)
        dest_path = os.path.join(UPLOAD_DIR, clean_name)
        
        contents = await file.read()
        with open(dest_path, "wb") as f:
            f.write(contents)
            
        print(f"[Server] Video uploaded successfully ({len(contents)} bytes) to: {dest_path}")
        
        # Switch vision engine to this video file
        vision_engine.set_config(
            source_type="video_file",
            video_file_path=dest_path
        )
        vision_engine.start_threads(vision_engine.get_camera_source())

        # Automatically generate & register a dynamic node on the Heatmap for this uploaded video across KL zones
        preset = random.choice(KL_DIVERSE_ZONES)
        rand_lat = round(preset["lat"] + random.uniform(-0.005, 0.005), 4)
        rand_lon = round(preset["lon"] + random.uniform(-0.005, 0.005), 4)

        # Remove previous video node so only the current active video stream node is pinned
        global DYNAMIC_NODES
        DYNAMIC_NODES = [n for n in DYNAMIC_NODES if not n.get("is_video_node")]

        new_node_id = f"KL-DN-0{len(DYNAMIC_NODES) + 1}"
        new_node = {
            "id": new_node_id,
            "name": f"{preset['name']} ({clean_name})",
            "lat": rand_lat,
            "lon": rand_lon,
            "latitude": rand_lat,
            "longitude": rand_lon,
            "rodent_count": 2,
            "rainfall_mm": 0.1,
            "water_level_pct": 85.0,
            "historical_risk": 45.0,
            "status": "active",
            "is_video_node": True,
        }
        DYNAMIC_NODES.append(new_node)
        
        return {
            "status": "success",
            "filename": clean_name,
            "video_path": dest_path,
            "node": new_node,
            "config": get_config()
        }
    except Exception as err:
        import traceback
        tb = traceback.format_exc()
        print(f"[Upload Video Error] {tb}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(err)} | {tb[:200]}")


@app.post("/api/risk")
def calculate_risk_endpoint(req: RiskRequest):
    """Calculates exposure risk score and outputs guidance."""
    rodent_count = req.manual_count if req.manual_override else req.rodent_count
    
    rodent_index = min(100, rodent_count * 35)
    water_level_index = req.water_level * 0.85
    rainfall_index = min(100, req.rainfall_mm * 2.0)
    historical_index = req.historical_risk * 0.75

    score = (
        rodent_index * 0.45
        + water_level_index * 0.30
        + rainfall_index * 0.15
        + historical_index * 0.10
    )

    explanation = [
        f"Rodent activity index: {rodent_index:.0f}/100 from {rodent_count} detections.",
        f"Rainfall index: {rainfall_index:.0f}/100 from {req.rainfall_mm:.1f} mm recent rainfall.",
        f"Water level index: {water_level_index:.0f}/100 (suggested/manual override).",
        f"Historical area risk: {historical_index:.0f}/100 based on local prior baseline.",
    ]

    synergy_applied = False
    if rodent_count > 0 and req.rainfall_mm >= 25 and req.water_level >= 70:
        score = min(100, score + 5)
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


def parse_weather_condition(code: int) -> Tuple[str, str]:
    """Returns (description, icon) for WMO weather code."""
    if code == 0:
        return "Clear sky", "☀️"
    if code in (1, 2, 3):
        return "Partly cloudy", "⛅"
    if code in (45, 48):
        return "Foggy", "🌫️"
    if code in (51, 53, 55, 61, 63, 65, 80, 81, 82):
        return "Showers / Rain", "🌧️"
    if code in (95, 96, 99):
        return "Thunderstorm", "⛈️"
    return "Scattered clouds", "🌤️"


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
        "current": "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,wind_speed_10m",
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

        temp_c = round(float(current.get("temperature_2m", 34.5)), 1)
        humidity = int(current.get("relative_humidity_2m", 62))
        wind_kmh = round(float(current.get("wind_speed_10m", 9.0)), 1)
        wcode = int(current.get("weather_code", 2))
        condition_text, condition_icon = parse_weather_condition(wcode)
        current_rain = float(current.get("precipitation", 0.0))

        return {
            "success": True,
            "location_key": location,
            "city": city_name,
            "latitude": target_lat,
            "longitude": target_lon,
            "temperature_c": temp_c,
            "humidity_pct": humidity,
            "wind_kmh": wind_kmh,
            "condition": condition_text,
            "condition_icon": condition_icon,
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
            "temperature_c": 34.5,
            "humidity_pct": 62,
            "wind_kmh": 9.0,
            "condition": "Partly cloudy",
            "condition_icon": "⛅",
            "current_rain_mm": 0.0,
            "rainfall_mm": 1.6,
            "presets": {k: v["name"] for k, v in LOCATION_PRESETS.items()},
            "error": str(e)
        }


@app.get("/api/snapshot")
def get_live_snapshot():
    """Captures and returns the current video frame with annotations as a JPEG download."""
    jpeg_bytes = vision_engine.get_snapshot_jpeg()
    if jpeg_bytes is None:
        raise HTTPException(status_code=500, detail="Could not capture snapshot frame")
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={"Content-Disposition": "attachment; filename=rattus_live_snapshot.jpg"}
    )


@app.get("/api/health")
def get_system_health():
    """Returns system status & component telemetry."""
    stats = vision_engine.stats
    return {
        "status": "healthy",
        "ai_inference": "Active",
        "model_load_pct": 100,
        "camera_status": "Live" if vision_engine.source_type != "off" else "Offline",
        "fps": stats.fps,
        "active_model": os.path.basename(vision_engine.weights_path),
        "source_type": vision_engine.source_type,
        "current_rodent_count": stats.current_count,
        "timestamp": os.getenv("APP_ENV", "production"),
    }


# Core Baseline Kuala Lumpur Monitoring Nodes (Presentation Preset: Realistic Spread)
BASELINE_NODES = [
    {
        "id": "KL-DN-01",
        "name": "Chow Kit Drain Station",
        "lat": 3.1638,
        "lon": 101.6980,
        "rodent_count": 3,
        "rainfall_mm": 24.5,
        "water_level_pct": 88.0,
        "historical_risk": 75.0,
        "status": "active",
    },
    {
        "id": "KL-DN-05",
        "name": "Bangsar South Stormwater Outlet",
        "lat": 3.1110,
        "lon": 101.6660,
        "rodent_count": 2,
        "rainfall_mm": 14.0,
        "water_level_pct": 65.0,
        "historical_risk": 50.0,
        "status": "active",
    },
    {
        "id": "KL-DN-03",
        "name": "Brickfields Underground Culvert",
        "lat": 3.1305,
        "lon": 101.6862,
        "rodent_count": 1,
        "rainfall_mm": 6.5,
        "water_level_pct": 45.0,
        "historical_risk": 40.0,
        "status": "active",
    },
    {
        "id": "KL-DN-02",
        "name": "Bukit Bintang Commercial Drain",
        "lat": 3.1466,
        "lon": 101.7115,
        "rodent_count": 1,
        "rainfall_mm": 2.0,
        "water_level_pct": 30.0,
        "historical_risk": 25.0,
        "status": "active",
    },
    {
        "id": "KL-DN-04",
        "name": "Kampung Baru Main Channel",
        "lat": 3.1610,
        "lon": 101.7065,
        "rodent_count": 0,
        "rainfall_mm": 0.5,
        "water_level_pct": 15.0,
        "historical_risk": 15.0,
        "status": "active",
    },
]

# Active Kuala Lumpur Monitoring Nodes (Dynamic State)
DYNAMIC_NODES = [dict(item) for item in BASELINE_NODES]

# Diverse Kuala Lumpur Municipal Presets for Demo Upload Pinning
KL_DIVERSE_ZONES = [
    {"name": "Setapak Central Drain", "lat": 3.1944, "lon": 101.7172},
    {"name": "Mont Kiara Drainage Outlet", "lat": 3.1667, "lon": 101.6528},
    {"name": "Ampang Jaya Storm Channel", "lat": 3.1492, "lon": 101.7618},
    {"name": "Cheras Commercial Drain", "lat": 3.1068, "lon": 101.7259},
    {"name": "Pantai Dalam Culvert", "lat": 3.0985, "lon": 101.6645},
    {"name": "TTDI Rain Runoff Station", "lat": 3.1412, "lon": 101.6288},
    {"name": "Segambut Main Channel", "lat": 3.1856, "lon": 101.6631},
    {"name": "Sri Permaisuri Outlet", "lat": 3.1002, "lon": 101.7118},
]

@app.get("/api/nodes")
def get_drain_nodes(water_level: Optional[float] = None, rainfall: Optional[float] = None):
    """Returns geospatial monitoring nodes across Kuala Lumpur with computed LERS risk scores."""
    processed_nodes = []
    
    # Ensure at least one video node is flagged if streaming a video file
    has_video_node = any(n.get("is_video_node") for n in DYNAMIC_NODES)
    if not has_video_node and vision_engine.source_type == "video_file" and DYNAMIC_NODES:
        DYNAMIC_NODES[-1]["is_video_node"] = True

    for item in DYNAMIC_NODES:
        # Active live camera stream node dynamically syncs live Open-Meteo rainfall & live water level slider
        if item.get("is_video_node"):
            w_lvl = item["water_level_pct"] if water_level is None else water_level
            item_rain = float(rainfall) if rainfall is not None else float(item.get("rainfall_mm", 0.1))
            item_rodents = 2 if vision_engine.source_type == "video_file" else max(1, vision_engine.stats.current_count)
            item["rainfall_mm"] = item_rain
            item["water_level_pct"] = w_lvl
        else:
            w_lvl = float(item["water_level_pct"])
            item_rain = float(item["rainfall_mm"])
            item_rodents = int(item["rodent_count"])

        rodent_index = min(100, item_rodents * 35)
        water_level_index = w_lvl * 0.85
        rainfall_index = min(100, item_rain * 2.0)
        historical_index = item["historical_risk"] * 0.75

        score = (
            rodent_index * 0.45
            + water_level_index * 0.30
            + rainfall_index * 0.15
            + historical_index * 0.10
        )

        synergy = False
        if item_rodents > 0 and item_rain >= 25 and w_lvl >= 70:
            score = min(100, score + 5)
            synergy = True

        final_score = int(round(min(100, max(0, score))))
        band = risk_band(final_score)
        actions = DEFAULT_ACTIONS.get(band, DEFAULT_ACTIONS["Low"])

        processed_nodes.append({
            "id": item["id"],
            "name": item["name"],
            "lat": item["lat"],
            "lon": item["lon"],
            "latitude": item.get("latitude", item["lat"]),
            "longitude": item.get("longitude", item["lon"]),
            "rodent_count": item_rodents,
            "rodent_index": round(rodent_index, 1),
            "rainfall_mm": item_rain,
            "rainfall_index": round(rainfall_index, 1),
            "water_level_pct": w_lvl,
            "historical_risk": item["historical_risk"],
            "lers_score": final_score,
            "risk_level": band,
            "synergy_boost": synergy,
            "municipal_action": actions["municipal"],
            "citizen_action": actions["citizen"],
            "status": item["status"],
            "is_video_node": item.get("is_video_node", False),
        })

    # Sort descending by LERS risk score
    processed_nodes.sort(key=lambda x: x["lers_score"], reverse=True)
    return {"success": True, "count": len(processed_nodes), "nodes": processed_nodes}

@app.api_route("/api/nodes/reset", methods=["GET", "POST"])
def reset_drain_nodes(water_level: Optional[float] = None):
    """Resets Kuala Lumpur monitoring nodes back to the 5 core baseline stations."""
    global DYNAMIC_NODES
    DYNAMIC_NODES = [dict(item) for item in BASELINE_NODES]
    res = get_drain_nodes(water_level=water_level)
    res["message"] = "Map reset to 5 core baseline monitoring stations."
    return res

@app.api_route("/api/reset_all", methods=["GET", "POST"])
def reset_all_system_state():
    """Resets complete system state: stops video streaming, resets nodes to baseline 5, and clears vision stats."""
    global DYNAMIC_NODES
    DYNAMIC_NODES = [dict(item) for item in BASELINE_NODES]
    
    # Stop video stream and clear active video files
    vision_engine.set_config(
        source_type="off",
        video_file_path=""
    )
    vision_engine.start_threads("off")
    
    with vision_engine.lock:
        vision_engine.latest_detections = []
        vision_engine.latest_rodent_count = 0
        vision_engine.latest_conf_scores = []
        vision_engine.stats.current_count = 0
        vision_engine.stats.max_session_count = 0
        vision_engine.stats.total_frames_processed = 0
        vision_engine.stats.fps = 0.0
        vision_engine.stats.last_detection_time = None
        vision_engine.stats.avg_confidence = 0.0

    return {"status": "success", "message": "System state fully reset to baseline"}




# --- COMMUNITY MOBILE APP ENDPOINTS ---

class CitizenReportRequest(BaseModel):
    location_name: str = "Chow Kit Market Drain"
    latitude: float = 3.1638
    longitude: float = 101.6980
    rodent_count: int = 3
    video_filename: Optional[str] = "sighting_chowkit.mp4"
    description: str = "Multiple rodents spotted near drain overflow."
    reporter_name: str = "Citizen Resident"

class ForumPostRequest(BaseModel):
    author: str = "Ahmad Rizal"
    location: str = "Chow Kit"
    tag: str = "Rodent Sighting"  # 'Rodent Sighting', 'Drain Blockage', 'Authority Update', 'Health Tip'
    content: str = ""
    media_url: Optional[str] = None

class ForumCommentRequest(BaseModel):
    post_id: str
    author: str = "Resident"
    comment: str = ""

class UpvoteRequest(BaseModel):
    post_id: str


# In-memory datasets
CITIZEN_REPORTS = [
    {
        "id": "CR-101",
        "timestamp": "Today, 02:15 PM",
        "location_name": "Chow Kit Market Main Drain",
        "latitude": 3.1638,
        "longitude": 101.6980,
        "rodent_count": 5,
        "risk_level": "CRITICAL",
        "status": "Submitted to DBKL",
        "reporter_name": "Citizen Resident",
        "description": "Recorded 5 rats running through open drain near food stalls after rain.",
    },
    {
        "id": "CR-102",
        "timestamp": "Today, 11:30 AM",
        "location_name": "Bangsar South Stormwater Outlet",
        "latitude": 3.1110,
        "longitude": 101.6660,
        "rodent_count": 4,
        "risk_level": "CRITICAL",
        "status": "Work Order Assigned",
        "reporter_name": "Siti Nurhaliza",
        "description": "Culvert grate blocked by trash, rat activity high near sidewalk.",
    },
    {
        "id": "CR-103",
        "timestamp": "Yesterday, 04:45 PM",
        "location_name": "Brickfields Commercial Alley",
        "latitude": 3.1305,
        "longitude": 101.6862,
        "rodent_count": 2,
        "risk_level": "HIGH",
        "status": "Under Review",
        "reporter_name": "Kavitha M.",
        "description": "Stagnant water accumulation behind restaurant alleyway.",
    }
]

FORUM_POSTS = [
    {
        "id": "POST-001",
        "author": "Ahmad Rizal",
        "timestamp": "10 mins ago",
        "location": "Chow Kit",
        "tag": "Rodent Sighting",
        "content": "Just uploaded video footage of 5 rats running in the Chow Kit main drain! Water level is very high after today's downpour. Please avoid walking near the drain runoff!",
        "upvotes": 24,
        "comments": [
            {"author": "Farah L.", "timestamp": "5 mins ago", "text": "Reported to DBKL as well! Glad the AI video scanner tagged it automatically."},
            {"author": "DBKL Officer", "timestamp": "2 mins ago", "text": "Work order dispatched for drain clearing team today."}
        ]
    },
    {
        "id": "POST-002",
        "author": "DBKL Municipal Sanitation Team",
        "timestamp": "1 hour ago",
        "location": "Bangsar South",
        "tag": "Authority Update",
        "content": "📢 OFFICIAL UPDATE: Sanitation team dispatched to Bangsar South Stormwater Outlet. Drain clearing and rodent baiting in progress.",
        "upvotes": 42,
        "comments": [
            {"author": "Jason K.", "timestamp": "45 mins ago", "text": "Thank you for the quick response!"}
        ]
    },
    {
        "id": "POST-003",
        "author": "Dr. Mei Ling (Public Health)",
        "timestamp": "3 hours ago",
        "location": "Kuala Lumpur Zone",
        "tag": "Health Tip",
        "content": "💡 Leptospirosis Prevention Reminder: Heavy rainfall washes rat urine into stagnant floodwaters. Wear rubber boots if walking near flooded streets and cover all open cuts!",
        "upvotes": 58,
        "comments": []
    }
]


@app.get("/api/reports")
def get_citizen_reports():
    """Returns submitted citizen rodent sighting reports."""
    return {"success": True, "count": len(CITIZEN_REPORTS), "reports": CITIZEN_REPORTS}


@app.post("/api/reports")
def submit_citizen_report(req: CitizenReportRequest):
    """Submits a new citizen video report to municipal authorities."""
    risk_level = "HIGH"
    if req.rodent_count >= 5:
        risk_level = "CRITICAL"
    elif req.rodent_count <= 2:
        risk_level = "MODERATE"

    # Select a diverse KL municipal zone preset for spatial dispersion
    preset = random.choice(KL_DIVERSE_ZONES)
    rand_lat = round(preset["lat"] + random.uniform(-0.004, 0.004), 4)
    rand_lon = round(preset["lon"] + random.uniform(-0.004, 0.004), 4)

    new_report = {
        "id": f"CR-{100 + len(CITIZEN_REPORTS) + 1}",
        "timestamp": "Just now",
        "location_name": req.location_name or preset["name"],
        "latitude": rand_lat,
        "longitude": rand_lon,
        "rodent_count": req.rodent_count,
        "risk_level": risk_level,
        "status": "Submitted to DBKL",
        "reporter_name": req.reporter_name,
        "description": req.description,
    }
    CITIZEN_REPORTS.insert(0, new_report)

    # Automatically create & register a dynamic node on the Heatmap
    new_node_id = f"KL-DN-0{len(DYNAMIC_NODES) + 1}"
    new_node = {
        "id": new_node_id,
        "name": f"{req.location_name or preset['name']}",
        "lat": rand_lat,
        "lon": rand_lon,
        "rodent_count": req.rodent_count,
        "rainfall_mm": 28.0,
        "water_level_pct": 85.0,
        "historical_risk": 70.0,
        "status": "active",
    }
    DYNAMIC_NODES.append(new_node)

    # Automatically log report into Supabase if configured
    if supabase:
        try:
            supabase.table("citizen_reports").insert(new_report).execute()
        except Exception as err:
            print(f"[Supabase Report Error] {err}")

    return {"success": True, "message": "Report submitted & dynamic pin generated on Kuala Lumpur Heatmap!", "report": new_report, "node": new_node}


@app.get("/api/forum")
def get_forum_posts():
    """Returns community forum posts."""
    return {"success": True, "count": len(FORUM_POSTS), "posts": FORUM_POSTS}


@app.post("/api/forum")
def create_forum_post(req: ForumPostRequest):
    """Creates a new community discussion post."""
    new_post = {
        "id": f"POST-00{len(FORUM_POSTS) + 1}",
        "author": req.author or "Anonymous Resident",
        "timestamp": "Just now",
        "location": req.location or "Kuala Lumpur",
        "tag": req.tag or "General",
        "content": req.content,
        "upvotes": 1,
        "comments": []
    }
    FORUM_POSTS.insert(0, new_post)
    return {"success": True, "message": "Post created successfully!", "post": new_post}


@app.post("/api/forum/upvote")
def upvote_forum_post(req: UpvoteRequest):
    """Increments upvote count for a forum post."""
    for post in FORUM_POSTS:
        if post["id"] == req.post_id:
            post["upvotes"] += 1
            return {"success": True, "upvotes": post["upvotes"]}
    raise HTTPException(status_code=404, detail="Post not found")


@app.post("/api/forum/comment")
def add_forum_comment(req: ForumCommentRequest):
    """Adds a comment to a forum post."""
    for post in FORUM_POSTS:
        if post["id"] == req.post_id:
            comment_obj = {
                "author": req.author,
                "timestamp": "Just now",
                "text": req.comment
            }
            post["comments"].append(comment_obj)
            return {"success": True, "comments": post["comments"]}
    raise HTTPException(status_code=404, detail="Post not found")


class LoginRequest(BaseModel):
    email: str = "admin@example.com"
    password: str = "admin123"


@app.post("/api/login")
def login_user(req: LoginRequest):
    """Handles role-based user login for Authorities and Community Members."""
    email_clean = (req.email or "").strip().lower()

    if "admin" in email_clean or "authority" in email_clean or "dbkl" in email_clean or "gov" in email_clean:
        user_info = {
            "email": req.email,
            "name": "DBKL Municipal Officer",
            "role": "authority",
            "department": "Public Health & Sanitation",
        }
        return {"success": True, "user": user_info, "redirect": "/", "message": "Welcome to Authority Command Center"}
    else:
        user_info = {
            "email": req.email,
            "name": "Citizen Resident",
            "role": "community",
            "zone": "Kuala Lumpur Central",
        }
        return {"success": True, "user": user_info, "redirect": "/community", "message": "Welcome to Citizen Community App"}


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
    return {"message": "RATTUS FastAPI backend is running."}


@app.get("/community")
def read_community():
    community_file = os.path.join(static_dir, "community.html")
    if os.path.exists(community_file):
        return FileResponse(community_file)
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/login")
def read_login():
    login_file = os.path.join(static_dir, "login.html")
    if os.path.exists(login_file):
        return FileResponse(login_file)
    return FileResponse(os.path.join(static_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)


