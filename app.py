from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
import requests
from urllib.parse import urlparse
import streamlit as st
from ultralytics import YOLO


st.set_page_config(
    page_title="RATTUS AI",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)


RODENT_LABEL_HINTS = {"rat", "mouse", "mice", "rodent", "squirrel"}
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


@dataclass
class RiskResult:
    score: int
    level: str
    explanation: list[str]


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


def risk_band(score: float) -> str:
    if score >= 80:
        return "Critical"
    if score >= 60:
        return "High"
    if score >= 35:
        return "Moderate"
    return "Low"


def calculate_risk(
    rodent_count: int,
    rainfall_mm: float,
    water_level: float,
    historical_risk: float,
) -> RiskResult:
    rodent_index = min(100, rodent_count * 18)
    rainfall_index = min(100, rainfall_mm * 2.5)
    water_level_index = water_level
    historical_index = historical_risk

    score = (
        rodent_index * 0.35
        + rainfall_index * 0.30
        + water_level_index * 0.25
        + historical_index * 0.10
    )

    explanation = [
        f"Rodent activity index: {rodent_index:.0f}/100 from {rodent_count} detections.",
        f"Rainfall index: {rainfall_index:.0f}/100 from {rainfall_mm:.1f} mm recent rainfall.",
        f"Water level index: {water_level_index:.0f}/100 (suggested/manual override).",
        f"Historical area risk: {historical_index:.0f}/100 based on local prior risk.",
    ]

    if rodent_count > 0 and rainfall_mm >= 20 and water_level >= 60:
        score = min(100, score + 10)
        explanation.append("Synergy boost applied because rodents, heavy rain, and high water levels overlap.")

    rounded = int(round(min(100, max(0, score))))
    return RiskResult(score=rounded, level=risk_band(rounded), explanation=explanation)


def get_open_meteo_rainfall(latitude: float, longitude: float) -> float | None:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "rain",
        "past_days": 1,
        "forecast_days": 1,
        "timezone": "auto",
    }
    try:
        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        rain_values = data.get("hourly", {}).get("rain", [])
        recent = [float(value or 0) for value in rain_values[-24:]]
        return round(sum(recent), 1)
    except Exception:
        return None


def parse_target_labels(raw: str) -> set[str]:
    labels = {item.strip().lower() for item in raw.split(",") if item.strip()}
    return labels or set(RODENT_LABEL_HINTS)


def get_person_boxes(frame: np.ndarray, coco_model: YOLO, min_conf: float = 0.35) -> list[list[int]]:
    try:
        results = coco_model.predict(frame, conf=min_conf, imgsz=320, verbose=False)
        result = results[0]
        if result.boxes is None:
            return []
        person_boxes = []
        names = coco_model.names
        for cls, conf, box in zip(
            result.boxes.cls.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.xyxy.tolist(),
        ):
            class_id = int(cls)
            label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
            if str(label).lower() == "person":
                person_boxes.append([int(v) for v in box])
        return person_boxes
    except Exception:
        return []


def is_human_false_positive(
    box: list[int],
    frame_shape: tuple[int, ...],
    person_boxes: list[list[int]] | None = None,
) -> bool:
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    frame_h, frame_w = frame_shape[:2]

    # Heuristic 1: Tall vertical bounding boxes covering large frame height (typical standing/sitting person)
    if (bh / frame_h > 0.35) and (bh / bw > 1.35):
        return True

    # Heuristic 2: Overlap with detected person box from COCO model
    if person_boxes:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        box_area = bw * bh

        for px1, py1, px2, py2 in person_boxes:
            # Check if rat box center is inside person box
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                return True

            # Check IoR (Intersection over Rat Box Area)
            ix1 = max(x1, px1)
            iy1 = max(y1, py1)
            ix2 = min(x2, px2)
            iy2 = min(y2, py2)
            inter_w = max(0, ix2 - ix1)
            inter_h = max(0, iy2 - iy1)
            inter_area = inter_w * inter_h
            if box_area > 0 and (inter_area / box_area) > 0.25:
                return True

    return False


def count_target_detections(
    results,
    names,
    target_labels: set[str],
    min_confidence: float,
    max_box_area_ratio: float,
    person_boxes: list[list[int]] | None = None,
    suppress_human_false_positives: bool = True,
    frame_to_draw: np.ndarray | None = None,
) -> tuple[int, np.ndarray]:
    result = results[0]
    if frame_to_draw is not None:
        annotated = frame_to_draw.copy()
    else:
        original = getattr(result, "orig_img", None)
        annotated = original.copy() if original is not None else result.plot()
    count = 0

    if result.boxes is None:
        return count, annotated

    height, width = annotated.shape[:2]
    frame_area = max(1, width * height)
    for cls, conf, box in zip(
        result.boxes.cls.tolist(),
        result.boxes.conf.tolist(),
        result.boxes.xyxy.tolist(),
    ):
        confidence_value = float(conf)
        if confidence_value < min_confidence:
            continue

        class_id = int(cls)
        label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
        label = str(label).lower()
        if label not in target_labels:
            continue

        x1, y1, x2, y2 = [int(value) for value in box]
        box_area = max(0, x2 - x1) * max(0, y2 - y1)
        if box_area / frame_area > max_box_area_ratio:
            continue

        if suppress_human_false_positives and is_human_false_positive(
            [x1, y1, x2, y2], annotated.shape, person_boxes
        ):
            continue

        count += 1
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            f"{label} {confidence_value:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return count, annotated


def camera_source_value(source_type: str, camera_index: int, camera_url: str) -> int | str:
    if source_type == "Laptop webcam":
        return camera_index
    return camera_url.strip()


def capture_one_frame(source: int | str) -> np.ndarray | None:
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        if not ok:
            return None
        return frame
    finally:
        cap.release()


def test_ip_camera_url(url: str) -> tuple[bool, str]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False, "Use a full HTTP URL such as http://192.168.1.23:8080/video."

    try:
        with requests.get(url, stream=True, timeout=6) as response:
            content_type = response.headers.get("content-type", "unknown")
            if response.status_code >= 400:
                return False, f"Phone returned HTTP {response.status_code} ({content_type})."
            if "text/html" in content_type.lower():
                return False, "This URL returns a web page. Use the app's live video endpoint, usually ending in /video."
            return True, f"Reachable: HTTP {response.status_code}, content type {content_type}."
    except requests.RequestException as exc:
        return False, f"Laptop could not reach the phone: {exc}"

def render_metric_strip(risk: RiskResult, rodent_count: int, rainfall_mm: float, water_level: float) -> None:
    cols = st.columns(4)
    cols[0].metric("Exposure Risk", f"{risk.score}/100", risk.level)
    cols[1].metric("Rodents Detected", rodent_count)
    cols[2].metric("Recent Rainfall", f"{rainfall_mm:.1f} mm")
    cols[3].metric("Water Level", f"{water_level:.0f}/100")


def render_gauge(score: int) -> None:
    pct = max(0, min(100, score))
    color = "#16a34a"
    if pct >= 80:
        color = "#dc2626"
    elif pct >= 60:
        color = "#ea580c"
    elif pct >= 35:
        color = "#ca8a04"

    st.markdown(
        f"""
        <div class="risk-gauge">
            <div class="risk-fill" style="width:{pct}%; background:{color};"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def app_styles() -> None:
    st.markdown(
        """
        <style>
        .main .block-container { padding-top: 1.4rem; }
        .risk-gauge {
            width: 100%;
            height: 18px;
            background: #e5e7eb;
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid #d1d5db;
        }
        .risk-fill { height: 100%; transition: width 220ms ease; }
        .note {
            border-left: 4px solid #2563eb;
            padding: .7rem .9rem;
            background: #eff6ff;
            color: #1e3a8a;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    app_styles()

    st.title("RATTUS AI")
    st.caption("Hyperlocal leptospirosis environmental exposure-risk demo")

    st.markdown(
        """
        <div class="note">
        Use a rodent-trained YOLOv8 model for credible rat detection. The default YOLOv8 COCO model is included only to prove the camera-to-risk dashboard pipeline.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Detection")
        weights_path = st.text_input(
            "YOLO weights path",
            value="runs/detect/runs/rat_yolov8/weights/best.pt",
            help="Use a custom rodent model such as runs/detect/train/weights/best.pt for the demo.",
        )
        confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.60, 0.05)
        max_box_area_ratio = st.slider("Maximum detection box area (%)", 5, 90, 25, 5) / 100
        suppress_human = st.toggle(
            "Suppress human false positives",
            value=True,
            help="Uses COCO person detection and shape heuristics to avoid misidentifying humans as rats.",
        )
        live_until_end = st.toggle("Run until stream ends", value=True, help="Keeps the live demo running until the camera stops or you refresh the page.")
        inference_size = st.select_slider("Inference image size", options=[320, 416, 640], value=320)
        frame_stride = st.slider("Process every Nth frame", 1, 5, 5)
        live_seconds = st.slider("Live demo duration (seconds)", 3, 600, 60, disabled=live_until_end)
        target_labels = parse_target_labels(
            st.text_input("Rodent class labels", value="rat, mouse, mice, rodent")
        )

        source_type = st.radio("Camera source", ["Laptop webcam", "Phone/IP camera URL"])
        camera_index = st.number_input("Webcam index", 0, 10, 0)
        camera_url = st.text_input(
            "Phone camera URL",
            value="",
            placeholder="Example: http://192.168.1.23:8080/video",
        )

        test_phone_url = st.button("Test Phone URL", use_container_width=True, disabled=not camera_url.strip())

        st.header("Risk Inputs")
        use_weather_api = st.toggle("Use Open-Meteo rainfall", value=False)
        latitude = st.number_input("Latitude", value=3.1390, format="%.6f")
        longitude = st.number_input("Longitude", value=101.6869, format="%.6f")
        manual_rainfall = st.slider("Manual recent rainfall (mm)", 0.0, 120.0, 18.0, 1.0)
        
        # Resolve rainfall first to suggest water level dynamically
        rainfall_mm = manual_rainfall
        if use_weather_api:
            api_rainfall = get_open_meteo_rainfall(latitude, longitude)
            if api_rainfall is None:
                st.warning("Weather API unavailable. Using manual rainfall input.")
            else:
                rainfall_mm = api_rainfall

        suggested_water_level = min(100, int(round(rainfall_mm * 5.0)))
        water_level = st.slider(
            "Water Level Measure Indicator", 
            0, 100, 
            value=suggested_water_level,
            help="Suggested based on 24h rainfall (1mm = 5%). Adjust to override."
        )
        historical_risk = st.slider("Historical area risk", 0, 100, 45)

    try:
        model = load_model(weights_path)
    except Exception as exc:
        st.error(f"Could not load YOLO weights from {weights_path}. Check the path and model file.")
        st.exception(exc)
        st.stop()

    coco_model = None
    if suppress_human:
        try:
            coco_model = load_model("yolov8n.pt")
        except Exception:
            coco_model = None

    left, right = st.columns([1.35, 1])
    frame_placeholder = left.empty()

    source = camera_source_value(source_type, int(camera_index), camera_url)

    if test_phone_url:
        with st.spinner("Testing phone camera URL..."):
            reachable, detail = test_ip_camera_url(str(source))
        if reachable:
            st.success(detail)
        else:
            st.error(detail)
    frame = None
    rodent_count = 0

    capture_col, live_col = left.columns(2)
    capture_once = capture_col.button("Capture Frame", type="primary", use_container_width=True)
    run_live = live_col.button("Run Live Demo", use_container_width=True)

    if capture_once or run_live:
        if source_type == "Phone/IP camera URL" and not str(source).strip():
            left.error("Enter a phone camera stream URL first.")
        elif capture_once:
            frame = capture_one_frame(source)
            if frame is None:
                left.error("Could not read a frame from the selected camera source.")
            else:
                person_boxes = get_person_boxes(frame, coco_model) if (suppress_human and coco_model is not None) else None
                results = model.predict(frame, conf=confidence, imgsz=inference_size, verbose=False)
                rodent_count, annotated = count_target_detections(
                    results, model.names, target_labels, confidence, max_box_area_ratio, person_boxes, suppress_human
                )
                frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
        else:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                left.error("Could not open the phone stream. Use the live video endpoint, usually ending in /video, and verify both devices share the same Wi-Fi.")
                cap.release()
                cap = None
            else:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            max_count = 0
            start = time.time()
            frame_number = 0
            last_results = None
            last_person_boxes = None

            try:
                while cap is not None and (live_until_end or time.time() - start < live_seconds):
                    ok, frame = cap.read()
                    if not ok:
                        left.error("Camera stream stopped or could not provide a frame.")
                        break

                    frame_number += 1

                    if frame_number % frame_stride == 0 or last_results is None:
                        last_person_boxes = get_person_boxes(frame, coco_model) if (suppress_human and coco_model is not None) else None
                        last_results = model.predict(frame, conf=confidence, imgsz=inference_size, verbose=False)
                        current_count, annotated = count_target_detections(
                            last_results, model.names, target_labels, confidence, max_box_area_ratio, last_person_boxes, suppress_human
                        )
                        max_count = max(max_count, current_count)
                    else:
                        _, annotated = count_target_detections(
                            last_results, model.names, target_labels, confidence, max_box_area_ratio, last_person_boxes, suppress_human, frame_to_draw=frame
                        )

                    frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
                rodent_count = max_count
            finally:
                if cap is not None:
                    cap.release()

    if frame is None:
        frame_placeholder.info("Capture a frame or run the live demo to analyze the selected camera with YOLOv8.")

    manual_override = right.toggle("Manual rodent count override", value=False)
    if manual_override:
        rodent_count = right.number_input("Rodent count", 0, 100, max(rodent_count, 2))

    risk = calculate_risk(rodent_count, rainfall_mm, float(water_level), float(historical_risk))
    render_metric_strip(risk, rodent_count, rainfall_mm, float(water_level))
    render_gauge(risk.score)

    with right:
        st.subheader("Risk Explanation")
        for item in risk.explanation:
            st.write(f"- {item}")

        actions = DEFAULT_ACTIONS[risk.level]
        st.subheader("Municipal Action")
        st.write(actions["municipal"])
        st.subheader("Citizen Alert")
        st.write(actions["citizen"])

    with st.expander("Phone camera setup for demo"):
        st.write(
            "Install an IP camera app on the phone, connect the phone and laptop to the same Wi-Fi, "
            "then paste the stream URL into the sidebar. DroidCam also works if it exposes the phone "
            "as a normal webcam index."
        )


if __name__ == "__main__":
    main()
