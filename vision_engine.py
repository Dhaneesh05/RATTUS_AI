from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass
from typing import Generator, Optional, Tuple, List, Set, Dict, Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Set PyTorch thread count for optimal CPU performance
try:
    torch.set_num_threads(min(4, os.cpu_count() or 4))
except Exception:
    pass

RODENT_LABEL_HINTS = {"rat", "mouse", "mice", "rodent", "squirrel"}

DEFAULT_RAT_WEIGHTS = os.path.join("runs", "detect", "runs", "rat_yolov8", "weights", "best.pt")
FALLBACK_COCO_WEIGHTS = "yolov8n.pt"

@dataclass
class DetectionStats:
    current_count: int = 0
    max_session_count: int = 0
    total_frames_processed: int = 0
    fps: float = 0.0
    last_detection_time: Optional[float] = None
    avg_confidence: float = 0.0


class VisionEngine:
    def __init__(self, default_weights: Optional[str] = None):
        self.lock = threading.Lock()
        
        # Select best available model weights
        if default_weights and os.path.exists(default_weights):
            self.weights_path = default_weights
        elif os.path.exists(DEFAULT_RAT_WEIGHTS):
            self.weights_path = DEFAULT_RAT_WEIGHTS
        else:
            self.weights_path = FALLBACK_COCO_WEIGHTS

        self.model: Optional[YOLO] = None
        self.coco_person_model: Optional[YOLO] = None
        
        # Configuration
        self.conf_threshold: float = 0.50
        self.suppress_human_fp: bool = True
        self.max_box_area_ratio: float = 0.25
        self.source_type: str = "webcam"  # 'webcam', 'ip_cam', 'video_file', 'off'
        self.camera_index: int = 0
        self.camera_url: str = ""
        self.video_file_path: str = ""
        self.target_labels: Set[str] = set(RODENT_LABEL_HINTS)
        
        # Caching & Frame Skipping
        self.cached_person_boxes: List[List[int]] = []
        self.person_check_counter: int = 0
        self.person_check_interval: int = 5
        
        # Camera Capture Thread
        self.cap_thread: Optional[threading.Thread] = None
        self.cap_running: bool = False
        self.current_source: Any = None
        
        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        
        # Async Inference Thread
        self.inference_thread: Optional[threading.Thread] = None
        self.inference_running: bool = False
        self.latest_detections: List[Dict[str, Any]] = []
        self.latest_rodent_count: int = 0
        self.latest_conf_scores: List[float] = []

        # Live Stats
        self.stats = DetectionStats()
        self._inference_times: List[float] = []
        
        # Load models
        self.load_model(self.weights_path)
        self.load_coco_person_model()

    def load_coco_person_model(self):
        try:
            if os.path.exists(FALLBACK_COCO_WEIGHTS):
                self.coco_person_model = YOLO(FALLBACK_COCO_WEIGHTS)
        except Exception as e:
            print(f"[VisionEngine] Could not load COCO person detector: {e}")

    def load_model(self, weights_path: str) -> bool:
        with self.lock:
            try:
                print(f"[VisionEngine] Loading YOLO model from {weights_path}...")
                self.model = YOLO(weights_path)
                self.weights_path = weights_path
                return True
            except Exception as e:
                print(f"[VisionEngine] Error loading model {weights_path}: {e}")
                if weights_path != FALLBACK_COCO_WEIGHTS and os.path.exists(FALLBACK_COCO_WEIGHTS):
                    try:
                        self.model = YOLO(FALLBACK_COCO_WEIGHTS)
                        self.weights_path = FALLBACK_COCO_WEIGHTS
                        return True
                    except Exception:
                        pass
                return False

    def set_config(
        self,
        conf_threshold: Optional[float] = None,
        suppress_human_fp: Optional[bool] = None,
        weights_path: Optional[str] = None,
        source_type: Optional[str] = None,
        camera_index: Optional[int] = None,
        camera_url: Optional[str] = None,
        video_file_path: Optional[str] = None,
        max_box_area_ratio: Optional[float] = None,
    ):
        with self.lock:
            if conf_threshold is not None:
                self.conf_threshold = max(0.10, min(0.95, float(conf_threshold)))
            if suppress_human_fp is not None:
                self.suppress_human_fp = bool(suppress_human_fp)
            if max_box_area_ratio is not None:
                self.max_box_area_ratio = float(max_box_area_ratio)
            if source_type is not None:
                self.source_type = source_type
            if camera_index is not None:
                self.camera_index = int(camera_index)
            if camera_url is not None:
                self.camera_url = camera_url.strip()
            if video_file_path is not None:
                self.video_file_path = video_file_path.strip()
                
        if weights_path and weights_path != self.weights_path:
            self.load_model(weights_path)

    def get_camera_source(self) -> Any:
        if self.source_type == "off":
            return "off"
        if self.source_type == "webcam":
            return self.camera_index
        if self.source_type == "video_file":
            return self.video_file_path if self.video_file_path else self.camera_index
        return self.camera_url if self.camera_url else self.camera_index

    def _camera_worker(self, source: Any):
        if source == "off":
            print("[VisionEngine] Camera worker paused (Source is OFF)")
            with self.frame_lock:
                self.latest_frame = None
            return

        print(f"[VisionEngine] Camera worker started for source: {source}")
        cap = cv2.VideoCapture(source)
        
        is_file = isinstance(source, str) and os.path.exists(source)
        fps_target = 30.0
        
        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif is_file:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps > 0:
                fps_target = video_fps

        frame_delay = 1.0 / fps_target

        while self.cap_running and source == self.current_source:
            t_start = time.time()
            ret, frame = cap.read()
            
            # Loop video file if reached end
            if not ret and is_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

            if ret and frame is not None:
                # Resize video frames if too large for uniform processing
                if frame.shape[1] > 960:
                    scale = 960 / frame.shape[1]
                    frame = cv2.resize(frame, (960, int(frame.shape[0] * scale)))
                with self.frame_lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.01)

            # Throttle file playback to match video FPS
            if is_file:
                elapsed = time.time() - t_start
                sleep_time = frame_delay - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        cap.release()
        print(f"[VisionEngine] Camera worker stopped for source: {source}")

    def detect_person_boxes_cached(self, frame: np.ndarray) -> List[List[int]]:
        self.person_check_counter += 1
        if self.person_check_counter % self.person_check_interval == 1 or not self.cached_person_boxes:
            if self.coco_person_model is not None:
                try:
                    res = self.coco_person_model.predict(frame, conf=0.35, imgsz=256, verbose=False)
                    if res and res[0].boxes is not None:
                        boxes = []
                        names = res[0].names
                        for cls, box in zip(res[0].boxes.cls.tolist(), res[0].boxes.xyxy.tolist()):
                            class_id = int(cls)
                            label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
                            if str(label).lower() == "person":
                                boxes.append([int(v) for v in box])
                        self.cached_person_boxes = boxes
                except Exception:
                    pass
        return self.cached_person_boxes

    def is_human_false_positive(
        self,
        box: List[int],
        frame_height: int,
        frame_width: int,
        person_boxes: List[List[int]]
    ) -> bool:
        x1, y1, x2, y2 = box
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        box_area = bw * bh
        frame_area = frame_height * frame_width

        if bh / frame_height > 0.40 and bh / bw > 0.7:
            return True

        if box_area / frame_area > self.max_box_area_ratio:
            return True

        for px1, py1, px2, py2 in person_boxes:
            pbw = max(1, px2 - px1)
            pbh = max(1, py2 - py1)
            p_area = pbw * pbh
            ix1 = max(x1, px1)
            iy1 = max(y1, py1)
            ix2 = min(x2, px2)
            iy2 = min(y2, py2)
            inter_w = max(0, ix2 - ix1)
            inter_h = max(0, iy2 - iy1)
            inter_area = inter_w * inter_h

            if inter_area > 0:
                if (inter_area / box_area) > 0.60 and (box_area / frame_area) > 0.15:
                    return True
                union_area = box_area + p_area - inter_area
                if (inter_area / union_area) > 0.40:
                    return True

        return False

    def _inference_worker(self):
        print("[VisionEngine] Async Inference worker started")
        while self.inference_running:
            if self.source_type == "off":
                with self.lock:
                    self.latest_detections = []
                    self.latest_rodent_count = 0
                    self.stats.current_count = 0
                time.sleep(0.1)
                continue

            with self.frame_lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None
            
            if frame is None or self.model is None:
                time.sleep(0.05)
                continue

            t_start = time.time()
            height, width = frame.shape[:2]
            
            try:
                with torch.no_grad():
                    results = self.model.predict(
                        frame,
                        conf=self.conf_threshold,
                        imgsz=320,
                        verbose=False
                    )
            except Exception as e:
                print(f"[VisionEngine] Inference error: {e}")
                time.sleep(0.1)
                continue

            person_boxes = self.detect_person_boxes_cached(frame) if self.suppress_human_fp else []

            new_detections = []
            rodent_count = 0
            conf_scores = []

            if results and len(results) > 0 and results[0].boxes is not None:
                res = results[0]
                names = res.names
                is_custom_rodent_model = "rat" in self.weights_path.lower() or "rodent" in self.weights_path.lower()

                for cls, conf, box in zip(
                    res.boxes.cls.tolist(),
                    res.boxes.conf.tolist(),
                    res.boxes.xyxy.tolist(),
                ):
                    confidence = float(conf)
                    if confidence < self.conf_threshold:
                        continue

                    class_id = int(cls)
                    raw_label = str(names.get(class_id, class_id)) if isinstance(names, dict) else str(names[class_id])
                    label_lower = raw_label.lower()

                    is_rodent = (
                        is_custom_rodent_model
                        or label_lower in self.target_labels
                        or any(hint in label_lower for hint in RODENT_LABEL_HINTS)
                    )

                    if not is_rodent:
                        continue

                    x1, y1, x2, y2 = [int(v) for v in box]

                    if self.suppress_human_fp and self.is_human_false_positive([x1, y1, x2, y2], height, width, person_boxes):
                        continue

                    rodent_count += 1
                    conf_scores.append(confidence)
                    new_detections.append({
                        "label": raw_label,
                        "confidence": round(confidence, 2),
                        "box": [x1, y1, x2, y2]
                    })

            t_end = time.time()
            self._inference_times.append(t_end - t_start)
            if len(self._inference_times) > 10:
                self._inference_times.pop(0)

            fps = round(1.0 / (sum(self._inference_times) / len(self._inference_times)), 1) if self._inference_times else 0.0

            with self.lock:
                self.latest_detections = new_detections
                self.latest_rodent_count = rodent_count
                self.latest_conf_scores = conf_scores
                
                self.stats.current_count = rodent_count
                self.stats.max_session_count = max(self.stats.max_session_count, rodent_count)
                self.stats.total_frames_processed += 1
                self.stats.fps = fps
                if rodent_count > 0:
                    self.stats.last_detection_time = time.time()
                    self.stats.avg_confidence = round(float(np.mean(conf_scores)), 2) if conf_scores else 0.0

            time.sleep(0.01)

        print("[VisionEngine] Async Inference worker stopped")

    def start_threads(self, source: Any):
        if source == "off":
            self.cap_running = False
            self.current_source = "off"
            return

        if self.current_source != source or not self.cap_running or (self.cap_thread and not self.cap_thread.is_alive()):
            self.cap_running = False
            if self.cap_thread and self.cap_thread.is_alive():
                self.cap_thread.join(timeout=1.0)

            self.current_source = source
            self.cap_running = True
            self.cap_thread = threading.Thread(target=self._camera_worker, args=(source,), daemon=True)
            self.cap_thread.start()

        if not self.inference_running or (self.inference_thread and not self.inference_thread.is_alive()):
            self.inference_running = False
            if self.inference_thread and self.inference_thread.is_alive():
                self.inference_thread.join(timeout=1.0)
            
            self.inference_running = True
            self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
            self.inference_thread.start()

    def generate_frames(self) -> Generator[bytes, None, None]:
        while True:
            source = self.get_camera_source()
            
            if source == "off":
                fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    fallback_frame,
                    "Camera Off — Input Stream Paused",
                    (110, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (120, 120, 120),
                    2
                )
                _, buffer = cv2.imencode('.jpg', fallback_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.2)
                continue

            self.start_threads(source)

            with self.frame_lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None

            if frame is None:
                fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    fallback_frame,
                    f"Connecting to source: {source}...",
                    (60, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (100, 180, 255),
                    2
                )
                _, buffer = cv2.imencode('.jpg', fallback_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.2)
                continue

            annotated_frame = frame

            with self.lock:
                current_detections = list(self.latest_detections)
                current_count = self.latest_rodent_count

            for det in current_detections:
                x1, y1, x2, y2 = det["box"]
                conf = det["confidence"]
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 220), 3)
                cv2.rectangle(annotated_frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 255, 255), 1)

                text = f"RODENT {conf * 100:.0f}%"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                bg_y1 = max(0, y1 - th - 10)
                bg_y2 = y1
                cv2.rectangle(annotated_frame, (x1, bg_y1), (x1 + tw + 10, bg_y2), (0, 0, 220), -1)
                cv2.putText(
                    annotated_frame,
                    text,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            hud_bg_color = (0, 0, 180) if current_count > 0 else (40, 140, 40)
            hud_text = f"LIVE RODENTS DETECTED: {current_count}"
            cv2.rectangle(annotated_frame, (15, 15), (340, 55), (0, 0, 0), -1)
            cv2.rectangle(annotated_frame, (15, 15), (340, 55), hud_bg_color, 2)
            cv2.putText(
                annotated_frame,
                hud_text,
                (25, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)
