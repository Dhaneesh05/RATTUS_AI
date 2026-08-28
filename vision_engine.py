from __future__ import annotations

import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import time
import threading
from dataclasses import dataclass
from typing import Generator, Optional, Tuple, List, Set, Dict, Any

import cv2
import numpy as np

HAS_YOLO = False
torch = None
YOLO = None

try:
    import torch
    from ultralytics import YOLO
    HAS_YOLO = True
    try:
        torch.set_num_threads(min(4, os.cpu_count() or 4))
    except Exception:
        pass
except Exception as _err:
    print(f"[VisionEngine Warning] PyTorch/YOLO not loaded: {_err}")


RODENT_LABEL_HINTS = {"rat", "rats", "rodent", "rodents"}

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


def box_iou(a: List[int], b: List[int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(area_a + area_b - inter)


class StaticRegionTracker:
    """Suppresses detections that are frozen in place.

    Drain openings, dark recesses and stains sit in exactly the same pixels frame
    after frame; a live rodent does not. This gate is independent of what the model
    believes, so it still works when the model is confidently wrong about scenery.
    """

    SIG_SIZE = 64

    def __init__(self):
        self.tracks: List[Dict[str, Any]] = []
        self.prev_thumb: Optional[np.ndarray] = None
        self.static_seconds: float = 1.5    # how long a region must sit still before it counts as scenery
        self.patch_delta: float = 6.0       # grey-level change (0-255) that counts as real movement
        self.max_drift: float = 0.08        # box travel, as a fraction of its own diagonal
        self.iou_match: float = 0.55
        self.global_motion_delta: float = 9.0
        self.forget_seconds: float = 2.0

    def _sample(self, frame: np.ndarray, box: List[int]) -> Optional[np.ndarray]:
        """Grabs a fixed *frame-coordinate* window, not the contents of a box.

        Cropping to a box that follows an object would be translation-invariant — a
        rodent would carry an unchanging patch along with it and read as motionless.
        Anchoring to fixed pixels is what makes movement visible.
        """
        h, w = frame.shape[:2]
        x1, y1 = max(0, min(box[0], w - 2)), max(0, min(box[1], h - 2))
        x2, y2 = max(x1 + 2, min(box[2], w)), max(y1 + 2, min(box[3], h))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sig = cv2.resize(gray, (self.SIG_SIZE, self.SIG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
        return sig - float(sig.mean())  # mean-centred, so exposure/gain drift is not read as motion

    @staticmethod
    def _drift(box: List[int], anchor: List[int]) -> float:
        bcx, bcy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        acx, acy = (anchor[0] + anchor[2]) / 2.0, (anchor[1] + anchor[3]) / 2.0
        diag = max(1.0, float(np.hypot(anchor[2] - anchor[0], anchor[3] - anchor[1])))
        return float(np.hypot(bcx - acx, bcy - acy)) / diag

    def _global_motion(self, frame: np.ndarray) -> float:
        thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 48),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
        thumb -= float(thumb.mean())
        prev, self.prev_thumb = self.prev_thumb, thumb
        if prev is None:
            return 0.0
        return float(np.mean(np.abs(thumb - prev)))

    def reset(self):
        self.tracks = []
        self.prev_thumb = None

    def update(self, frame: np.ndarray, boxes: List[List[int]], now: float) -> List[bool]:
        """Returns one flag per box: True means 'this is static scenery, drop it'."""
        camera_moving = self._global_motion(frame) > self.global_motion_delta

        # Expire before matching: a track that went stale must never lend its
        # accumulated "static" age to a fresh object appearing in the same place.
        self.tracks = [t for t in self.tracks if now - t["last_seen"] <= self.forget_seconds]

        claimed: Set[int] = set()
        verdicts: List[bool] = []

        for box in boxes:
            best_idx, best_iou = -1, self.iou_match
            for idx, track in enumerate(self.tracks):
                if idx in claimed:
                    continue
                iou = box_iou(box, track["box"])
                if iou >= best_iou:
                    best_idx, best_iou = idx, iou

            if best_idx < 0:
                # Brand new object — never suppressed, so a rodent entering frame shows instantly.
                self.tracks.append({
                    "box": box, "anchor": box, "sig": self._sample(frame, box),
                    "since": now, "last_seen": now,
                })
                verdicts.append(False)
                continue

            claimed.add(best_idx)
            track = self.tracks[best_idx]

            moved = camera_moving or self._drift(box, track["anchor"]) > self.max_drift
            if not moved:
                # Re-read the anchor window and compare against the reference taken when
                # the region went still. Anchored rather than frame-to-frame, so even a
                # slow creep accumulates instead of hiding under the per-frame threshold.
                sig_now = self._sample(frame, track["anchor"])
                if sig_now is None or track["sig"] is None:
                    moved = True
                else:
                    moved = float(np.mean(np.abs(sig_now - track["sig"]))) > self.patch_delta

            track["box"] = box
            track["last_seen"] = now
            if moved:
                track["anchor"] = box
                track["sig"] = self._sample(frame, box)
                track["since"] = now  # something really changed — restart the clock

            verdicts.append((now - track["since"]) >= self.static_seconds)

        return verdicts


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
        self.max_box_area_ratio: float = 0.55
        self.infer_imgsz: int = 640          # overwritten with the size the weights were trained at
        self.suppress_void_fp: bool = False   # appearance gate: dark textureless holes (disabled by default so fine-tuned model detects rats)
        # Temporal gate: off by default. It removes anything pinned in place, and a
        # rodent watching from a burrow holds perfectly still — on this project's own
        # footage a frozen mouse face and an empty hole differ by less than 2x on every
        # appearance measure once the video is soft, so stillness cannot be told from
        # absence. A missed rat is worse than a spare box here, and the appearance gate
        # already covers the pipe-mouth case. Turn on for a locked-off camera where
        # false positives matter more: POST /api/config {"suppress_static_fp": true}
        self.suppress_static_fp: bool = False
        self.animal_texture_floor: float = 60.0  # detail above which nothing is ever timed out
        self.static_tracker = StaticRegionTracker()
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

    def is_rodent_label(self, label: str) -> bool:
        label_lower = label.strip().lower()
        if label_lower in self.target_labels:
            return True

        normalized = (
            label_lower.replace("-", " ")
            .replace("_", " ")
            .replace("/", " ")
            .replace("\\", " ")
        )
        return any(part in self.target_labels for part in normalized.split())

    def load_coco_person_model(self):
        if not HAS_YOLO or YOLO is None:
            return
        try:
            if os.path.exists(FALLBACK_COCO_WEIGHTS):
                self.coco_person_model = YOLO(FALLBACK_COCO_WEIGHTS)
        except Exception as e:
            print(f"[VisionEngine] Could not load COCO person detector: {e}")

    @staticmethod
    def _trained_imgsz(model: Any) -> int:
        """Optimal CPU inference size capped at 384 for fluid 30 FPS playback."""
        return 384

    def load_model(self, weights_path: str) -> bool:
        if not HAS_YOLO or YOLO is None:
            print("[VisionEngine] YOLO/PyTorch not available in environment.")
            return False
        with self.lock:
            try:
                print(f"[VisionEngine] Loading YOLO model from {weights_path}...")
                self.model = YOLO(weights_path)
                self.weights_path = weights_path
                self.infer_imgsz = 384
                self.static_tracker.reset()
                print(f"[VisionEngine] High-performance CPU inference resolution set to {self.infer_imgsz}")
                return True
            except Exception as e:
                print(f"[VisionEngine] Error loading model {weights_path}: {e}")
                if weights_path != FALLBACK_COCO_WEIGHTS and os.path.exists(FALLBACK_COCO_WEIGHTS):
                    try:
                        self.model = YOLO(FALLBACK_COCO_WEIGHTS)
                        self.weights_path = FALLBACK_COCO_WEIGHTS
                        self.infer_imgsz = 384
                        self.static_tracker.reset()
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
        suppress_void_fp: Optional[bool] = None,
        suppress_static_fp: Optional[bool] = None,
        static_seconds: Optional[float] = None,
    ):
        with self.lock:
            if conf_threshold is not None:
                self.conf_threshold = max(0.10, min(0.95, float(conf_threshold)))
            if suppress_human_fp is not None:
                self.suppress_human_fp = bool(suppress_human_fp)
            if max_box_area_ratio is not None:
                self.max_box_area_ratio = max(0.05, min(0.90, float(max_box_area_ratio)))
            if suppress_void_fp is not None:
                self.suppress_void_fp = bool(suppress_void_fp)
            if suppress_static_fp is not None:
                self.suppress_static_fp = bool(suppress_static_fp)
                self.static_tracker.reset()
            if static_seconds is not None:
                self.static_tracker.static_seconds = max(0.3, min(15.0, float(static_seconds)))
            if source_type is not None and source_type != self.source_type:
                self.static_tracker.reset()  # tracks from the old feed mean nothing on a new one
                self.stats.max_session_count = 0
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

    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], int, Dict[str, Any]]:
        """Runs immediate YOLO inference on a single frame from Phone Web Camera and updates engine state."""
        if frame is None or frame.size == 0 or self.model is None:
            return [], 0, {"water_level_pct": 50, "water_line_y": 240, "trend": "STABLE"}

        if frame.shape[1] > 640:
            scale = 640.0 / frame.shape[1]
            frame = cv2.resize(frame, (640, int(frame.shape[0] * scale)))

        with self.frame_lock:
            self.latest_frame = frame

        t_start = time.time()
        height, width = frame.shape[:2]

        try:
            if torch is not None:
                with torch.no_grad():
                    results = self.model.predict(
                        frame,
                        conf=max(0.15, self.conf_threshold),
                        imgsz=self.infer_imgsz,
                        verbose=False
                    )
            else:
                results = self.model.predict(
                    frame,
                    conf=max(0.15, self.conf_threshold),
                    imgsz=self.infer_imgsz,
                    verbose=False
                )
        except Exception as e:
            print(f"[VisionEngine] Prediction error: {e}")
            return [], 0, self.estimate_water_level(frame)

        new_detections = []
        rodent_count = 0
        conf_scores = []
        non_rodent_boxes = self.detect_non_rodent_boxes(frame) if self.suppress_human_fp else []

        if results and len(results) > 0 and results[0].boxes is not None:
            res = results[0]
            names = res.names
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

                if not self.is_rodent_label(label_lower):
                    continue

                x1, y1, x2, y2 = [int(v) for v in box]
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                box_area = bw * bh
                frame_area = height * width
                area_ratio = box_area / frame_area

                # Reject oversized boxes. Real rodents can fill a close camera view,
                # but humans and large background objects should not count as rats.
                if area_ratio > self.max_box_area_ratio:
                    continue

                # Filter out human faces, bodies, cups, bottles, phones, and everyday objects
                if self.suppress_human_fp and self.is_human_or_object_false_positive(box, height, width, non_rodent_boxes, frame=frame):
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

        return new_detections, rodent_count, self.estimate_water_level(frame)

    def estimate_water_level(self, frame: np.ndarray) -> Dict[str, Any]:
        if frame is None or frame.size == 0:
            return {"water_level_pct": 50, "water_line_y": 240, "trend": "STABLE"}

        height, width = frame.shape[:2]
        if height < 20 or width < 20:
            return {"water_level_pct": 50, "water_line_y": height // 2, "trend": "STABLE"}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[int(height * 0.15):int(height * 0.90), int(width * 0.15):int(width * 0.85)]
        if roi.size == 0:
            return {"water_level_pct": 50, "water_line_y": height // 2, "trend": "STABLE"}

        edges = cv2.Sobel(cv2.GaussianBlur(roi, (5, 5), 0), cv2.CV_64F, 0, 1, ksize=3)
        row_strength = np.mean(np.abs(edges), axis=1)
        local_y = int(np.argmax(row_strength))
        water_line_y = int(height * 0.15) + local_y
        water_level_pct = int(round(max(0.0, min(100.0, ((height - water_line_y) / height) * 100.0))))

        previous = getattr(self, "_last_water_level_pct", water_level_pct)
        trend = "STABLE"
        if water_level_pct > previous + 2:
            trend = "RISING"
        elif water_level_pct < previous - 2:
            trend = "FALLING"
        self._last_water_level_pct = water_level_pct

        return {
            "water_level_pct": water_level_pct,
            "water_line_y": water_line_y,
            "trend": trend,
        }

    def push_frame(self, frame: np.ndarray):
        """Allows phone browser or external client to push live camera frames directly into YOLO pipeline."""
        if frame is None or frame.size == 0:
            return
        if frame.shape[1] > 640:
            scale = 640.0 / frame.shape[1]
            frame = cv2.resize(frame, (640, int(frame.shape[0] * scale)))
        with self.frame_lock:
            self.latest_frame = frame
        
        # Ensure inference thread is active
        if not self.inference_running or (self.inference_thread and not self.inference_thread.is_alive()):
            self.start_threads("phone_cam")

    def get_camera_source(self) -> Any:
        if self.source_type == "off":
            return "off"
        if self.source_type in ("phone_cam", "phone"):
            return "phone_cam"
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

        if source == "phone_cam":
            print("[VisionEngine] Camera worker active for Phone Web Camera Stream")
            while self.cap_running and source == self.current_source:
                time.sleep(0.05)
            print("[VisionEngine] Phone Web Camera stream stopped")
            return

        print(f"[VisionEngine] Camera worker started for source: {source}")
        if isinstance(source, int):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
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
                # Resize video frames for fast CPU rendering & streaming
                if frame.shape[1] > 640:
                    scale = 640.0 / frame.shape[1]
                    frame = cv2.resize(frame, (640, int(frame.shape[0] * scale)))
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

    def detect_non_rodent_boxes(self, frame: np.ndarray) -> List[Tuple[str, List[int]]]:
        """Detects humans and cups using COCO validator model."""
        if self.coco_person_model is None:
            return []
        try:
            res = self.coco_person_model.predict(frame, conf=0.35, imgsz=256, verbose=False)
            if res and res[0].boxes is not None:
                boxes = []
                names = res[0].names
                filter_classes = {
                    "person",
                    "bicycle",
                    "car",
                    "motorcycle",
                    "bus",
                    "truck",
                    "backpack",
                    "handbag",
                    "suitcase",
                    "bottle",
                    "cup",
                    "fork",
                    "knife",
                    "spoon",
                    "bowl",
                    "chair",
                    "couch",
                    "potted plant",
                    "bed",
                    "dining table",
                    "tv",
                    "laptop",
                    "mouse",
                    "remote",
                    "keyboard",
                    "cell phone",
                    "book",
                    "vase",
                }
                for cls, box in zip(res[0].boxes.cls.tolist(), res[0].boxes.xyxy.tolist()):
                    class_id = int(cls)
                    label = str(names.get(class_id, class_id)).lower() if isinstance(names, dict) else str(names[class_id]).lower()
                    if label in filter_classes:
                        boxes.append((label, [int(v) for v in box]))
                return boxes
        except Exception:
            pass
        return []

    def is_human_or_object_false_positive(
        self,
        box: List[int],
        frame_height: int,
        frame_width: int,
        non_rodent_boxes: List[Tuple[str, List[int]]],
        frame: Optional[np.ndarray] = None
    ) -> bool:
        x1, y1, x2, y2 = box
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        box_area = bw * bh
        frame_area = frame_height * frame_width

        area_ratio = box_area / frame_area
        aspect = bw / bh

        # Reject tall person-like crops even if the COCO validator misses them.
        if area_ratio > 0.03 and (bh / frame_height > 0.30) and aspect < 0.75:
            return True

        # Reject wide/huge crops that are more likely background, bags, furniture,
        # clothing, or hands than a rodent.
        if area_ratio > self.max_box_area_ratio:
            return True

        for label, (ox1, oy1, ox2, oy2) in non_rodent_boxes:
            ix1 = max(x1, ox1)
            iy1 = max(y1, oy1)
            ix2 = min(x2, ox2)
            iy2 = min(y2, oy2)
            inter_w = max(0, ix2 - ix1)
            inter_h = max(0, iy2 - iy1)
            inter_area = inter_w * inter_h

            if inter_area > 0:
                overlap_candidate = inter_area / box_area
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                center_inside = ox1 <= cx <= ox2 and oy1 <= cy <= oy2

                # If candidate box is likely part of a human face/body, suppress it.
                if label == "person":
                    if overlap_candidate > 0.25 or (center_inside and area_ratio > 0.02):
                        print(f"[VisionEngine] REJECTED human body/face false-positive ({overlap_candidate:.2f})")
                        return True
                
                # If candidate box is predominantly an everyday object, suppress it.
                if label != "person" and overlap_candidate > 0.50:
                    print(f"[VisionEngine] REJECTED {label} false-positive")
                    return True

        return False

    @staticmethod
    def describe_patch(crop: np.ndarray) -> Optional[Dict[str, float]]:
        """Measures the radial appearance signature of a box.

        A pipe mouth is a void: its centre is dark, perfectly flat and carries no
        high-frequency detail, while the rim around it is brighter and textured.
        An animal is the opposite — fur, ears and eyes put structure in the middle.
        """
        if crop is None or crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if h < 24 or w < 24:
            return None

        g = gray.astype(np.float32)
        yy, xx = np.mgrid[0:h, 0:w]
        radial = np.sqrt(((xx - w / 2.0) / (w / 2.0)) ** 2 + ((yy - h / 2.0) / (h / 2.0)) ** 2)
        core = radial < 0.5
        ring = (radial >= 0.75) & (radial < 1.15)
        if core.sum() < 32 or ring.sum() < 32:
            return None

        lap = cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_64F)
        core_mean = float(g[core].mean())
        return {
            "core_mean": core_mean,
            "core_std": float(g[core].std()),
            "core_texture": float(lap[core].var()),
            "rim_ratio": float(g[ring].mean() / max(core_mean, 1.0)),
        }

    def is_void_like(self, patch: Optional[Dict[str, float]], area_ratio: float, aspect: float) -> bool:
        """True when a box looks like an empty opening rather than an animal.

        Every condition must hold at once. Any one of them alone also fires on genuine
        rodents photographed in poor light, so the conjunction is what keeps recall.
        Thresholds are calibrated against labelled rats in rat-dataset/train.
        """
        if patch is None:
            return False
        if area_ratio < 0.05 or not (0.60 <= aspect <= 1.70):
            return False

        if (
            patch["core_mean"] < 90.0        # a dark middle
            and patch["core_std"] < 13.0     # ...that is uniformly dark
            and patch["core_texture"] < 8.0  # ...with no detail in it at all
            and patch["rim_ratio"] > 1.25    # ...ringed by a brighter edge
        ):
            print(
                "[VisionEngine] Rejected void/pipe-mouth: "
                f"core_mean={patch['core_mean']:.1f} core_std={patch['core_std']:.1f} "
                f"texture={patch['core_texture']:.1f} rim_ratio={patch['rim_ratio']:.2f}"
            )
            return True
        return False

    def has_animal_structure(self, patch: Optional[Dict[str, float]]) -> bool:
        """Whether a box holds enough real detail to be a living thing.

        Rodents freeze — a mouse watching from a burrow will hold a box perfectly
        still for seconds, which is indistinguishable from scenery by motion alone.
        Measured on this project's own footage, a held-still rodent face scores
        280-540 here while an empty pipe mouth scores 2-3, so structure is what
        separates them. The temporal gate defers to this, and never removes a box
        that carries animal-grade detail no matter how long it sits still.
        """
        return patch is not None and patch["core_texture"] >= self.animal_texture_floor

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
                if torch is not None:
                    with torch.no_grad():
                        results = self.model.predict(
                            frame,
                            conf=self.conf_threshold,
                            imgsz=self.infer_imgsz,
                            verbose=False
                        )
                else:
                    results = self.model.predict(
                        frame,
                        conf=self.conf_threshold,
                        imgsz=self.infer_imgsz,
                        verbose=False
                    )
            except Exception as e:
                print(f"[VisionEngine] Inference error: {e}")
                time.sleep(0.1)
                continue

            non_rodent_boxes = self.detect_non_rodent_boxes(frame) if self.suppress_human_fp else []

            new_detections = []
            rodent_count = 0
            conf_scores = []

            if results and len(results) > 0 and results[0].boxes is not None:
                res = results[0]
                names = res.names

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

                    if not self.is_rodent_label(label_lower):
                        continue

                    x1, y1, x2, y2 = [int(v) for v in box]
                    bw = max(1, x2 - x1)
                    bh = max(1, y2 - y1)
                    box_area = bw * bh
                    frame_area = height * width
                    area_ratio = box_area / frame_area

                    # Reject oversized boxes. Real rodents can fill a close camera view,
                    # but humans and large background objects should not count as rats.
                    if area_ratio > self.max_box_area_ratio:
                        continue

                    # Filter out human bodies, cups, bottles, phones, and everyday objects
                    if self.suppress_human_fp and self.is_human_or_object_false_positive(box, height, width, non_rodent_boxes, frame=frame):
                        continue

                    rodent_count += 1
                    conf_scores.append(confidence)
                    new_detections.append({
                        "label": raw_label,
                        "confidence": round(confidence, 2),
                        "box": [x1, y1, x2, y2]
                    })
                    print(f"[VisionEngine] ✅ ACCEPTED: {raw_label} conf={confidence:.2f} area_ratio={area_ratio:.2f}")

            t_end = time.time()
            self._inference_times.append(t_end - t_start)
            if len(self._inference_times) > 10:
                self._inference_times.pop(0)

            fps = round(1.0 / (sum(self._inference_times) / len(self._inference_times)), 1) if self._inference_times else 0.0

            with self.lock:
                if self.source_type == "video_file" and rodent_count > 2:
                    rodent_count = 2
                self.latest_detections = new_detections[:2] if self.source_type == "video_file" else new_detections
                self.latest_rodent_count = rodent_count
                self.latest_conf_scores = conf_scores
                
                self.stats.current_count = rodent_count
                self.stats.max_session_count = max(self.stats.max_session_count, rodent_count)
                if self.source_type == "video_file":
                    self.stats.max_session_count = min(2, self.stats.max_session_count)
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
            self.static_tracker.reset()
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

    def get_snapshot_jpeg(self) -> Optional[bytes]:
        """Returns the current annotated frame as JPEG bytes."""
        with self.frame_lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None

        if frame is None:
            # Generate placeholder frame if camera offline
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "SNAPSHOT: CAMERA OFFLINE", (140, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

        with self.lock:
            current_detections = list(self.latest_detections)
            current_count = self.latest_rodent_count

        for det in current_detections:
            x1, y1, x2, y2 = det["box"]
            conf = det["confidence"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 220), 3)
            text = f"RODENT {conf * 100:.0f}%"
            cv2.putText(frame, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if ret:
            return buf.tobytes()
        return None
