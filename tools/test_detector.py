"""Smoke-test the rodent detector on images or sampled video frames.

Examples:
    python tools/test_detector.py runs/detect/runs/rat_smoke_test/*.jpg
    python tools/test_detector.py uploads/demo.mp4 --every 30 --conf 0.5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision_engine import DEFAULT_RAT_WEIGHTS, VisionEngine  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def iter_paths(inputs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS:
                    paths.append(child)
        elif path.exists():
            paths.append(path)
        else:
            print(f"missing: {path}", file=sys.stderr)
    return paths


def test_image(engine: VisionEngine, path: Path) -> tuple[int, int]:
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"{path}: unreadable")
        return 0, 0
    result = engine.process_frame(frame)
    detections, count = result[0], result[1]
    details = ", ".join(f"{d['label']} {d['confidence']:.2f} {d['box']}" for d in detections)
    print(f"{path}: {count} rodent(s)" + (f" | {details}" if details else ""))
    return 1, count


def test_video(engine: VisionEngine, path: Path, every: int, max_frames: int) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"{path}: unreadable")
        return 0, 0

    sampled = 0
    positive = 0
    frame_index = 0
    while sampled < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % every == 0:
            result = engine.process_frame(frame)
            detections, count = result[0], result[1]
            sampled += 1
            if count:
                positive += 1
                details = ", ".join(f"{d['confidence']:.2f} {d['box']}" for d in detections)
                print(f"{path}: frame {frame_index}: {count} rodent(s) | {details}")
        frame_index += 1
    cap.release()
    print(f"{path}: sampled {sampled} frame(s), positives {positive}")
    return sampled, positive


def main() -> int:
    parser = argparse.ArgumentParser(description="Test RATTUS AI rodent detector on local media.")
    parser.add_argument("inputs", nargs="+", help="image, video, or directory paths")
    parser.add_argument("--weights", default=DEFAULT_RAT_WEIGHTS, help="YOLO weights path")
    parser.add_argument("--conf", type=float, default=0.50, help="minimum confidence")
    parser.add_argument("--max-box-area", type=float, default=0.55, help="max accepted box area as frame ratio")
    parser.add_argument("--every", type=int, default=30, help="sample every Nth video frame")
    parser.add_argument("--max-frames", type=int, default=300, help="max sampled frames per video")
    parser.add_argument("--no-human-filter", action="store_true", help="disable human/object suppression")
    args = parser.parse_args()

    engine = VisionEngine(default_weights=args.weights)
    engine.set_config(
        conf_threshold=args.conf,
        max_box_area_ratio=args.max_box_area,
        suppress_human_fp=not args.no_human_filter,
    )

    print(f"weights: {engine.weights_path}")
    print(f"classes: {engine.model.names if engine.model is not None else {}}")
    print(f"conf: {engine.conf_threshold:.2f}, max_box_area: {engine.max_box_area_ratio:.2f}")

    paths = iter_paths(args.inputs)
    if not paths:
        print("No readable media paths were provided.", file=sys.stderr)
        return 2

    total_samples = 0
    total_positive = 0
    for path in paths:
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTS:
            sampled, positives = test_image(engine, path)
        elif suffix in VIDEO_EXTS:
            sampled, positives = test_video(engine, path, max(1, args.every), max(1, args.max_frames))
        else:
            continue
        total_samples += sampled
        total_positive += positives

    print(f"summary: {total_positive}/{total_samples} sample(s) had rodent detections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
