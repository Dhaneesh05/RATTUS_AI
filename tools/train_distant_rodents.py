"""Train YOLOv8 specifically optimized for Far & Small Rodent Detection.

Key Small-Object Detection Techniques included:
1. Higher resolution training (imgsz=640 instead of 416/384).
2. Multi-scale & scale augmentation (scale=0.9) to train on shrunk representations.
3. Mosaic augmentation (mosaic=1.0) which combines 4 scenes into 1, effectively
   reducing target rodent size by ~4x to simulate far-distance viewing.
4. Support for yolov8s.pt (Small) which has stronger feature extraction for small
   objects compared to yolov8n (Nano).

Usage:
    python tools/train_distant_rodents.py --data path/to/data.yaml --epochs 50
    python tools/train_distant_rodents.py --data path/to/data.yaml --base yolov8s.pt --epochs 50
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import torch
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Small & Distant Rodent Detection")
    parser.add_argument(
        "--data",
        default=os.path.join(ROOT, "rat-dataset", "data.yaml"),
        help="Path to dataset data.yaml file",
    )
    parser.add_argument(
        "--base",
        default="yolov8s.pt",
        help="Starting model checkpoint (yolov8s.pt recommended for distant objects, or yolov8n.pt / existing best.pt)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (640 recommended)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (reduce if VRAM/RAM limited)")
    parser.add_argument("--name", default="rat_distant_yolov8", help="Experiment name")
    parser.add_argument("--device", default=None, help="Device to use ('0' for CUDA GPU, 'cpu' for CPU)")

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"\n[ERROR] Dataset YAML file not found: {args.data}")
        print("\nTo train the model, you need a dataset in YOLO format (with data.yaml, train/, val/).")
        print("Example directory structure:")
        print("  rat-dataset/")
        print("  |-- data.yaml")
        print("  |-- images/")
        print("  |   |-- train/")
        print("  |   |-- val/")
        print("  |-- labels/")
        print("      |-- train/")
        print("      |-- val/")
        return 1

    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("  RATTUS AI - FAR & SMALL RODENT MODEL TRAINING PIPELINE")
    print("=" * 60)
    print(f"Base Weights : {args.base}")
    print(f"Dataset YAML : {args.data}")
    print(f"Image Size   : {args.imgsz} (High detail for distant objects)")
    print(f"Epochs       : {args.epochs}")
    print(f"Batch Size   : {args.batch}")
    print(f"Compute Dev  : {device}")
    print("=" * 60)

    model = YOLO(args.base)

    t0 = time.time()
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=os.path.join(ROOT, "runs", "detect"),
        name=args.name,
        exist_ok=True,
        # Hyperparameters targeted specifically for small/distant objects:
        scale=0.9,          # Heavy scale augmentation: shrinks objects randomly to teach distant detection
        mosaic=1.0,         # 4-image mosaic: shrinks targets by ~4x to simulate distance
        mixup=0.15,         # Blend background clutter to prevent false positives
        translate=0.15,     # Object translation
        degrees=10.0,       # Rotation tolerance
        fliplr=0.5,         # Horizontal flips
        patience=20,        # Early stopping patience
        save=True,
        plots=True,
        verbose=True,
    )

    elapsed_min = (time.time() - t0) / 60
    print(f"\n[Training Complete] Finished in {elapsed_min:.1f} minutes.")

    best_weights = os.path.join(ROOT, "runs", "detect", args.name, "weights", "best.pt")
    if os.path.exists(best_weights):
        print(f"\nSuccessfully saved new model weights to:\n  -> {best_weights}")
        print("\nTo load these new weights into your running RATTUS AI app:")
        print(f"Option A (API): POST /api/config with {{\"weights_path\": \"{best_weights}\"}}")
        print("Option B (Web UI): Select the new model from the settings dropdown.")
    else:
        print(f"\nCheck run output under: {os.path.join(ROOT, 'runs', 'detect', args.name)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
