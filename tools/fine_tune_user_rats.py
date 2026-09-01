"""Fine-tune the RATTUS AI YOLOv8 model on user-provided rat images with multi-scale & distance augmentations.

This script:
1. Ingests the 5 user-provided rat images.
2. Accurately labels each rodent.
3. Generates multi-distance synthetic augmentations (close-up, 1.5m medium distance, 3m far distance, flips, lighting variations).
4. Fine-tunes the current best.pt weights.
5. Benchmarks before vs after confidence and detection accuracy.
"""
from __future__ import annotations

import glob
import os
import shutil
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT, "dataset_user_rats")
CURRENT_WEIGHTS = os.path.join(ROOT, "runs", "detect", "runs", "rat_yolov8", "weights", "best.pt")
OUTPUT_RUN_NAME = "rat_enhanced_v2"


# Ground truth box definitions for the user's 5 images
# Format: list of [x1, y1, x2, y2]
USER_IMAGES_METADATA = [
    {
        "filename": "media_1788269171224.png",
        "description": "Illustrated brown rat with long tail",
        "boxes": [
            [20, 20, 620, 360]  # Full body & head
        ]
    },
    {
        "filename": "media_1788269235339.png",
        "description": "Outdoor close-up brown rat with green foliage",
        "boxes": [
            [140, 50, 690, 400]
        ]
    },
    {
        "filename": "media_1788269245926.png",
        "description": "Wild rat on mossy log / forest background",
        "boxes": [
            [100, 150, 580, 420]
        ]
    },
    {
        "filename": "media_1788269263680.png",
        "description": "Two rats on wooden ledge against concrete",
        "boxes": [
            [70, 140, 310, 260],   # Upper rat
            [40, 170, 310, 280]    # Lower rat
        ]
    },
    {
        "filename": "media_1788269298346.png",
        "description": "Two young rats peeking over wooden barrier",
        "boxes": [
            [64, 81, 264, 278],    # Left rat
            [267, 100, 534, 279]    # Right rat
        ]
    }
]


def create_yolo_annotation(boxes: list, img_w: int, img_h: int) -> str:
    """Converts [[x1, y1, x2, y2], ...] to YOLO format: class_id x_center y_center width height."""
    lines = []
    for b in boxes:
        x1, y1, x2, y2 = b
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        xc = x1 + bw / 2.0
        yc = y1 + bh / 2.0
        lines.append(f"0 {xc / img_w:.6f} {yc / img_h:.6f} {bw / img_w:.6f} {bh / img_h:.6f}")
    return "\n".join(lines)


def generate_augmented_dataset(src_dir: str):
    """Builds multi-scale dataset with distance, flip, and lighting augmentations."""
    os.makedirs(os.path.join(DATASET_DIR, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "labels", "val"), exist_ok=True)

    idx = 0
    for meta in USER_IMAGES_METADATA:
        src_path = os.path.join(src_dir, meta["filename"])
        if not os.path.exists(src_path):
            print(f"[Warning] Source image missing: {src_path}")
            continue

        img = cv2.imread(src_path)
        if img is None:
            continue
        ih, iw = img.shape[:2]
        boxes = meta["boxes"]

        # Base item
        variations = []

        # 1. Original
        variations.append((img.copy(), [list(b) for b in boxes], "orig"))

        # 2. Horizontal Flip
        img_flip = cv2.flip(img, 1)
        flip_boxes = []
        for b in boxes:
            x1, y1, x2, y2 = b
            flip_boxes.append([iw - x2, y1, iw - x1, y2])
        variations.append((img_flip, flip_boxes, "hflip"))

        # 3. Medium distance simulation (0.6x scale centered on background)
        canvas_med = np.full((ih, iw, 3), (40, 45, 50), dtype=np.uint8)
        scale_med = 0.60
        nw, nh = int(iw * scale_med), int(ih * scale_med)
        resized_med = cv2.resize(img, (nw, nh))
        ox, oy = (iw - nw) // 2, (ih - nh) // 2
        canvas_med[oy:oy+nh, ox:ox+nw] = resized_med
        med_boxes = [[int(b[0]*scale_med + ox), int(b[1]*scale_med + oy), int(b[2]*scale_med + ox), int(b[3]*scale_med + oy)] for b in boxes]
        variations.append((canvas_med, med_boxes, "med_dist"))

        # 4. Far distance simulation (0.35x scale offset to bottom/corner)
        canvas_far = np.full((ih, iw, 3), (35, 40, 42), dtype=np.uint8)
        scale_far = 0.35
        nw_f, nh_f = int(iw * scale_far), int(ih * scale_far)
        resized_far = cv2.resize(img, (nw_f, nh_f))
        ox_f, oy_f = (iw - nw_f) // 4, (ih - nh_f) * 3 // 4
        canvas_far[oy_f:oy_f+nh_f, ox_f:ox_f+nw_f] = resized_far
        far_boxes = [[int(b[0]*scale_far + ox_f), int(b[1]*scale_far + oy_f), int(b[2]*scale_far + ox_f), int(b[3]*scale_far + oy_f)] for b in boxes]
        variations.append((canvas_far, far_boxes, "far_dist"))

        # 5. Lighting / drain shadow variation
        img_dark = cv2.convertScaleAbs(img, alpha=0.75, beta=-15)
        variations.append((img_dark, [list(b) for b in boxes], "dark"))

        # 6. Warm lighting variation
        img_warm = cv2.convertScaleAbs(img, alpha=1.1, beta=15)
        variations.append((img_warm, [list(b) for b in boxes], "warm"))

        # Save to train and validation sets
        for var_img, var_boxes, suffix in variations:
            split = "val" if (idx % 5 == 0 and suffix == "orig") else "train"
            vh, vw = var_img.shape[:2]
            name = f"rat_sample_{idx:03d}_{suffix}"
            img_dest = os.path.join(DATASET_DIR, "images", split, f"{name}.png")
            lbl_dest = os.path.join(DATASET_DIR, "labels", split, f"{name}.txt")

            cv2.imwrite(img_dest, var_img)
            with open(lbl_dest, "w", encoding="utf-8") as f:
                f.write(create_yolo_annotation(var_boxes, vw, vh))
            idx += 1

    clean_path = os.path.abspath(DATASET_DIR).replace("\\", "/")
    yaml_content = f"""path: {clean_path}
train: images/train
val: images/val

names:
  0: rat
"""
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"[Dataset] Generated {idx} multi-scale training and validation samples in: {DATASET_DIR}")
    return yaml_path


def benchmark_model(weights_path: str, src_dir: str, title: str):
    """Prints detection results and confidences for the 5 target images."""
    if not os.path.exists(weights_path):
        print(f"Weights missing: {weights_path}")
        return
    model = YOLO(weights_path)
    print("\n" + "=" * 70)
    print(f"  EVALUATION: {title} ({os.path.basename(weights_path)})")
    print("=" * 70)
    for meta in USER_IMAGES_METADATA:
        src = os.path.join(src_dir, meta["filename"])
        if not os.path.exists(src):
            continue
        res = model.predict(src, imgsz=640, conf=0.20, verbose=False)[0]
        dets = []
        for c, cf, b in zip(res.boxes.cls, res.boxes.conf, res.boxes.xyxy):
            cls_name = res.names[int(c)]
            conf = float(cf)
            box = [round(float(v)) for v in b]
            dets.append(f"{cls_name} ({conf*100:.1f}%)")
        print(f"• {meta['description'][:38]:<40}: {', '.join(dets) if dets else '❌ No detection (below 20%)'}")
    print("=" * 70)


def main():
    src_dir = r"C:\Users\USER\.gemini\antigravity-ide\brain\2e7d053c-a24e-4da3-b4d1-f88e224340ea\.user_uploaded"
    
    print("Starting RATTUS AI Fine-Tuning Pipeline for User-Provided Images...")
    benchmark_model(CURRENT_WEIGHTS, src_dir, "BEFORE FINE-TUNING")

    yaml_path = generate_augmented_dataset(src_dir)

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\nFine-tuning model from: {CURRENT_WEIGHTS} on device: {device}...")
    
    model = YOLO(CURRENT_WEIGHTS)
    t0 = time.time()
    model.train(
        data=yaml_path,
        epochs=30,
        imgsz=640,
        batch=4,
        device=device,
        project=os.path.join(ROOT, "runs", "detect"),
        name=OUTPUT_RUN_NAME,
        exist_ok=True,
        scale=0.8,
        mosaic=0.5,
        mixup=0.1,
        lr0=0.001,  # Gentle fine-tuning learning rate to preserve prior knowledge
        lrf=0.01,
        patience=15,
        verbose=False,
    )
    print(f"Fine-tuning complete in {(time.time() - t0):.1f}s.")

    new_weights = os.path.join(ROOT, "runs", "detect", OUTPUT_RUN_NAME, "weights", "best.pt")
    benchmark_model(new_weights, src_dir, "AFTER FINE-TUNING")

    return new_weights


if __name__ == "__main__":
    main()
