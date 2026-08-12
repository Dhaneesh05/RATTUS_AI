"""Retrains the rodent detector with the hard negatives applied, then proves it helped.

Trains into a *new* run directory, so the current weights stay untouched and usable
until the replacement is shown to be better. The comparison that matters here is not
mAP alone but the false-positive count on the empty-pipe negatives, since that is the
failure being fixed — reported side by side for the old and new weights.

Usage:
    python tools/retrain.py                 # train, then compare
    python tools/retrain.py --compare-only  # just score existing weights
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from vision_engine import DEFAULT_RAT_WEIGHTS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(ROOT, "rat-dataset", "data.yaml")
LEDGER = os.path.join(ROOT, "rat-dataset", ".hard_negatives_applied.json")


def background_images() -> List[str]:
    """The negative images now in the dataset — our false-positive test set."""
    if not os.path.exists(LEDGER):
        return []
    with open(LEDGER, encoding="utf-8") as fh:
        files = json.load(fh).get("files", [])
    return [os.path.join(ROOT, "rat-dataset", f) for f in files
            if f.lower().endswith((".jpg", ".png"))]


def score(weights: str, imgsz: int, conf: float, negatives: List[str]) -> Optional[Dict[str, float]]:
    if not os.path.exists(weights):
        print(f"  (missing: {weights})")
        return None
    model = YOLO(weights)
    metrics = model.val(data=DATA_YAML, imgsz=imgsz, verbose=False, plots=False)

    fp_boxes = fp_images = 0
    for i in range(0, len(negatives), 16):
        for res in model.predict(negatives[i:i + 16], conf=conf, imgsz=imgsz, verbose=False):
            n = len(res.boxes)
            fp_boxes += n
            fp_images += 1 if n else 0

    return {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "recall": float(metrics.box.mr),
        "precision": float(metrics.box.mp),
        "fp_boxes": fp_boxes,
        "fp_images": fp_images,
        "negatives": len(negatives),
    }


def report(old: Optional[Dict[str, float]], new: Optional[Dict[str, float]]) -> None:
    print("\n" + "=" * 66)
    print("BEFORE / AFTER")
    print("=" * 66)
    if not old or not new:
        for name, s in (("old", old), ("new", new)):
            if s:
                print(f"  {name}: mAP50={s['mAP50']:.3f} recall={s['recall']:.3f} "
                      f"FP boxes={s['fp_boxes']}")
        return

    rows = [("mAP50", "mAP50", "higher"), ("mAP50-95", "mAP50-95", "higher"),
            ("precision", "precision", "higher"), ("recall", "recall", "higher")]
    print(f"  {'metric':<12}{'old':>10}{'new':>10}{'change':>12}")
    for label, key, _ in rows:
        o, n = old[key], new[key]
        print(f"  {label:<12}{o:>10.3f}{n:>10.3f}{n - o:>+12.3f}")

    o, n = old["fp_boxes"], new["fp_boxes"]
    print(f"  {'FP boxes':<12}{o:>10d}{n:>10d}{n - o:>+12d}"
          f"   (on {new['negatives']} empty-pipe images)")
    oi, ni = old["fp_images"], new["fp_images"]
    print(f"  {'FP images':<12}{oi:>10d}{ni:>10d}{ni - oi:>+12d}")

    print()
    if n < o and new["recall"] >= old["recall"] - 0.03:
        print("  VERDICT: false positives down, recall held. Ship it.")
    elif n < o:
        print(f"  VERDICT: false positives down, but recall fell "
              f"{old['recall'] - new['recall']:.3f}. Check the negatives are "
              f"genuinely rat-free before adopting.")
    else:
        print("  VERDICT: no false-positive improvement. Mine more/better negatives "
              "rather than training longer.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=416, help="must match how you infer")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--base", default="yolov8n.pt", help="starting weights")
    ap.add_argument("--name", default="rat_yolov8_hn")
    ap.add_argument("--conf", type=float, default=0.40, help="threshold for FP counting")
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--new-weights", default=None)
    args = ap.parse_args()

    negatives = background_images()
    if not negatives and not args.compare_only:
        print("No hard negatives applied yet — run tools/apply_hard_negatives.py first.")
        return 1
    print(f"False-positive test set: {len(negatives)} background images")

    old_weights = os.path.join(ROOT, DEFAULT_RAT_WEIGHTS)
    new_weights = args.new_weights or os.path.join(
        ROOT, "runs", "detect", args.name, "weights", "best.pt")

    if not args.compare_only:
        device = 0 if torch.cuda.is_available() else "cpu"
        print(f"\nTraining {args.base} -> {args.name} | imgsz={args.imgsz} "
              f"epochs={args.epochs} device={device}")
        if device == "cpu":
            print("  No CUDA available; on CPU this takes a while. "
                  "Reduce --epochs to iterate faster.")
        t0 = time.time()
        YOLO(args.base).train(
            data=DATA_YAML, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            name=args.name, project=os.path.join(ROOT, "runs", "detect"),
            device=device, verbose=True, exist_ok=True,
        )
        print(f"\nTraining finished in {(time.time() - t0) / 60:.1f} min")
        found = glob.glob(os.path.join(ROOT, "runs", "detect", args.name, "weights", "best.pt"))
        if found:
            new_weights = found[0]

    print(f"\nScoring OLD: {old_weights}")
    old = score(old_weights, args.imgsz, args.conf, negatives)
    print(f"Scoring NEW: {new_weights}")
    new = score(new_weights, args.imgsz, args.conf, negatives)
    report(old, new)

    if new:
        print(f"\nTo adopt, point the app at the new weights:\n"
              f'  curl -X POST localhost:8000/api/config -H "Content-Type: application/json" \\\n'
              f'       -d \'{{"weights_path": "{new_weights}"}}\'')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
