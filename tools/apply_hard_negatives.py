"""Files reviewed hard negatives into the training set as background images.

YOLO treats an image with an empty label file as background: everything in it is
something the model must learn *not* to fire on. Ultralytics recommends backgrounds
be roughly 0-10% of a dataset — enough to correct false positives, not so much that
the model turns timid.

Only images still present in hard_negatives/images are applied, so review is simply
deleting the bad ones. Every file added is recorded, and --undo removes exactly
those again, so a bad batch is never baked in.

Usage:
    python tools/apply_hard_negatives.py            # review first!
    python tools/apply_hard_negatives.py --undo
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "rat-dataset")
LEDGER = os.path.join(DATASET, ".hard_negatives_applied.json")
MAX_BACKGROUND_SHARE = 0.10


def count_images(split: str) -> int:
    d = os.path.join(DATASET, split, "images")
    return len(os.listdir(d)) if os.path.isdir(d) else 0


def undo() -> int:
    if not os.path.exists(LEDGER):
        print("Nothing to undo — no ledger found.")
        return 0
    with open(LEDGER, encoding="utf-8") as fh:
        ledger = json.load(fh)
    removed = 0
    for rel in ledger.get("files", []):
        p = os.path.join(DATASET, rel)
        if os.path.exists(p):
            os.remove(p)
            removed += 1
    os.remove(LEDGER)
    print(f"Removed {removed} files. Dataset restored to "
          f"train={count_images('train')} valid={count_images('valid')}.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.path.join(ROOT, "hard_negatives"))
    ap.add_argument("--val-share", type=float, default=0.2,
                    help="fraction held out in valid/, to measure the FP fix honestly")
    ap.add_argument("--undo", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="apply even if backgrounds would exceed the 10%% share guidance")
    args = ap.parse_args()

    if args.undo:
        return undo()

    if os.path.exists(LEDGER):
        print("Hard negatives are already applied. Run --undo first to re-apply.")
        return 1

    manifest_path = os.path.join(args.src, "manifest.json")
    img_dir = os.path.join(args.src, "images")
    if not os.path.exists(manifest_path):
        print(f"No manifest at {manifest_path} — run mine_hard_negatives.py first.")
        return 1

    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    proposed = [s["file"] for s in manifest["samples"]]
    surviving = [f for f in proposed if os.path.exists(os.path.join(img_dir, f))]
    extra = [f for f in sorted(os.listdir(img_dir))
             if f.lower().endswith((".jpg", ".png")) and f not in proposed]
    surviving.extend(extra)

    rejected = len(proposed) - len([f for f in proposed if f in surviving])
    if not surviving:
        print("No images left in the review folder — nothing to apply.")
        return 1

    base = count_images("train") + count_images("valid")
    share = len(surviving) / float(base + len(surviving))
    print(f"Reviewed: {len(surviving)} kept, {rejected} rejected during review.")
    print(f"Dataset is currently {base} images; adding {len(surviving)} backgrounds "
          f"= {share * 100:.1f}% of the total.")
    if share > MAX_BACKGROUND_SHARE and not args.force:
        print(f"\nThat exceeds the {MAX_BACKGROUND_SHARE * 100:.0f}% guidance and risks "
              "suppressing real detections.\nTrim the review folder, or pass --force "
              "if you're sure.")
        return 1

    random.seed(0)
    shuffled = surviving[:]
    random.shuffle(shuffled)
    n_val = int(len(shuffled) * args.val_share)
    assignment = [("valid", f) for f in shuffled[:n_val]] + \
                 [("train", f) for f in shuffled[n_val:]]

    added: List[str] = []
    for split, fname in assignment:
        stem = os.path.splitext(fname)[0]
        img_rel = os.path.join(split, "images", fname)
        lbl_rel = os.path.join(split, "labels", stem + ".txt")
        shutil.copy2(os.path.join(img_dir, fname), os.path.join(DATASET, img_rel))
        # An empty label file is the whole point: "this image contains nothing".
        open(os.path.join(DATASET, lbl_rel), "w", encoding="utf-8").close()
        added.extend([img_rel, lbl_rel])

    meta: Dict[str, Any] = {"source": args.src, "count": len(assignment),
                            "val_share": args.val_share, "files": added}
    with open(LEDGER, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nApplied {len(assignment)} backgrounds "
          f"({len(assignment) - n_val} train, {n_val} valid).")
    print(f"  train images: {count_images('train')}   valid images: {count_images('valid')}")
    print(f"  ledger -> {LEDGER}  (undo with: python tools/apply_hard_negatives.py --undo)")
    print("\nNEXT: python tools/retrain.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
