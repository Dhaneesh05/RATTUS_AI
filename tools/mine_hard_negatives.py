"""Mines hard negatives from demo footage.

A hard negative is a frame the current model gets *wrong* — a pipe mouth it calls a
rat. Training on those directly is far more effective than training on arbitrary
rat-free images, because it corrects the exact mistake being made.

The danger is mislabelling: a frame that actually contains a rat, filed as
background, teaches the model to ignore rats. Asking the model whether a rat is
present is circular — it is the thing being wrong. So presence is decided on
*motion* instead: a live animal moves, sediment and pipe walls do not. A frame is
only proposed when every detection in it has been pinned to the same pixels for
--static-seconds, and everything proposed still goes to a contact sheet for human
review before it can reach the training set (see apply_hard_negatives.py).

For footage you already know is rat-free, skip the inference entirely:
    --rat-free "uploads/clip.mp4:12.5-18.0"

Usage:
    python tools/mine_hard_negatives.py
    python tools/mine_hard_negatives.py --videos "uploads/rat in drain demo video.mp4"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision_engine import DEFAULT_RAT_WEIGHTS, StaticRegionTracker, VisionEngine  # noqa: E402

# Any detection this weak that is *not* void-like disqualifies a sample. Well below
# the 0.40 runtime threshold: we would rather throw away a good negative than teach
# the model to ignore a real rat.
VETO_CONF = 0.20
# When a rat shares the frame with a false positive, the frame is still unusable
# whole — but a window around the false positive is fine, provided no rat comes
# within this many pixels of it.
CROP_PAD = 0.6      # context to keep around the box, as a fraction of its size
CROP_SAFETY_PX = 24
MIN_CROP_PX = 96
GRID = (4, 3)  # contact sheet tiles per page


def dhash(frame: np.ndarray, size: int = 8) -> int:
    """Perceptual hash, so 300 near-identical frames don't become 300 samples."""
    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (size + 1, size),
                       interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    return int("".join("1" if b else "0" for b in bits.flatten()), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def scan_video(engine: VisionEngine, path: str, keep_every: int, max_keep: int,
               min_hamming: int, static_seconds: float) -> List[Dict[str, Any]]:
    """Walks a clip in order, keeping frames whose every detection is motionless."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  !! cannot open {path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = StaticRegionTracker()
    tracker.static_seconds = static_seconds

    kept: List[Dict[str, Any]] = []
    hashes: List[int] = []
    stats = {"scanned": 0, "moving": 0, "dupes": 0, "clean": 0}
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Every frame is scanned: the motion tracker needs an unbroken sequence to
        # tell a pinned region from one that merely looks similar between samples.
        if frame.shape[1] > 960:  # match the resolution the live engine works at
            scale = 960 / frame.shape[1]
            frame = cv2.resize(frame, (960, int(frame.shape[0] * scale)))
        h, w = frame.shape[:2]
        now = idx / fps
        idx += 1
        stats["scanned"] += 1

        results = engine.model.predict(frame, conf=VETO_CONF, imgsz=engine.infer_imgsz,
                                       verbose=False)
        boxes = [[int(v) for v in b] for b in results[0].boxes.xyxy.tolist()]
        confs = results[0].boxes.conf.tolist()
        frozen = tracker.update(frame, boxes, now)

        if not boxes:
            stats["clean"] += 1
            continue  # nothing to learn from: the model already gets this right

        still: List[Dict[str, Any]] = []
        movers: List[List[int]] = []
        for box, conf, is_frozen in zip(boxes, confs, frozen):
            if is_frozen:
                x1, y1, x2, y2 = box
                bw, bh = max(1, x2 - x1), max(1, y2 - y1)
                metrics = engine.describe_patch(frame[max(0, y1):min(h, y2),
                                                      max(0, x1):min(w, x2)])
                still.append({"box": box, "conf": round(float(conf), 3),
                              "metrics": {k: round(v, 1) for k, v in (metrics or {}).items()}})
            else:
                movers.append(box)

        if not still:
            stats["moving"] += 1
            continue

        if not movers:
            # Nothing in this frame is moving, so nothing alive is in it. The whole
            # frame is usable, and a full frame is the most faithful thing to train on.
            proposals = [(frame, still, "frame")]
        else:
            # Something is alive elsewhere in shot. Salvage a padded window around
            # each motionless false positive, provided no mover comes near it.
            proposals = []
            for hit in still:
                x1, y1, x2, y2 = hit["box"]
                px, py = int((x2 - x1) * CROP_PAD), int((y2 - y1) * CROP_PAD)
                cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
                cx2, cy2 = min(w, x2 + px), min(h, y2 + py)
                if cx2 - cx1 < MIN_CROP_PX or cy2 - cy1 < MIN_CROP_PX:
                    continue
                s = CROP_SAFETY_PX
                if any(mx1 < cx2 + s and mx2 > cx1 - s and my1 < cy2 + s and my2 > cy1 - s
                       for mx1, my1, mx2, my2 in movers):
                    stats["moving"] += 1
                    continue
                shifted = dict(hit, box=[x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1])
                proposals.append((frame[cy1:cy2, cx1:cx2].copy(), [shifted], "crop"))

        if (idx - 1) % keep_every:
            continue  # thin out the survivors; consecutive frames are near-identical

        for image, hits, kind in proposals:
            fh = dhash(image)
            if any(hamming(fh, prev) < min_hamming for prev in hashes):
                stats["dupes"] += 1
                continue
            hashes.append(fh)
            kept.append({"frame_index": idx - 1, "source": path, "kind": kind,
                         "false_positives": hits, "image": image})

        if len(kept) >= max_keep:
            break

    cap.release()
    frames = sum(1 for k in kept if k["kind"] == "frame")
    print(f"  {os.path.basename(path):34s} {total:5d}f | scanned {stats['scanned']:4d} "
          f"| no-detections {stats['clean']:4d} | moving {stats['moving']:4d} "
          f"| dupes {stats['dupes']:4d} | KEPT {len(kept)} "
          f"({frames} full, {len(kept) - frames} crops)")
    return kept


def harvest_rat_free(spec: str, keep_every: int, min_hamming: int) -> List[Dict[str, Any]]:
    """Takes every Nth frame of a clip range the user has declared rat-free.

    No inference and no veto — the operator's word is the ground truth here, which
    makes this both the highest-yield and the least ambiguous source of negatives.
    """
    path, _, span = spec.rpartition(":")
    if not path or not os.path.exists(path):
        print(f"  !! bad --rat-free spec (expected 'file.mp4:start-end'): {spec}")
        return []
    try:
        start_s, _, end_s = span.partition("-")
        start, end = float(start_s), float(end_s)
    except ValueError:
        print(f"  !! bad time range in: {spec}")
        return []

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    first, last = int(start * fps), int(end * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    kept: List[Dict[str, Any]] = []
    hashes: List[int] = []
    dupes = 0
    for idx in range(first, last + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - first) % keep_every:
            continue
        if frame.shape[1] > 960:
            scale = 960 / frame.shape[1]
            frame = cv2.resize(frame, (960, int(frame.shape[0] * scale)))
        fh = dhash(frame)
        if any(hamming(fh, prev) < min_hamming for prev in hashes):
            dupes += 1
            continue
        hashes.append(fh)
        kept.append({"frame_index": idx, "source": path, "kind": "rat-free",
                     "false_positives": [], "image": frame})
    cap.release()
    print(f"  {os.path.basename(path):34s} {start:.1f}-{end:.1f}s declared rat-free "
          f"| dupes {dupes:4d} | KEPT {len(kept)}")
    return kept


def write_contact_sheets(samples: List[Dict[str, Any]], out_dir: str) -> List[str]:
    """Renders the proposals as reviewable grids with the false positives outlined."""
    cols, rows = GRID
    per_page = cols * rows
    tile_w, tile_h = 420, 260
    sheets = []

    for page in range((len(samples) + per_page - 1) // per_page):
        chunk = samples[page * per_page:(page + 1) * per_page]
        sheet = np.full((rows * tile_h, cols * tile_w, 3), 22, np.uint8)
        for i, s in enumerate(chunk):
            img = s["image"].copy()
            for fp in s["false_positives"]:
                x1, y1, x2, y2 = fp["box"]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 215, 255), 2)
            sx, sy = tile_w / img.shape[1], (tile_h - 22) / img.shape[0]
            scale = min(sx, sy)
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
            r, c = divmod(i, cols)
            oy, ox = r * tile_h + 22, c * tile_w
            sheet[oy:oy + img.shape[0], ox:ox + img.shape[1]] = img
            cv2.putText(sheet, f"#{page * per_page + i:03d}  frame {s['frame_index']}  "
                              f"fp={len(s['false_positives'])}",
                        (ox + 6, r * tile_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 215, 255), 1, cv2.LINE_AA)
        path = os.path.join(out_dir, f"contact_sheet_{page + 1:02d}.jpg")
        cv2.imwrite(path, sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        sheets.append(path)
    return sheets


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="*", default=None, help="video paths or globs")
    ap.add_argument("--weights", default=os.path.join(root, DEFAULT_RAT_WEIGHTS))
    ap.add_argument("--out", default=os.path.join(root, "hard_negatives"))
    ap.add_argument("--keep-every", type=int, default=5,
                    help="thin surviving frames to every Nth (all frames are still scanned)")
    ap.add_argument("--max-per-video", type=int, default=60)
    ap.add_argument("--min-hamming", type=int, default=6,
                    help="perceptual distance required between kept frames (0-64)")
    ap.add_argument("--static-seconds", type=float, default=1.5,
                    help="how long a detection must hold still to count as lifeless")
    ap.add_argument("--rat-free", nargs="*", default=[], metavar="FILE:START-END",
                    help="clip ranges you know contain no rat; harvested without inference")
    args = ap.parse_args()

    # `--videos` with no values means "none" — only `None` (flag absent) takes the default.
    patterns = (args.videos if args.videos is not None
                else [os.path.join(root, "uploads", "*.mp4")])
    videos: List[str] = []
    for pattern in patterns:
        videos.extend(sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[")
                      else [pattern])
    videos = [v for v in videos if os.path.getsize(v) > 10_000]
    if not videos and not args.rat_free:
        print("No usable videos found.")
        return 1

    samples: List[Dict[str, Any]] = []
    engine: Optional[VisionEngine] = None

    for spec in args.rat_free:
        samples.extend(harvest_rat_free(spec, args.keep_every, args.min_hamming))

    if videos:
        engine = VisionEngine(default_weights=args.weights)
        if engine.model is None:
            print("Could not load weights.")
            return 1
        print(f"\nMining with {engine.weights_path} @ imgsz={engine.infer_imgsz}, "
              f"detect conf={VETO_CONF}, static={args.static_seconds}s\n")
        for v in videos:
            samples.extend(scan_video(engine, v, args.keep_every, args.max_per_video,
                                      args.min_hamming, args.static_seconds))

    if not samples:
        print("\nNothing mined — every detection in this footage was moving, "
              "so none of it is safe to file as background.")
        return 0

    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    manifest = []
    for i, s in enumerate(samples):
        name = f"hn_{i:04d}_{os.path.splitext(os.path.basename(s['source']))[0]}".replace(" ", "_")
        name = f"{name}_f{s['frame_index']}.jpg"
        cv2.imwrite(os.path.join(img_dir, name), s["image"],
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        manifest.append({"file": name, "source": s["source"],
                         "frame_index": s["frame_index"], "kind": s["kind"],
                         "false_positives": s["false_positives"]})

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"veto_conf": VETO_CONF,
                   "weights": engine.weights_path if engine else None,
                   "samples": manifest}, fh, indent=2)

    sheets = write_contact_sheets(samples, args.out)
    total_fp = sum(len(s["false_positives"]) for s in samples)

    print(f"\n{len(samples)} hard negatives mined ({total_fp} false-positive boxes)")
    print(f"  images   -> {img_dir}")
    print(f"  manifest -> {os.path.join(args.out, 'manifest.json')}")
    for s in sheets:
        print(f"  review   -> {s}")
    print("\nNEXT: open the contact sheets. Every yellow box must be empty pipe, "
          "not a rat.\n      Delete any frame containing an animal from "
          f"{img_dir}, then run:\n      python tools/apply_hard_negatives.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
