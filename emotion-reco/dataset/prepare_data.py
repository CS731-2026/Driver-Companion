"""Offline feature extraction.

Reads a CK+-style folder tree, runs MediaPipe FaceMesh on 5 key frames per
sequence, builds the (N-1=4, A) geometric-feature tensor, and writes a single
.npz cache with arrays:

    X: (num_sequences, 4, A)   float32
    y: (num_sequences,)        int64    class indices
    labels: (num_classes,)     str       sorted class names
    pair_idx: (P, 2)           int32     landmark-pair indices (debug/repro)

Sequences where FaceMesh fails on any of the 5 frames are skipped (with a
warning) to avoid feeding NaNs to the network.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from .dataset import discover_sequences, sample_keyframes
from .features import build_pair_indices, sequence_features
from .landmarks import PRESETS, FaceMeshDetector, get_landmark_config


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", required=True, help="Root with one folder per class")
    ap.add_argument("--output", required=True, help="Destination .npz file")
    ap.add_argument("--num_keyframes", type=int, default=5)
    ap.add_argument("--landmarks", type=int, default=61, choices=PRESETS,
                    help="Landmark preset: 61, 122 or 250 (paper experiments)")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    selected, groups = get_landmark_config(args.landmarks)
    pair_idx = build_pair_indices(groups, selected)

    sequences, labels = discover_sequences(args.data_root)
    label_to_idx = {name: i for i, name in enumerate(labels)}
    print(f"Found {len(sequences)} sequences across {len(labels)} classes: {labels}")
    print(f"Landmark preset: {args.landmarks}  →  {len(selected)} landmarks, "
          f"{len(pair_idx)} pairs, feature dim A = {2 * len(pair_idx)}")

    detector = FaceMeshDetector(static_image_mode=True)

    X: List[np.ndarray] = []
    y: List[int] = []
    skipped = 0

    try:
        for seq in tqdm(sequences, desc="Extracting features"):
            keyframes = sample_keyframes(seq.frames, k=args.num_keyframes)
            seq_landmarks = []
            failed = False
            for frame_path in keyframes:
                rgb = load_rgb(frame_path)
                pts = detector.detect_selected(rgb, selected=selected)
                if pts is None:
                    failed = True
                    break
                # Use pixel-space coords so feature magnitudes scale with face size;
                # the StandardScaler later normalizes per-feature mean/var.
                h, w = rgb.shape[:2]
                pts_px = pts.copy()
                pts_px[:, 0] *= w
                pts_px[:, 1] *= h
                seq_landmarks.append(pts_px)
            if failed:
                skipped += 1
                continue
            feats = sequence_features(np.stack(seq_landmarks), pair_idx)
            X.append(feats)
            y.append(label_to_idx[seq.label])
    finally:
        detector.close()

    if not X:
        print("No usable sequences extracted.", file=sys.stderr)
        return 1

    X_arr = np.stack(X).astype(np.float32)
    y_arr = np.array(y, dtype=np.int64)
    np.savez_compressed(
        out_path,
        X=X_arr,
        y=y_arr,
        labels=np.array(labels),
        pair_idx=pair_idx,
        selected_landmarks=np.array(selected, dtype=np.int32),
    )
    print(f"Saved {X_arr.shape[0]} sequences → {out_path}")
    print(f"X shape: {X_arr.shape}  y shape: {y_arr.shape}  (skipped {skipped})")
    # Per-class distribution
    counts = {labels[i]: int((y_arr == i).sum()) for i in range(len(labels))}
    print("Per-class counts:", json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
