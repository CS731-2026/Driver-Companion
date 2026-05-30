"""Offline feature extraction for the Juárez-Jiménez et al. 2025 baseline.

Reimplements the feature pipeline of "Facial Landmark Visualization and Emotion
Recognition Through Neural Networks" (arXiv:2506.17191). No official code was
released, so this is a from-scratch port kept as a baseline for comparison
against the ConvLSTM1D model.

The paper used Dlib's 68-point detector; this project is built on MediaPipe
FaceMesh, so we reuse the existing detector and the project's landmark presets.
Keeping the same landmarks means a baseline run is directly comparable to a
ConvLSTM1D run — the only thing that differs is the method.

For each sequence:
  1. Detect 478 landmarks on the neutral (first) and peak (last) frame.
  2. Normalize each frame: translate so the eye-midpoint is the origin, then
     divide by the landmark coordinate range (Min-Max scale). Centering removes
     face position; the scale removes face size.
  3. Keep the selected landmark preset.
  4. Build two feature vectors:
       displacement = peak - neutral      (paper's headline feature)
       absolute     = peak coordinates    (paper's comparison feature)
  5. Optional IQR outlier clipping per feature column (paper Section 3.3).

Output .npz arrays:
    X_disp:  (num_sequences, 2L)  float32   displacement features
    X_abs:   (num_sequences, 2L)  float32   absolute-position features
    y:       (num_sequences,)     int64     class indices
    labels:  (num_classes,)       str       sorted class names
    selected_landmarks: (L,)      int32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from .dataset import discover_sequences
from .landmarks import PRESETS, FaceMeshDetector, get_landmark_config
from .prepare_data import load_rgb

# MediaPipe FaceMesh eye-corner indices, used to find the midpoint between the
# eyes (the paper's normalization origin). Independent of the landmark preset.
_LEFT_EYE = [33, 133]
_RIGHT_EYE = [263, 362]


def normalize_frame(pts_xy: np.ndarray) -> np.ndarray:
    """Center landmarks on the eye-midpoint, then scale to unit face size.

    pts_xy: (478, 2) normalized FaceMesh coordinates for one frame.
    A single scalar scale (the larger axis range) is used so the face aspect
    ratio is preserved.
    """
    mid = 0.5 * (pts_xy[_LEFT_EYE].mean(0) + pts_xy[_RIGHT_EYE].mean(0))
    centered = pts_xy - mid
    scale = float((centered.max(0) - centered.min(0)).max())
    return centered / (scale if scale > 0 else 1.0)


def iqr_clip(X: np.ndarray) -> np.ndarray:
    """Clip each feature column to the Tukey whiskers [Q1-1.5*IQR, Q3+1.5*IQR].

    The paper's Section 3.3 outlier handling. Applied dataset-wide to match the
    paper's global quartile analysis.
    """
    q1, q3 = np.percentile(X, 25, axis=0), np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    return np.clip(X, q1 - 1.5 * iqr, q3 + 1.5 * iqr)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data_root", required=True, help="Root with one folder per class")
    ap.add_argument("--output", required=True, help="Destination .npz file")
    ap.add_argument("--landmarks", type=int, default=61, choices=PRESETS,
                    help="Landmark preset (keep consistent with the ConvLSTM model)")
    ap.add_argument("--no_clip", action="store_true",
                    help="Disable the IQR outlier clipping (paper Section 3.3)")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    selected, _ = get_landmark_config(args.landmarks)
    sequences, labels = discover_sequences(args.data_root)
    label_to_idx = {name: i for i, name in enumerate(labels)}
    print(f"Found {len(sequences)} sequences across {len(labels)} classes: {labels}")
    print(f"Landmark preset: {args.landmarks} -> {len(selected)} landmarks, "
          f"feature dim = {2 * len(selected)}")

    detector = FaceMeshDetector(static_image_mode=True)
    disp: List[np.ndarray] = []
    absolute: List[np.ndarray] = []
    y: List[int] = []
    skipped = 0

    try:
        for seq in tqdm(sequences, desc="Extracting features"):
            frames = []
            failed = False
            # Paper: select the first frame (neutral) and the last (peak).
            for frame_path in (seq.frames[0], seq.frames[-1]):
                pts = detector.detect(load_rgb(frame_path))
                if pts is None:
                    failed = True
                    break
                frames.append(normalize_frame(pts[:, :2])[selected])
            if failed:
                skipped += 1
                continue
            neutral_lm, peak_lm = frames
            disp.append((peak_lm - neutral_lm).ravel())
            absolute.append(peak_lm.ravel())
            y.append(label_to_idx[seq.label])
    finally:
        detector.close()

    if not disp:
        print("No usable sequences extracted.", file=sys.stderr)
        return 1

    X_disp = np.stack(disp).astype(np.float32)
    X_abs = np.stack(absolute).astype(np.float32)
    if not args.no_clip:
        X_disp = iqr_clip(X_disp).astype(np.float32)
        X_abs = iqr_clip(X_abs).astype(np.float32)
    y_arr = np.array(y, dtype=np.int64)

    np.savez_compressed(
        out_path,
        X_disp=X_disp,
        X_abs=X_abs,
        y=y_arr,
        labels=np.array(labels),
        selected_landmarks=np.array(selected, dtype=np.int32),
    )
    print(f"Saved {X_disp.shape[0]} sequences -> {out_path}")
    print(f"X_disp {X_disp.shape}  X_abs {X_abs.shape}  (skipped {skipped})")
    counts = {labels[i]: int((y_arr == i).sum()) for i in range(len(labels))}
    print("Per-class counts:", json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
