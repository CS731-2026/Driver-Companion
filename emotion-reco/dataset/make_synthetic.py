"""Synthesise a fake feature cache for end-to-end pipeline testing.

Writes a .npz file in the same format as `dataset.prepare_data` — so
`training.train` can be invoked on it without needing a real dataset or
MediaPipe FaceMesh. No real faces are involved; this only verifies that
shapes, scaler, model, and training loop all wire up correctly.

How the fake data is built
--------------------------
* A "neutral" landmark template is sampled once from a Gaussian centred at the
  position of the 61 selected FaceMesh indices on the canonical face mesh —
  here approximated as random points in [0, 1)^2 (we don't need anatomical
  realism, only consistent per-class motion).
* For each class we pick a fixed displacement vector per landmark (its
  "expression direction"). A sequence's frames interpolate the neutral toward
  this target with a per-sequence amplitude + noise; this gives each class a
  reproducible motion signature the ConvLSTM1D can latch onto.
* Frames are converted to features via the real `sequence_features` function
  so the resulting cache is bit-for-bit compatible with the training script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .features import build_pair_indices, sequence_features
from .landmarks import SELECTED_LANDMARKS


DEFAULT_LABELS = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]


def make_dataset(
    labels: list[str],
    sequences_per_class: int,
    num_keyframes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    L = len(SELECTED_LANDMARKS)
    pair_idx = build_pair_indices()

    # One fixed "neutral" template and one fixed motion direction per class.
    neutral = rng.uniform(0, 1, size=(L, 2)).astype(np.float32)
    class_dirs = rng.normal(0, 0.05, size=(len(labels), L, 2)).astype(np.float32)

    X = []
    y = []
    for cls_idx, _ in enumerate(labels):
        for _ in range(sequences_per_class):
            amp = float(rng.uniform(0.5, 1.5))
            # Per-sequence noise so the network can't just memorise positions.
            noise = rng.normal(0, 0.005, size=(num_keyframes, L, 2)).astype(np.float32)
            t = np.linspace(0.0, 1.0, num_keyframes, dtype=np.float32)
            # Each frame interpolates neutral → neutral + amp * direction.
            frames = (
                neutral[None]
                + t[:, None, None] * (amp * class_dirs[cls_idx])[None]
                + noise
            )
            X.append(sequence_features(frames, pair_idx))
            y.append(cls_idx)
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, help="Destination .npz")
    ap.add_argument("--sequences_per_class", type=int, default=60)
    ap.add_argument("--num_keyframes", type=int, default=5)
    ap.add_argument("--labels", default=",".join(DEFAULT_LABELS),
                    help="Comma-separated class names")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    X, y = make_dataset(labels, args.sequences_per_class, args.num_keyframes, args.seed)
    pair_idx = build_pair_indices()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        labels=np.array(labels),
        pair_idx=pair_idx,
    )
    print(f"Wrote {out_path}  X={X.shape}  y={y.shape}  labels={labels}")
    print(f"Per-class counts: "
          f"{json.dumps({l: int((y == i).sum()) for i, l in enumerate(labels)})}")
    print(f"\nNext:  python -m training.train --features {out_path} --out runs/synthetic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
