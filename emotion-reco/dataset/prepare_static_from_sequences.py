"""Build a static-image .npz cache from a sequential folder layout.

For each sequence under `<data_root>/<class>/<seq>/`, this takes the LAST
frame (the apex), runs MediaPipe FaceLandmarker on it, and writes
`<output>/<name>_train.npz` + `<output>/<name>_val.npz` with the same schema
the static-image trainers (train_emonet / train_blendshape / train_geo_static)
expect.

Use this when you want to train the static-image methods on a sequential
dataset you already have organized — e.g. CK+ at full resolution from
dataset.organize_ckplus_apex — instead of the HuggingFace `ckplus-dataset`
mirror, which is only 48 x 48 and unusable for FaceMesh.

Output keys mirror dataset.prepare_static:
    landmarks       : (N, 478, 3)  float32   if mode in {landmarks, both}
    blendshapes     : (N, 52)      float32   if mode in {blendshapes, both}
    blendshape_names: (52,)        str       same as prepare_static
    labels          : (N,)         int64     class index into label_names
    label_names     : (C,)         str       original folder names

Usage:
    python -m dataset.prepare_static_from_sequences \\
        --data_root data/ckplus --output cache/ckplus_static --mode both
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .dataset import discover_sequences
from . import prepare_static as _base


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data_root", required=True,
                    help="Sequential folder layout (classes / sequences / frames)")
    ap.add_argument("--output", required=True,
                    help="Destination directory for the train/val .npz files")
    ap.add_argument("--name", default="ckplus",
                    help="Name prefix for the output files (default: ckplus). "
                         "Files are written as <name>_train.npz / <name>_val.npz.")
    ap.add_argument("--mode", default="both",
                    choices=["landmarks", "blendshapes", "both"],
                    help="Feature mode (default: both)")
    ap.add_argument("--val_frac", type=float, default=0.20,
                    help="Validation split fraction (default: 0.20)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=_base.MODEL_PATH,
                    help="Path to face_landmarker.task (auto-downloaded)")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences, labels = discover_sequences(args.data_root)
    label_to_idx = {name: i for i, name in enumerate(labels)}
    print(f"Found {len(sequences)} sequences across {len(labels)} classes: {labels}")

    landmarker = _base.build_landmarker(
        args.model, blendshapes=(args.mode in ("blendshapes", "both"))
    )

    lm_list: List[np.ndarray] = []
    bs_list: List[np.ndarray] = []
    y: List[int] = []
    skipped = 0

    try:
        for seq in tqdm(sequences, desc="Extracting apex landmarks"):
            apex = seq.frames[-1]
            pil = Image.open(apex).convert("RGB")
            if args.mode == "landmarks":
                lm, reason = _base.extract_landmarks(landmarker, pil)
                if lm is None:
                    skipped += 1
                    continue
                lm_list.append(lm)
            elif args.mode == "blendshapes":
                bs, reason = _base.extract_blendshapes(landmarker, pil)
                if bs is None:
                    skipped += 1
                    continue
                bs_list.append(bs)
            else:  # both
                lm, bs, reason = _base.extract_combined(landmarker, pil)
                if lm is None or bs is None:
                    skipped += 1
                    continue
                lm_list.append(lm)
                bs_list.append(bs)
            y.append(label_to_idx[seq.label])
    finally:
        landmarker.close()

    if not y:
        print("No usable sequences — every apex frame failed face detection.",
              file=sys.stderr)
        return 1

    y_arr = np.array(y, dtype=np.int64)
    indices = np.arange(len(y_arr))
    tr_idx, va_idx = train_test_split(
        indices, test_size=args.val_frac, random_state=args.seed,
        stratify=y_arr,
    )

    label_arr = np.array(labels)

    def save(split_idx: np.ndarray, suffix: str) -> Path:
        kwargs = {
            "labels": y_arr[split_idx],
            "label_names": label_arr,
        }
        if lm_list:
            kwargs["landmarks"] = np.stack(lm_list)[split_idx].astype(np.float32)
        if bs_list:
            kwargs["blendshapes"] = np.stack(bs_list)[split_idx].astype(np.float32)
            kwargs["blendshape_names"] = np.array(_base.BLENDSHAPE_NAMES)
        path = out_dir / f"{args.name}_{suffix}.npz"
        np.savez_compressed(path, **kwargs)
        return path

    train_path = save(tr_idx, "train")
    val_path = save(va_idx, "val")

    counts = {labels[i]: int((y_arr == i).sum()) for i in range(len(labels))}
    print(f"Saved {len(tr_idx)} train + {len(va_idx)} val → {out_dir}")
    print(f"  {train_path.name}")
    print(f"  {val_path.name}")
    print(f"Per-class counts: {json.dumps(counts, indent=2)}")
    print(f"Skipped {skipped} sequences (no face detected on apex frame)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
