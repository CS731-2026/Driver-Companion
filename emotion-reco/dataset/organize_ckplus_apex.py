"""Adapter for the flattened (apex-only) CK+ layout.

Some CK+ redistributions keep only two frames per sequence instead of the full
neutral→apex video:

    ck_root/
    ├── anger/      S010_004_00000019.png ...   # one APEX frame per sequence
    ├── disgust/    ...
    ├── surprise/   ...
    └── neutral/    S010_004_00000001.png ...   # the NEUTRAL frame of every seq

This script pairs each apex image with its neutral frame (matched by the
`S<subject>_<sequence>` filename prefix) and writes the per-sequence folder
layout `prepare_data.py` expects, with two frames each:

    out_root/<emotion>/<subject_seq>/0001.png   # neutral
                                    /0002.png   # apex

Then run:
    python -m dataset.prepare_data --data_root out_root \
        --output cache/ckplus.npz --num_keyframes 2

Note: with only 2 frames the model sees a single apex-minus-neutral feature
vector — the paper's intermediate onset/offset dynamics are not available in
this reduced distribution.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Emotion folders to look for (CK+ peak-frame classes).
EMOTIONS = ["anger", "contempt", "disgust", "fear", "happiness", "sadness", "surprise"]


def seq_prefix(filename: str) -> str:
    """`S010_004_00000019.png` → `S010_004` (subject + sequence id)."""
    return "_".join(filename.split("_")[:2])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ck_root", required=True,
                    help="Flattened CK+ dir (emotion folders + a 'neutral' folder)")
    ap.add_argument("--out_root", required=True, help="Destination sequence layout")
    ap.add_argument("--mode", choices=["symlink", "copy"], default="symlink",
                    help="symlink is fast; copy is portable")
    args = ap.parse_args()

    ck_root = Path(args.ck_root)
    neutral_dir = ck_root / "neutral"
    if not neutral_dir.is_dir():
        print(f"No 'neutral' folder under {ck_root}", file=sys.stderr)
        return 1

    # Map each sequence prefix → its neutral frame.
    neutral_by_prefix = {
        seq_prefix(p.name): p for p in sorted(neutral_dir.glob("*.png"))
    }
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    def place(src: Path, dst: Path) -> None:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if args.mode == "symlink":
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)

    per_class: dict[str, int] = {}
    missing_neutral = 0

    for emotion in EMOTIONS:
        emo_dir = ck_root / emotion
        if not emo_dir.is_dir():
            print(f"  (skip) no '{emotion}' folder", file=sys.stderr)
            continue
        for apex in sorted(emo_dir.glob("*.png")):
            prefix = seq_prefix(apex.name)
            neutral = neutral_by_prefix.get(prefix)
            if neutral is None:
                missing_neutral += 1
                continue
            seq_dir = out_root / emotion / prefix
            seq_dir.mkdir(parents=True, exist_ok=True)
            place(neutral, seq_dir / "0001.png")   # neutral frame
            place(apex, seq_dir / "0002.png")      # apex frame
            per_class[emotion] = per_class.get(emotion, 0) + 1

    if not per_class:
        print("No sequences written — check --ck_root.", file=sys.stderr)
        return 1

    print("Per-class sequence counts:")
    for name in sorted(per_class):
        print(f"  {name:10s} {per_class[name]}")
    print(f"\nTotal: {sum(per_class.values())} sequences under {out_root}")
    if missing_neutral:
        print(f"Skipped {missing_neutral} apex images with no matching neutral frame.")
    print(f"\nNext:  python -m dataset.prepare_data --data_root {out_root} "
          f"--output cache/ckplus.npz --num_keyframes 2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
