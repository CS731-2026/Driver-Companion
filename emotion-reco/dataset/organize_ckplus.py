"""Reorganize an extracted CK+ archive into the pipeline's layout.

The official CK+ release (granted on request — *not* downloaded here) unpacks
into roughly this shape:

    ckplus_root/
    ├── cohn-kanade-images/
    │   └── S005/001/S005_001_00000001.png ...
    └── Emotion/
        └── S005/001/S005_001_00000011_emotion.txt   # one integer 0..7

Only sequences that have a corresponding `*_emotion.txt` file are labelled in
CK+ (about 327 out of 593 sequences). This script walks the label tree, copies
or symlinks each labelled sequence into the CK+-style layout that
`dataset.prepare_data` expects, mapping the integer code to its emotion name.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from tqdm import tqdm


CKPLUS_EMOTIONS = {
    0: "neutral",
    1: "anger",
    2: "contempt",
    3: "disgust",
    4: "fear",
    5: "happy",
    6: "sadness",
    7: "surprise",
}


def find_label(label_dir: Path) -> int | None:
    """Return the integer emotion code stored in <subject>/<seq>/*_emotion.txt, or None."""
    for txt in label_dir.glob("*_emotion.txt"):
        try:
            value = txt.read_text().strip()
            if not value:
                continue
            return int(float(value))
        except ValueError:
            return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckplus_root", required=True,
                    help="Directory containing 'cohn-kanade-images/' and 'Emotion/'")
    ap.add_argument("--out_root", required=True, help="Destination CK+-style layout")
    ap.add_argument("--mode", choices=["copy", "symlink"], default="symlink",
                    help="How to materialize the frames (symlink is fast, copy is portable)")
    ap.add_argument("--include_neutral", action="store_true",
                    help="Keep the 'neutral' (code 0) sequences (excluded by default — "
                         "CK+ sequences always start neutral so labelling the whole sequence "
                         "neutral leaks the apex into the wrong class)")
    args = ap.parse_args()

    root = Path(args.ckplus_root)
    images_root = root / "cohn-kanade-images"
    labels_root = root / "Emotion"
    if not images_root.is_dir() or not labels_root.is_dir():
        print(f"Expected 'cohn-kanade-images/' and 'Emotion/' under {root}", file=sys.stderr)
        return 1

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Walk the label tree first so we only process sequences that have a label.
    sequences = []
    for subj_dir in sorted(labels_root.iterdir()):
        if not subj_dir.is_dir():
            continue
        for seq_dir in sorted(subj_dir.iterdir()):
            code = find_label(seq_dir)
            if code is None:
                continue
            sequences.append((subj_dir.name, seq_dir.name, code))

    if not sequences:
        print(f"No labelled sequences under {labels_root}", file=sys.stderr)
        return 1

    per_class: dict[str, int] = {}
    skipped_unlabeled = 0
    skipped_missing_frames = 0

    for subject, seq_id, code in tqdm(sequences, desc="Organizing"):
        if code == 0 and not args.include_neutral:
            skipped_unlabeled += 1
            continue
        emotion = CKPLUS_EMOTIONS[code]
        src_dir = images_root / subject / seq_id
        frames = sorted(src_dir.glob("*.png")) + sorted(src_dir.glob("*.jpg"))
        if not frames:
            skipped_missing_frames += 1
            continue
        dst_dir = out_root / emotion / f"{subject}_{seq_id}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in frames:
            dst = dst_dir / f.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if args.mode == "symlink":
                dst.symlink_to(f.resolve())
            else:
                shutil.copy2(f, dst)
        per_class[emotion] = per_class.get(emotion, 0) + 1

    print("\nPer-class sequence counts:")
    for name in sorted(per_class):
        print(f"  {name:10s} {per_class[name]}")
    print(f"\nTotal: {sum(per_class.values())} sequences under {out_root}")
    if skipped_unlabeled:
        print(f"Skipped {skipped_unlabeled} 'neutral' (code 0) sequences. "
              "Use --include_neutral to keep them.")
    if skipped_missing_frames:
        print(f"Skipped {skipped_missing_frames} sequences with no frames on disk.")
    print(f"\nNext:  python -m dataset.prepare_data --data_root {out_root} "
          f"--output cache/ckplus.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
