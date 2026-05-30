"""Sequence loading utilities.

Expected on-disk layout (CK+-style):

    data_root/
    ├── anger/
    │   ├── S005_001/
    │   │   ├── 0001.png
    │   │   ├── 0002.png
    │   │   └── ...
    │   └── S010_002/...
    ├── happiness/
    │   └── ...

Each leaf folder is one sequence and contains the raw frames in chronological
order. From every sequence we uniformly sample `num_keyframes` (default 5)
frames, going from the first frame (neutral) to the last (apex), matching the
paper's setup.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass
class FrameSequence:
    """One expression sequence: an emotion label and its ordered frame paths."""

    label: str
    frames: list[Path]


def discover_sequences(
    data_root: str | os.PathLike,
) -> tuple[list[FrameSequence], list[str]]:
    """Walk the dataset root and return (sequences, sorted class list)."""
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"data_root not found: {root}")

    labels = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not labels:
        raise RuntimeError(f"No class subdirectories under {root}")

    sequences: list[FrameSequence] = []
    for label in labels:
        for seq_dir in sorted((root / label).iterdir()):
            if not seq_dir.is_dir():
                continue
            frames = sorted(
                p for p in seq_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
            )
            if len(frames) >= 2:
                sequences.append(FrameSequence(label=label, frames=frames))
    if not sequences:
        raise RuntimeError(f"No sequences with >=2 frames under {root}")
    return sequences, labels


def sample_keyframes(frames: list[Path], k: int = 5) -> list[Path]:
    """Uniformly sample k frames from neutral to apex (inclusive on both ends)."""
    n = len(frames)
    if n <= k:
        # Pad by repeating the apex (last) frame to reach k.
        return list(frames) + [frames[-1]] * (k - n)
    idx = np.linspace(0, n - 1, num=k).round().astype(int)
    return [frames[i] for i in idx]
