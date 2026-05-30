"""Geometric feature creation.

Per the paper, for each landmark pair inside the same FACS group we compute:
    distance d  = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)             (eq. 1)
    angle    θ  = arctan((x_i - x_j) / (y_i - y_j))                (eq. 2)

For a sequence of N frames we then take frame-to-frame differences:
    f_t = features(frame_t) - features(frame_{t-1})
producing an (N-1, A) tensor where A = 2 * (#unique pairs across all groups).
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Sequence, Tuple

import numpy as np

from .landmarks import LANDMARK_GROUPS, SELECTED_LANDMARKS


def build_pair_indices(
    groups: Dict[str, Sequence[int]] = LANDMARK_GROUPS,
    selected: Sequence[int] = SELECTED_LANDMARKS,
) -> np.ndarray:
    """Return unique (i, j) pairs in the local index space of `selected`.

    The paper's eq. for pair count is:
        C(n, 2) > Unique( C(a,2) + C(b,2) + C(c,2) + C(d,2) + C(e,2) )
    i.e. pairs are taken within each group, then deduplicated across groups.
    Pair indices returned here are positions inside `selected`, not raw
    FaceMesh indices, so downstream code can index a (61, 2) landmark array
    directly.
    """
    fm_to_local = {fm_idx: local for local, fm_idx in enumerate(selected)}
    pairs: set[Tuple[int, int]] = set()
    for members in groups.values():
        local = [fm_to_local[i] for i in members if i in fm_to_local]
        for a, b in combinations(sorted(set(local)), 2):
            pairs.add((a, b))
    return np.array(sorted(pairs), dtype=np.int32)


def compute_frame_features(landmarks_xy: np.ndarray, pair_idx: np.ndarray) -> np.ndarray:
    """Distance + angle features for one frame.

    Args:
        landmarks_xy: (L, 2) selected landmarks (pixel or normalized — either is
            fine since they get standard-scaled later, but stay consistent
            across frames).
        pair_idx: (P, 2) int array of landmark-pair indices.

    Returns:
        (2 * P,) feature vector: [distances..., angles...].
    """
    p_i = landmarks_xy[pair_idx[:, 0]]
    p_j = landmarks_xy[pair_idx[:, 1]]
    dx = p_i[:, 0] - p_j[:, 0]
    dy = p_i[:, 1] - p_j[:, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    # Paper eq. 2 uses arctan(dx/dy). atan2(dx, dy) is the numerically stable
    # equivalent and matches the same quadrant convention.
    angle = np.arctan2(dx, dy)
    return np.concatenate([dist, angle]).astype(np.float32)


def sequence_features(
    landmark_seq: np.ndarray, pair_idx: np.ndarray
) -> np.ndarray:
    """Frame-difference feature tensor for one sequence.

    Args:
        landmark_seq: (N, L, 2) landmarks for N frames.
        pair_idx: (P, 2) pair index array.

    Returns:
        (N - 1, 2 * P) array — paper's "Merged Features #0..#(N-2)".
    """
    if landmark_seq.ndim != 3 or landmark_seq.shape[0] < 2:
        raise ValueError(f"Need (N>=2, L, 2) landmark sequence, got {landmark_seq.shape}")
    per_frame = np.stack(
        [compute_frame_features(landmark_seq[t], pair_idx) for t in range(landmark_seq.shape[0])]
    )
    return per_frame[1:] - per_frame[:-1]


def feature_dim(pair_idx: np.ndarray) -> int:
    """A = 2 * P, the feature count per frame."""
    return 2 * len(pair_idx)
