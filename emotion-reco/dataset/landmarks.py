"""MediaPipe FaceMesh wrapper + 61-landmark FACS grouping.

Implements the landmark selection described in Köksal & Gumus 2025, Table 3.
Each MediaPipe frame yields 478 3D landmarks; we keep 61 of them and group them
by facial region. Groups correspond to action-unit categories.

Uses the MediaPipe Tasks API (`FaceLandmarker`). The legacy `mediapipe.solutions`
FaceMesh API was removed in recent MediaPipe builds (0.10.30+), so the Tasks API
is the only one available. It needs a `face_landmarker.task` model file, which
is downloaded automatically on first use.
"""
from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:  # pragma: no cover
    mp = None
    mp_python = None
    mp_vision = None


# The 478-landmark FaceLandmarker model (float16, ~3.8 MB).
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def ensure_model(path: Path = _MODEL_PATH) -> Path:
    """Download the FaceLandmarker model on first use; return its local path."""
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading FaceLandmarker model → {path}")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(_MODEL_URL, tmp)
    tmp.rename(path)
    return path


# ---------------------------------------------------------------------------
# Landmark presets. The paper runs experiments with 61, 122 and 250 landmarks
# selected manually from the 478 FaceMesh points. Each preset is a dict of the
# seven facial regions; the five FACS categories (cat1..cat5) are derived from
# those regions via `_facs_groups`.
# ---------------------------------------------------------------------------

# 61-landmark preset — curated, matches the paper's best speed/accuracy point.
_REGIONS_61: Dict[str, List[int]] = {
    "left_eye": [33, 133, 159, 145, 158, 153, 160, 144],
    "right_eye": [263, 362, 386, 374, 385, 380, 387, 373],
    "left_eyebrow": [70, 63, 105, 66, 107, 55, 65],
    "right_eyebrow": [336, 296, 334, 293, 300, 285, 295],
    "nose": [1, 2, 98, 327, 168, 6, 197, 195, 5, 4],
    "mouth": [78, 308, 13, 14, 82, 312, 87, 317, 95, 324,
              88, 318, 81, 311, 80, 310],
    "lower_jaw": [152, 175, 199, 200, 18],
}

# Comprehensive preset (~253 landmarks) — dense FaceMesh coverage of every
# region. The 250-landmark preset uses this in full; 122 evenly subsamples it.
_REGIONS_FULL: Dict[str, List[int]] = {
    "left_eye": [
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
        130, 247, 30, 29, 27, 28, 56, 190,
        226, 113, 225, 224, 223, 222, 221, 189,
        31, 228, 229, 230, 231, 232, 233, 244,
        25, 110, 24, 23, 22, 26, 112, 243,
    ],
    "right_eye": [
        263, 362, 382, 381, 380, 374, 373, 390, 249, 466, 388, 387, 386, 385, 384, 398,
        359, 467, 260, 259, 257, 258, 286, 414,
        446, 342, 445, 444, 443, 442, 441, 413,
        261, 448, 449, 450, 451, 452, 453, 464,
        255, 339, 254, 253, 252, 256, 341, 463,
    ],
    "left_eyebrow": [70, 63, 105, 66, 107, 46, 53, 52, 65, 55],
    "right_eyebrow": [300, 293, 334, 296, 336, 276, 283, 282, 295, 285],
    "nose": [
        1, 2, 4, 5, 6, 19, 20, 94, 97, 98, 99, 125, 141, 164, 168, 195, 197, 326, 327, 328,
        358, 48, 49, 64, 102, 131, 134, 220, 218, 219, 278, 279, 294, 331, 360, 363, 440, 438, 439, 115,
    ],
    "mouth": [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        76, 77, 90, 180, 85, 16, 315, 404, 320, 307, 306, 292, 408, 304, 272, 271, 268, 12, 38, 41,
        42, 183, 73, 74, 184,
    ],
    "lower_jaw": [
        152, 175, 199, 200, 18, 83, 313, 172, 136, 150, 149, 176, 148, 377, 400, 378, 379, 365, 397,
        288, 361, 323, 132, 58, 207, 206, 427, 411, 216, 436, 212, 432,
    ],
}

PRESETS = (61, 122, 250)


def _facs_groups(regions: Dict[str, List[int]]) -> Dict[str, List[int]]:
    """Build the 5 FACS categories (paper Table 3) from the 7 facial regions."""
    le, re = regions["left_eye"], regions["right_eye"]
    lb, rb = regions["left_eyebrow"], regions["right_eyebrow"]
    nose, mouth, jaw = regions["nose"], regions["mouth"], regions["lower_jaw"]
    return {
        "cat1": le + lb + re + rb,             # AU 1,2,3,4,5
        "cat2": le + re + nose,                # AU 6
        "cat3": le + lb + re + rb + nose,      # AU 7,9
        "cat4": nose + mouth + jaw,            # AU 12,14,15,16,23,26
        "cat5": le + re + nose + mouth,        # AU 20
    }


def _subsample_regions(regions: Dict[str, List[int]], target: int) -> Dict[str, List[int]]:
    """Evenly thin each region so the deduped landmark total is near `target`."""
    total = len({i for v in regions.values() for i in v})
    if target >= total:
        return {k: list(v) for k, v in regions.items()}
    scale = target / total
    out: Dict[str, List[int]] = {}
    for key, members in regions.items():
        keep = max(2, round(len(members) * scale))   # >=2 so the region forms pairs
        if keep >= len(members):
            out[key] = list(members)
        else:
            picks = np.linspace(0, len(members) - 1, keep).round().astype(int)
            out[key] = [members[i] for i in sorted(set(picks.tolist()))]
    return out


def get_landmark_config(preset: int = 61) -> tuple[List[int], Dict[str, List[int]]]:
    """Return (selected landmark indices, FACS groups) for a landmark preset.

    preset 61  → curated 61-landmark set.
    preset 122 → comprehensive set evenly thinned to ~122 landmarks.
    preset 250 → full comprehensive set (~253 landmarks).
    """
    if preset == 61:
        regions = _REGIONS_61
    elif preset == 122:
        regions = _subsample_regions(_REGIONS_FULL, 122)
    elif preset == 250:
        regions = _REGIONS_FULL
    else:
        raise ValueError(f"landmark preset must be one of {PRESETS}, got {preset}")
    groups = _facs_groups(regions)
    selected: List[int] = []
    seen = set()
    for members in regions.values():
        for idx in members:
            if idx not in seen:
                seen.add(idx)
                selected.append(idx)
    return selected, groups


# Module-level defaults = 61-landmark preset (keeps existing imports working).
SELECTED_LANDMARKS, LANDMARK_GROUPS = get_landmark_config(61)
assert len(SELECTED_LANDMARKS) == 61


@dataclass
class FaceMeshDetector:
    """Wrapper around the MediaPipe Tasks `FaceLandmarker`.

    static_image_mode=True  → RunningMode.IMAGE  (independent per-image detect,
                              used for offline dataset preprocessing).
    static_image_mode=False → RunningMode.VIDEO  (uses internal tracking across
                              frames, used for the real-time webcam demo).

    output_blendshapes=True enables the 52 MediaPipe blendshape coefficients
    so detect_blendshapes() returns them — needed by the blendshape predictor.
    """

    static_image_mode: bool = True
    max_num_faces: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    output_blendshapes: bool = False
    model_path: Optional[str] = None

    def __post_init__(self) -> None:
        if mp is None:
            raise ImportError("mediapipe is required. Install with `pip install mediapipe`.")
        model = Path(self.model_path) if self.model_path else ensure_model()
        self._video = not self.static_image_mode
        self._last_ts = -1
        running_mode = (
            mp_vision.RunningMode.IMAGE if self.static_image_mode
            else mp_vision.RunningMode.VIDEO
        )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model)),
            running_mode=running_mode,
            num_faces=self.max_num_faces,
            min_face_detection_confidence=self.min_detection_confidence,
            min_face_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_face_blendshapes=self.output_blendshapes,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def _detect_raw(self, image_rgb: np.ndarray):
        """Single MediaPipe pass over `image_rgb`; returns the raw result.

        Shared by detect / detect_selected / detect_blendshapes so a single
        webcam frame triggers at most one MediaPipe call.
        """
        rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self._video:
            # VIDEO mode needs strictly-increasing millisecond timestamps.
            ts = max(self._last_ts + 1, int(time.monotonic() * 1000))
            self._last_ts = ts
            return self._landmarker.detect_for_video(mp_image, ts)
        return self._landmarker.detect(mp_image)

    def detect(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Return (478, 3) array of normalized landmarks, or None if no face."""
        result = self._detect_raw(image_rgb)
        if not result.face_landmarks:
            return None
        face = result.face_landmarks[0]
        return np.array([(lm.x, lm.y, lm.z) for lm in face], dtype=np.float32)

    def detect_selected(
        self, image_rgb: np.ndarray, selected: Optional[List[int]] = None
    ) -> Optional[np.ndarray]:
        """Return the selected landmarks as (L, 2) in normalized coords.

        `selected` defaults to the 61-landmark preset; pass a different index
        list (from `get_landmark_config`) for the 122/250 presets.
        """
        pts = self.detect(image_rgb)
        if pts is None:
            return None
        idx = SELECTED_LANDMARKS if selected is None else selected
        return pts[idx, :2]

    def detect_blendshapes(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Return (52,) MediaPipe blendshape coefficients, or None if no face.

        Requires `output_blendshapes=True` at construction time.
        """
        if not self.output_blendshapes:
            raise RuntimeError(
                "detect_blendshapes called on a detector built without "
                "output_blendshapes=True; rebuild the detector with that flag."
            )
        result = self._detect_raw(image_rgb)
        if not result.face_blendshapes:
            return None
        bs = result.face_blendshapes[0]
        return np.array([c.score for c in bs], dtype=np.float32)

    def close(self) -> None:
        self._landmarker.close()


def landmarks_to_pixels(landmarks_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert normalized landmarks (range 0-1) to pixel coordinates."""
    out = landmarks_xy.copy()
    out[..., 0] *= width
    out[..., 1] *= height
    return out
