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


# MediaPipe FaceMesh canonical indices grouped by facial region.
# Indices below are the standard FaceMesh contour/landmark indices used widely
# in the community; they cover the muscle locations referenced by the AU table
# in the paper (eye, eyebrow, nose, mouth, lower jaw).
LEFT_EYE = [33, 133, 159, 145, 158, 153, 160, 144]          # 8
RIGHT_EYE = [263, 362, 386, 374, 385, 380, 387, 373]        # 8
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65]               # 7
RIGHT_EYEBROW = [336, 296, 334, 293, 300, 285, 295]         # 7
NOSE = [1, 2, 98, 327, 168, 6, 197, 195, 5, 4]              # 10
MOUTH = [78, 308, 13, 14, 82, 312, 87, 317, 95, 324,
         88, 318, 81, 311, 80, 310]                          # 16
LOWER_JAW = [152, 175, 199, 200, 18]                        # 5

# Total = 61, matching the "61 landmarks with AU grouping" experiment that the
# paper reports as the best speed/accuracy tradeoff.
SELECTED_LANDMARKS: List[int] = (
    LEFT_EYE + RIGHT_EYE + LEFT_EYEBROW + RIGHT_EYEBROW
    + NOSE + MOUTH + LOWER_JAW
)
assert len(SELECTED_LANDMARKS) == 61

# Five FACS-derived categories. Each value is the list of FaceMesh indices that
# belong to that category. Only landmark pairs from the same group are kept,
# which reduces the C(61, 2)=1830 default pair count and acts as feature
# selection.
LANDMARK_GROUPS: Dict[str, List[int]] = {
    "cat1": LEFT_EYE + LEFT_EYEBROW + RIGHT_EYE + RIGHT_EYEBROW,      # AU 1,2,3,4,5
    "cat2": LEFT_EYE + RIGHT_EYE + NOSE,                              # AU 6
    "cat3": LEFT_EYE + LEFT_EYEBROW + RIGHT_EYE + RIGHT_EYEBROW + NOSE,  # AU 7,9
    "cat4": NOSE + MOUTH + LOWER_JAW,                                 # AU 12,14,15,16,23,26
    "cat5": LEFT_EYE + RIGHT_EYE + NOSE + MOUTH,                      # AU 20
}


@dataclass
class FaceMeshDetector:
    """Wrapper around the MediaPipe Tasks `FaceLandmarker`.

    static_image_mode=True  → RunningMode.IMAGE  (independent per-image detect,
                              used for offline dataset preprocessing).
    static_image_mode=False → RunningMode.VIDEO  (uses internal tracking across
                              frames, used for the real-time webcam demo).
    """

    static_image_mode: bool = True
    max_num_faces: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
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
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def detect(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Return (478, 3) array of normalized landmarks, or None if no face."""
        rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self._video:
            # VIDEO mode needs strictly-increasing millisecond timestamps.
            ts = max(self._last_ts + 1, int(time.monotonic() * 1000))
            self._last_ts = ts
            result = self._landmarker.detect_for_video(mp_image, ts)
        else:
            result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        face = result.face_landmarks[0]
        return np.array([(lm.x, lm.y, lm.z) for lm in face], dtype=np.float32)

    def detect_selected(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Return only the 61 selected landmarks as (61, 2) in normalized coords."""
        pts = self.detect(image_rgb)
        if pts is None:
            return None
        return pts[SELECTED_LANDMARKS, :2]

    def close(self) -> None:
        self._landmarker.close()


def landmarks_to_pixels(landmarks_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert normalized landmarks (range 0-1) to pixel coordinates."""
    out = landmarks_xy.copy()
    out[..., 0] *= width
    out[..., 1] *= height
    return out
