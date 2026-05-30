"""
convert_to_facemesh_dataset.py
================================
Downloads AffectnetHQ and RAF-DB-7emotions from HuggingFace,
runs MediaPipe FaceLandmarker on every image, and saves a clean
FaceMesh dataset as .npz files (one per split).

Three feature modes (select with --mode):
  landmarks  (default) — raw (x,y,z) coords, shape (N, 478, 3)
  blendshapes          — 52 named blendshape coefficients, shape (N, 52)
                         as used in:
                         Jakhete & Kulkarni, "A Comprehensive Survey and
                         Evaluation of MediaPipe Face Mesh for Human Emotion
                         Recognition", ICCUBEA 2024.
  both                 — landmarks AND blendshapes in the same .npz, single
                         MediaPipe pass per image. Use this when you want one
                         dataset that feeds every downstream training script.

Output layout
-------------
./facemesh_dataset/
    <ds>_<split>.npz  ->  landmarks mode : {landmarks, labels, label_names}
                          blendshapes mode: {blendshapes, blendshape_names,
                                             labels, label_names}
                          both mode      : {landmarks, blendshapes,
                                             blendshape_names, labels,
                                             label_names}
    metadata.json     ->  counts, label map, cleaning stats

Requirements
------------
    pip install datasets mediapipe opencv-python Pillow tqdm
    # Download the MediaPipe face model first:
    wget -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
         -O face_landmarker.task

Usage
-----
    python convert_to_facemesh_dataset.py                          # landmarks (default)
    python convert_to_facemesh_dataset.py --mode blendshapes       # blendshape coefficients
    python convert_to_facemesh_dataset.py --mode both              # both features in one .npz
    python convert_to_facemesh_dataset.py --datasets affectnet     # one dataset only
    python convert_to_facemesh_dataset.py --output ./my_output     # custom output dir
    python convert_to_facemesh_dataset.py --preview                # show live preview window
    python convert_to_facemesh_dataset.py --preview --preview-scale 1.5
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# ── MediaPipe ───────────────────────────────────────────────────────────────
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── HuggingFace datasets ─────────────────────────────────────────────────────
from datasets import load_dataset, load_from_disk

# ── Project-wide model downloader (shared with dataset/landmarks.py) ────────
from .landmarks import ensure_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATASETS_CONFIG = {
    "affectnet": {
        "repo_id": "Piro17/affectnethq",
        "local_path": "./datasets/AffectnetHQ/",
        # HuggingFace column names (adjust if the dataset schema differs)
        "image_col": "image",
        "label_col": "label",
    },
    "rafdb": {
        "repo_id": "deanngkl/raf-db-7emotions",
        "local_path": "./datasets/RAF-DB-7emotions/",
        "image_col": "image",
        "label_col": "label",
    },
    # ── Optional: MLI-DER (driving-related anger, Sun et al. 2024) ──────────
    # Listed in the CalmWheel design presentation but the dataset is not on
    # HuggingFace yet. To enable, place the unpacked MLI-DER frames under
    # ./datasets/MLI-DER/ as an `datasets.DatasetDict` (use
    # `datasets.Dataset.from_dict({...}).save_to_disk(...)`) and uncomment.
    # Note: MLI-DER has 6 classes (no "happy") — see the design slide for
    # per-class counts; trainers will silently skip "contempt" / unmapped names.
    # "mli_der": {
    #     "repo_id": None,
    #     "local_path": "./datasets/MLI-DER/",
    #     "image_col": "image",
    #     "label_col": "label",
    # },
    # ── Optional: CK+ apex frames (Lucey et al.) ─────────────────────────────
    # Used by Köksal & Gumus 2025 (arXiv:2512.05669) — sequential, but apex
    # frames form a clean 7-class static FER set. Same enablement pattern.
    # "ck_plus": {
    #     "repo_id": None,
    #     "local_path": "./datasets/CK+/",
    #     "image_col": "image",
    #     "label_col": "label",
    # },
}

# Default model location — same file the sequential pipeline auto-downloads,
# so both code paths share one copy. ensure_model() handles the download.
MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task")
NUM_LANDMARKS  = 478                  # MediaPipe FaceLandmarker v2
NUM_BLENDSHAPES = 52                  # MediaPipe canonical blendshape count

# Canonical MediaPipe blendshape names (order matches result.face_blendshapes[0])
BLENDSHAPE_NAMES = [
    "neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
    "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward",
    "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft",
    "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
    "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]

# Cleaning thresholds
MIN_IMAGE_DIM = 48          # pixels — smaller images are likely corrupt
MAX_BLUR_THRESHOLD = 20.0   # Laplacian variance; below = too blurry
MAX_FACES_ALLOWED = 1       # skip if more than one face is detected


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe helper
# ─────────────────────────────────────────────────────────────────────────────

def build_landmarker(model_path: str, blendshapes: bool = False) -> mp_vision.FaceLandmarker:
    """Create a re-usable FaceLandmarker instance.

    If the model file is missing it is auto-downloaded via the shared
    ensure_model() helper, so both this script and the sequential pipeline
    use the same on-disk model.
    """
    if not os.path.exists(model_path):
        ensure_model(Path(model_path))
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=blendshapes,         # enable when needed
        output_facial_transformation_matrixes=False,
        num_faces=2,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def pil_to_mp_image(pil_img: Image.Image) -> mp.Image:
    """Convert a PIL image to a MediaPipe Image (RGB uint8)."""
    rgb = pil_img.convert("RGB")
    arr = np.array(rgb, dtype=np.uint8)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)


# ─────────────────────────────────────────────────────────────────────────────
# Data-cleaning helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_image_valid(pil_img: Image.Image) -> tuple[bool, str]:
    """Return (ok, reason) — reason is empty string when ok."""
    w, h = pil_img.size
    if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
        return False, f"too_small_{w}x{h}"

    # Convert to grayscale numpy for blur check
    gray = np.array(pil_img.convert("L"))
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur < MAX_BLUR_THRESHOLD:
        return False, f"too_blurry_{blur:.1f}"

    return True, ""


def extract_landmarks(
    landmarker: mp_vision.FaceLandmarker,
    pil_img: Image.Image,
) -> tuple[np.ndarray | None, str]:
    """
    Run face landmark detection.

    Returns
    -------
    landmarks : ndarray of shape (478, 3) with (x, y, z) normalised coords,
                or None if detection failed.
    reason    : human-readable skip reason (empty on success).
    """
    mp_img = pil_to_mp_image(pil_img)
    result = landmarker.detect(mp_img)

    if not result.face_landmarks:
        return None, "no_face_detected"

    if len(result.face_landmarks) > MAX_FACES_ALLOWED:
        return None, f"multiple_faces_{len(result.face_landmarks)}"

    lm = result.face_landmarks[0]   # first (and only accepted) face
    coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)

    if coords.shape[0] != NUM_LANDMARKS:
        return None, f"wrong_landmark_count_{coords.shape[0]}"

    return coords, ""


def extract_blendshapes(
    landmarker: mp_vision.FaceLandmarker,
    pil_img: Image.Image,
) -> tuple[np.ndarray | None, str]:
    """
    Run face detection and return the 52 blendshape coefficients.
    Landmarker must have been built with blendshapes=True.

    Returns
    -------
    coeffs : float32 (52,) — one score per blendshape, range [0, 1]
             or None if detection failed.
    reason : human-readable skip reason (empty on success).
    """
    mp_img = pil_to_mp_image(pil_img)
    result = landmarker.detect(mp_img)

    if not result.face_landmarks:
        return None, "no_face_detected"

    if len(result.face_landmarks) > MAX_FACES_ALLOWED:
        return None, f"multiple_faces_{len(result.face_landmarks)}"

    if not result.face_blendshapes:
        return None, "no_blendshapes_returned"

    bs = result.face_blendshapes[0]   # first face
    coeffs = np.array([c.score for c in bs], dtype=np.float32)

    if len(coeffs) != NUM_BLENDSHAPES:
        return None, f"wrong_blendshape_count_{len(coeffs)}"

    return coeffs, ""


def extract_combined(
    landmarker: mp_vision.FaceLandmarker,
    pil_img: Image.Image,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """
    Single MediaPipe pass returning BOTH the 478×3 landmark coords and the
    52 blendshape coefficients. Landmarker must have been built with
    blendshapes=True.

    Returns (landmarks, blendshapes, reason). On any failure, both arrays
    are None and reason is non-empty.
    """
    mp_img = pil_to_mp_image(pil_img)
    result = landmarker.detect(mp_img)

    if not result.face_landmarks:
        return None, None, "no_face_detected"
    if len(result.face_landmarks) > MAX_FACES_ALLOWED:
        return None, None, f"multiple_faces_{len(result.face_landmarks)}"
    if not result.face_blendshapes:
        return None, None, "no_blendshapes_returned"

    lm = result.face_landmarks[0]
    coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    if coords.shape[0] != NUM_LANDMARKS:
        return None, None, f"wrong_landmark_count_{coords.shape[0]}"

    bs = result.face_blendshapes[0]
    coeffs = np.array([c.score for c in bs], dtype=np.float32)
    if len(coeffs) != NUM_BLENDSHAPES:
        return None, None, f"wrong_blendshape_count_{len(coeffs)}"

    return coords, coeffs, ""


# MediaPipe drawing utilities (only imported when --preview is active)
_mp_drawing = None          # mediapipe.tasks.python.vision.drawing_utils
_mp_drawing_styles = None   # mediapipe.tasks.python.vision.drawing_styles

EMOTION_COLORS = {
    # BGR colours for the emotion label badge
    "angry":     (0,   0,   220),
    "disgust":   (0,   140, 255),
    "fear":      (180, 0,   180),
    "happy":     (0,   200, 0  ),
    "neutral":   (160, 160, 160),
    "sad":       (200, 100, 0  ),
    "surprise":  (0,   200, 200),
}
_DEFAULT_BADGE_COLOR = (80, 80, 80)

WINDOW_NAME = "FaceMesh Conversion Preview  [Q to quit]"


def _init_drawing_utils():
    """Import MP tasks drawing utilities once (lazy, they're heavy)."""
    global _mp_drawing, _mp_drawing_styles
    if _mp_drawing is None:
        from mediapipe.tasks.python.vision import drawing_utils as du
        from mediapipe.tasks.python.vision import drawing_styles as ds
        _mp_drawing = du
        _mp_drawing_styles = ds


def _draw_landmarks_on_image(
    rgb_img: np.ndarray,
    face_landmarks_list,   # result.face_landmarks  (list of list of NormalizedLandmark)
) -> np.ndarray:
    """
    Draw face mesh connections using the mediapipe.tasks drawing API.
    face_landmarks_list is the raw result.face_landmarks from FaceLandmarker.
    Returns a BGR numpy array ready for cv2.imshow.
    """
    _init_drawing_utils()

    # The tasks drawing_utils works on an annotated image object
    annotated = np.copy(rgb_img)

    for face_landmarks in face_landmarks_list:
        _mp_drawing.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks,
            connections=mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=_mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        _mp_drawing.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks,
            connections=mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=_mp_drawing_styles.get_default_face_mesh_contours_style(),
        )

    return cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)


def build_preview_frame(
    pil_img: Image.Image,
    face_landmarks_list,  # result.face_landmarks (list of lists), or None if skipped
    label_name: str,
    skip_reason: str,
    stats: dict,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Compose a side-by-side preview frame:
      LEFT  — original image
      RIGHT — face mesh overlay (or red X if skipped)
      BOTTOM bar — emotion label + live kept/skipped counters
    """
    rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]

    panel_h = max(h, 200)
    panel_w = max(w, 200)

    # Left panel: original
    left = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (panel_w, panel_h))

    # Right panel: mesh overlay or skip message
    if face_landmarks_list is not None:
        right = cv2.resize(
            _draw_landmarks_on_image(rgb, face_landmarks_list),
            (panel_w, panel_h),
        )
    else:
        right = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        cv2.line(right, (10, 10), (panel_w - 10, panel_h - 10), (0, 0, 220), 3)
        cv2.line(right, (panel_w - 10, 10), (10, panel_h - 10), (0, 0, 220), 3)
        cv2.putText(
            right, skip_reason, (8, panel_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 220), 1, cv2.LINE_AA,
        )

    combined = np.hstack([left, right])

    # Bottom status bar
    bar_h = 36
    bar = np.zeros((bar_h, combined.shape[1], 3), dtype=np.uint8)

    badge_color = EMOTION_COLORS.get(label_name.lower(), _DEFAULT_BADGE_COLOR)
    kept    = stats.get("kept", 0)
    skipped = stats.get("skipped", 0)
    total   = stats.get("total", 0)
    pct     = 100 * kept / total if total else 0

    status_text = (
        f"  {label_name}   |   "
        f"kept {kept:,}  skipped {skipped:,}  "
        f"({pct:.1f}% kept)  total {total:,}"
    )
    cv2.rectangle(bar, (0, 0), (8, bar_h), badge_color, -1)
    cv2.putText(
        bar, status_text, (14, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
    )

    frame = np.vstack([combined, bar])

    if scale != 1.0:
        new_w = int(frame.shape[1] * scale)
        new_h = int(frame.shape[0] * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return frame


def show_preview(frame: np.ndarray) -> bool:
    """
    Display frame in an OpenCV window.
    Returns False if the user pressed Q/Esc or closed the window.
    """
    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), ord("Q"), 27)   # 27 = Esc


# ─────────────────────────────────────────────────────────────────────────────
# Dataset I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_hf_dataset(cfg: dict):
    """Load from disk cache if available, else download from HuggingFace."""
    local = cfg["local_path"]
    if os.path.exists(local):
        log.info("Loading from disk: %s", local)
        try:
            return load_from_disk(local)
        except Exception as e:
            log.warning("load_from_disk failed (%s), re-downloading…", e)

    log.info("Downloading %s …", cfg["repo_id"])
    ds = load_dataset(cfg["repo_id"])
    ds.save_to_disk(local)
    log.info("Saved to %s", local)
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Per-split processing
# ─────────────────────────────────────────────────────────────────────────────

def process_split(
    split_data,
    landmarker: mp_vision.FaceLandmarker,
    image_col: str,
    label_col: str,
    split_name: str,
    label_names: list | None = None,
    preview: bool = False,
    preview_scale: float = 1.0,
    mode: str = "landmarks",          # "landmarks" | "blendshapes" | "both"
) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    """
    Iterate every example in a split, clean & extract features.

    Returns
    -------
    features     : dict with one or both of:
                     "landmarks"   → float32 (N, 478, 3)
                     "blendshapes" → float32 (N, 52)
    labels_arr   : int64   (N,)
    stats        : dict with counts
    """
    import io as _io

    landmarks_list   = []
    blendshapes_list = []
    labels_list      = []

    stats = {
        "total": len(split_data),
        "kept": 0,
        "skipped": 0,
        "skip_reasons": {},
    }

    def _bump_reason(reason: str):
        stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1

    preview_alive = True   # becomes False when user quits the preview window

    for example in tqdm(split_data, desc=split_name, unit="img"):
        # ── get the image ──────────────────────────────────────────────────
        raw = example[image_col]
        pil_img = None
        skip_reason = ""

        if raw is None:
            _bump_reason("null_image")
            stats["skipped"] += 1
            skip_reason = "null_image"
        elif isinstance(raw, dict):
            try:
                pil_img = Image.open(_io.BytesIO(raw["bytes"]))
            except Exception:
                _bump_reason("corrupt_image_bytes")
                stats["skipped"] += 1
                skip_reason = "corrupt_image_bytes"
        elif isinstance(raw, Image.Image):
            pil_img = raw
        else:
            try:
                pil_img = Image.fromarray(np.array(raw))
            except Exception:
                _bump_reason("unknown_image_type")
                stats["skipped"] += 1
                skip_reason = "unknown_image_type"

        if pil_img is None:
            if preview and preview_alive and skip_reason:
                # Show a placeholder black frame for skipped images
                placeholder = Image.new("RGB", (224, 224), color=(20, 20, 20))
                label_idx = example.get(label_col, 0) or 0
                lname = (label_names[label_idx] if label_names else str(label_idx))
                frame = build_preview_frame(
                    placeholder, None, lname, skip_reason, stats, preview_scale
                )
                preview_alive = show_preview(frame)
            continue

        # ── basic image quality checks ────────────────────────────────────
        ok, reason = is_image_valid(pil_img)
        if not ok:
            _bump_reason(reason)
            stats["skipped"] += 1
            skip_reason = reason

            if preview and preview_alive:
                label_idx = example.get(label_col, 0) or 0
                lname = (label_names[label_idx] if label_names else str(label_idx))
                frame = build_preview_frame(
                    pil_img, None, lname, skip_reason, stats, preview_scale
                )
                preview_alive = show_preview(frame)
            continue

        # ── feature extraction (landmarks, blendshapes, or both) ─────────
        lm_feats = None
        bs_feats = None
        try:
            if mode == "blendshapes":
                bs_feats, reason = extract_blendshapes(landmarker, pil_img)
            elif mode == "both":
                lm_feats, bs_feats, reason = extract_combined(landmarker, pil_img)
            else:
                lm_feats, reason = extract_landmarks(landmarker, pil_img)
        except Exception as e:
            reason = f"mediapipe_error:{type(e).__name__}"
            lm_feats = None
            bs_feats = None

        if (mode == "landmarks"   and lm_feats is None) \
        or (mode == "blendshapes" and bs_feats is None) \
        or (mode == "both"        and (lm_feats is None or bs_feats is None)):
            _bump_reason(reason)
            stats["skipped"] += 1
            skip_reason = reason

            if preview and preview_alive:
                label_idx = example.get(label_col, 0) or 0
                lname = (label_names[label_idx] if label_names else str(label_idx))
                frame = build_preview_frame(
                    pil_img, None, lname, skip_reason, stats, preview_scale
                )
                preview_alive = show_preview(frame)
            continue

        # ── label ─────────────────────────────────────────────────────────
        label = example[label_col]
        if label is None:
            _bump_reason("null_label")
            stats["skipped"] += 1
            if preview and preview_alive:
                frame = build_preview_frame(
                    pil_img, None, "unknown", "null_label", stats, preview_scale
                )
                preview_alive = show_preview(frame)
            continue

        if lm_feats is not None:
            landmarks_list.append(lm_feats)
        if bs_feats is not None:
            blendshapes_list.append(bs_feats)
        labels_list.append(int(label))
        stats["kept"] += 1

        # ── preview: show successful detection ────────────────────────────
        if preview and preview_alive:
            lname = (label_names[int(label)] if label_names else str(label))
            try:
                mp_img = pil_to_mp_image(pil_img)
                result = landmarker.detect(mp_img)
                face_landmarks_list = result.face_landmarks if result.face_landmarks else None
            except Exception:
                face_landmarks_list = None

            frame = build_preview_frame(
                pil_img, face_landmarks_list, lname, "", stats, preview_scale
            )
            preview_alive = show_preview(frame)

    if preview:
        cv2.destroyAllWindows()

    features: dict[str, np.ndarray] = {}
    if mode in ("landmarks", "both"):
        features["landmarks"] = (
            np.stack(landmarks_list, axis=0).astype(np.float32)
            if landmarks_list
            else np.zeros((0, NUM_LANDMARKS, 3), dtype=np.float32)
        )
    if mode in ("blendshapes", "both"):
        features["blendshapes"] = (
            np.stack(blendshapes_list, axis=0).astype(np.float32)
            if blendshapes_list
            else np.zeros((0, NUM_BLENDSHAPES), dtype=np.float32)
        )
    labels_arr = np.array(labels_list, dtype=np.int64)
    return features, labels_arr, stats


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Which datasets to process
    selected = (
        {k: DATASETS_CONFIG[k] for k in args.datasets.split(",")}
        if args.datasets
        else DATASETS_CONFIG
    )

    log.info("Building MediaPipe FaceLandmarker from %s", MODEL_PATH)
    landmarker = build_landmarker(
        MODEL_PATH,
        blendshapes=(args.mode in ("blendshapes", "both")),
    )
    log.info("Feature mode: %s", args.mode)

    if args.preview:
        log.info(
            "Preview mode ON (scale=%.1fx). "
            "Window will open per split. Press Q or Esc to skip remaining preview.",
            args.preview_scale,
        )
        log.info(
            "NOTE: preview requires 'opencv-python' (not 'opencv-python-headless')."
        )

    all_metadata = {}

    for ds_name, cfg in selected.items():
        log.info("═══ Dataset: %s ═══", ds_name)
        ds = load_hf_dataset(cfg)

        # Collect label names if the feature has ClassLabel info
        label_names = None
        try:
            label_names = ds[list(ds.keys())[0]].features[cfg["label_col"]].names
        except Exception:
            pass

        ds_meta = {"dataset": cfg["repo_id"], "splits": {}}

        for split_name, split_data in ds.items():
            log.info("Processing split '%s' (%d examples)…", split_name, len(split_data))
            features, labels_arr, stats = process_split(
                split_data,
                landmarker,
                image_col=cfg["image_col"],
                label_col=cfg["label_col"],
                split_name=f"{ds_name}/{split_name}",
                label_names=label_names,
                preview=args.preview,
                preview_scale=args.preview_scale,
                mode=args.mode,
            )

            # Save .npz — keys reflect the mode so every downstream loader
            # (train.py, train_geo.py → "landmarks"; train2.py → "blendshapes")
            # finds what it needs from a single file.
            out_path = output_dir / f"{ds_name}_{split_name}.npz"
            save_kwargs: dict = dict(labels=labels_arr)
            if "landmarks" in features:
                save_kwargs["landmarks"] = features["landmarks"]
            if "blendshapes" in features:
                save_kwargs["blendshapes"] = features["blendshapes"]
                save_kwargs["blendshape_names"] = np.array(BLENDSHAPE_NAMES)
            if label_names:
                save_kwargs["label_names"] = np.array(label_names)
            np.savez_compressed(out_path, **save_kwargs)

            log.info(
                "  Saved %s  kept=%d / total=%d  (skipped=%d)",
                out_path.name, stats["kept"], stats["total"], stats["skipped"],
            )
            if stats["skip_reasons"]:
                for reason, count in sorted(
                    stats["skip_reasons"].items(), key=lambda x: -x[1]
                ):
                    log.info("    skip %-35s %d", reason, count)

            ds_meta["splits"][split_name] = stats

        if label_names:
            ds_meta["label_names"] = label_names
        all_metadata[ds_name] = ds_meta

    # Write metadata JSON
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    log.info("Metadata written to %s", meta_path)
    log.info("Done. Output folder: %s", output_dir.resolve())


# ─────────────────────────────────────────────────────────────────────────────
# Loading utility (use this in your training notebooks)
# ─────────────────────────────────────────────────────────────────────────────

def load_facemesh_split(npz_path: str):
    """
    Convenience loader for the saved .npz files.
    Auto-detects whether the file contains landmarks or blendshapes.

    Returns
    -------
    features    : float32
                  landmarks mode   → (N, 478, 3)
                  blendshapes mode → (N, 52)
    labels      : int64   (N,)
    label_names : list[str] | None
    feature_names : list[str] | None  — blendshape names when mode=blendshapes
    """
    data = np.load(npz_path, allow_pickle=True)
    if "blendshapes" in data:
        features = data["blendshapes"]
        feature_names = data["blendshape_names"].tolist() if "blendshape_names" in data else None
    else:
        features = data["landmarks"]
        feature_names = None
    labels      = data["labels"]
    label_names = data["label_names"].tolist() if "label_names" in data else None
    return features, labels, label_names, feature_names


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert face emotion datasets to MediaPipe FaceMesh features"
    )
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated subset of datasets to process (affectnet,rafdb). "
             "Default: all.",
    )
    parser.add_argument(
        "--output",
        default="cache/facemesh_dataset",
        help="Output directory for .npz files (default: cache/facemesh_dataset).",
    )
    parser.add_argument(
        "--mode",
        default="landmarks",
        choices=["landmarks", "blendshapes", "both"],
        help=(
            "Feature mode (default: landmarks). "
            "'blendshapes' extracts 52 named MediaPipe blend-shape coefficients "
            "(Jakhete & Kulkarni, ICCUBEA 2024). "
            "'both' saves landmarks AND blendshapes in a single .npz — one "
            "MediaPipe pass per image, satisfying every downstream training "
            "script in TrainVla/."
        ),
    )
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help="Path to face_landmarker.task model file.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a live OpenCV window while processing: left=original, "
             "right=face mesh (or red X if skipped). Press Q or Esc to close.",
    )
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=1.0,
        metavar="SCALE",
        help="Scale factor for the preview window (e.g. 1.5 = 50%% larger). "
             "Default: 1.0.",
    )
    args = parser.parse_args()
    MODEL_PATH = args.model   # allow override via CLI

    main(args)