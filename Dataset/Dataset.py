"""
Usage
-----
    python convert_to_facemesh_dataset.py
    python convert_to_facemesh_dataset.py --datasets affectnet   # one dataset only
    python convert_to_facemesh_dataset.py --output ./my_output   # custom output dir
    python convert_to_facemesh_dataset.py --preview              # show live preview window
    python convert_to_facemesh_dataset.py --preview --preview-scale 1.5  # larger window
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

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from datasets import load_dataset, load_from_disk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


DATASETS_CONFIG = {
    "affectnet": {
        "repo_id": "Piro17/affectnethq",
        "local_path": "./datasets/AffectnetHQ/",
        "image_col": "image",
        "label_col": "label",
    },
    "rafdb": {
        "repo_id": "deanngkl/raf-db-7emotions",
        "local_path": "./datasets/RAF-DB-7emotions/",
        "image_col": "image",
        "label_col": "label",
    },
}

MODEL_PATH = "face_landmarker.task"   
NUM_LANDMARKS = 478                   

MIN_IMAGE_DIM = 48          
MAX_BLUR_THRESHOLD = 20.0   
MAX_FACES_ALLOWED = 1       


def build_landmarker(model_path: str) -> mp_vision.FaceLandmarker:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"MediaPipe model not found at '{model_path}'."
        )
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=2,                         
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def pil_to_mp_image(pil_img: Image.Image) -> mp.Image:
    rgb = pil_img.convert("RGB")
    arr = np.array(rgb, dtype=np.uint8)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)

def is_image_valid(pil_img: Image.Image) -> tuple[bool, str]:
    w, h = pil_img.size
    if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
        return False, f"too_small_{w}x{h}"

    gray = np.array(pil_img.convert("L"))
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur < MAX_BLUR_THRESHOLD:
        return False, f"too_blurry_{blur:.1f}"

    return True, ""


def extract_landmarks(
    landmarker: mp_vision.FaceLandmarker,
    pil_img: Image.Image,
) -> tuple[np.ndarray | None, str]:
    
    mp_img = pil_to_mp_image(pil_img)
    result = landmarker.detect(mp_img)

    if not result.face_landmarks:
        return None, "no_face_detected"

    if len(result.face_landmarks) > MAX_FACES_ALLOWED:
        return None, f"multiple_faces_{len(result.face_landmarks)}"

    lm = result.face_landmarks[0]
    coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)

    if coords.shape[0] != NUM_LANDMARKS:
        return None, f"wrong_landmark_count_{coords.shape[0]}"

    return coords, ""


# Preview 
_mp_drawing = None          
_mp_drawing_styles = None 

EMOTION_COLORS = {
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
    global _mp_drawing, _mp_drawing_styles
    if _mp_drawing is None:
        from mediapipe.tasks.python.vision import drawing_utils as du
        from mediapipe.tasks.python.vision import drawing_styles as ds
        _mp_drawing = du
        _mp_drawing_styles = ds


def _draw_landmarks_on_image(
    rgb_img: np.ndarray,
    face_landmarks_list,
) -> np.ndarray:
    _init_drawing_utils()

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
    face_landmarks_list, 
    label_name: str,
    skip_reason: str,
    stats: dict,
    scale: float = 1.0,
) -> np.ndarray:

    rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]

    panel_h = max(h, 200)
    panel_w = max(w, 200)

    left = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (panel_w, panel_h))

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
    
    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), ord("Q"), 27)   # 27 = Esc

# Dataset loading and processing

def load_hf_dataset(cfg: dict):
    
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


def process_split(
    split_data,
    landmarker: mp_vision.FaceLandmarker,
    image_col: str,
    label_col: str,
    split_name: str,
    label_names: list | None = None,
    preview: bool = False,
    preview_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    import io as _io

    landmarks_list = []
    labels_list = []

    stats = {
        "total": len(split_data),
        "kept": 0,
        "skipped": 0,
        "skip_reasons": {},
    }

    def _bump_reason(reason: str):
        stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1

    preview_alive = True 

    for example in tqdm(split_data, desc=split_name, unit="img"):
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
                placeholder = Image.new("RGB", (224, 224), color=(20, 20, 20))
                label_idx = example.get(label_col, 0) or 0
                lname = (label_names[label_idx] if label_names else str(label_idx))
                frame = build_preview_frame(
                    placeholder, None, lname, skip_reason, stats, preview_scale
                )
                preview_alive = show_preview(frame)
            continue

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

        mp_landmarks = None
        try:
            coords, reason = extract_landmarks(landmarker, pil_img)
        except Exception as e:
            reason = f"mediapipe_error:{type(e).__name__}"
            coords = None

        if coords is None:
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

        landmarks_list.append(coords)
        labels_list.append(int(label))
        stats["kept"] += 1

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

    landmarks_arr = (
        np.stack(landmarks_list, axis=0) if landmarks_list
        else np.zeros((0, NUM_LANDMARKS, 3), dtype=np.float32)
    )
    labels_arr = np.array(labels_list, dtype=np.int64)
    return landmarks_arr, labels_arr, stats


def main(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = (
        {k: DATASETS_CONFIG[k] for k in args.datasets.split(",")}
        if args.datasets
        else DATASETS_CONFIG
    )

    log.info("Building MediaPipe FaceLandmarker from %s", MODEL_PATH)
    landmarker = build_landmarker(MODEL_PATH)

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

        label_names = None
        try:
            label_names = ds[list(ds.keys())[0]].features[cfg["label_col"]].names
        except Exception:
            pass

        ds_meta = {"dataset": cfg["repo_id"], "splits": {}}

        for split_name, split_data in ds.items():
            log.info("Processing split '%s' (%d examples)…", split_name, len(split_data))
            landmarks_arr, labels_arr, stats = process_split(
                split_data,
                landmarker,
                image_col=cfg["image_col"],
                label_col=cfg["label_col"],
                split_name=f"{ds_name}/{split_name}",
                label_names=label_names,
                preview=args.preview,
                preview_scale=args.preview_scale,
            )

            out_path = output_dir / f"{ds_name}_{split_name}.npz"
            save_kwargs = dict(landmarks=landmarks_arr, labels=labels_arr)
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

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    log.info("Metadata written to %s", meta_path)
    log.info("Done. Output folder: %s", output_dir.resolve())


# Loading utility (use this in your training notebooks)

def load_facemesh_split(npz_path: str):
    
    data = np.load(npz_path, allow_pickle=True)
    landmarks = data["landmarks"]
    labels = data["labels"]
    label_names = data["label_names"].tolist() if "label_names" in data else None
    return landmarks, labels, label_names


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert face emotion datasets to MediaPipe FaceMesh landmarks"
    )
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated subset of datasets to process (affectnet,rafdb). "
             "Default: all.",
    )
    parser.add_argument(
        "--output",
        default="./facemesh_dataset",
        help="Output directory for .npz files (default: ./facemesh_dataset).",
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
    MODEL_PATH = args.model   

    main(args)