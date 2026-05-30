"""
emotionrec/realtime_fer.py — Geo 122 Landmark Model
====================================================
Uses MediaPipe FaceLandmarker to extract 478 landmarks per frame, computes
pairwise geometric features (distance + angle) for the 122-landmark FACS-
grouped subset, then classifies emotion with GeoMLP (93.7 % on CK+).

Inputs:  camera frames
Outputs: on_emotion(emotion, confidence) callback
         on_frame(annotated_bgr) callback → ui.py camera panel

Usage (standalone):
    python -m emotionrec.realtime_fer
    python -m emotionrec.realtime_fer --checkpoint emotion-reco/runs/geo_static/checkpoints/best.pt
"""

import argparse
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

EMOTIONS    = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(EMOTIONS)

EMOTION_COLORS_BGR = {
    "angry":    (0,   0,   220),
    "disgust":  (0,   140, 0  ),
    "fear":     (160, 0,   160),
    "happy":    (0,   200, 200),
    "neutral":  (180, 180, 180),
    "sad":      (200, 80,  0  ),
    "surprise": (0,   180, 255),
}


# ── GeoMLP (must match emotion-reco/training/train_geo_static.py) ─────────
class GeoMLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 7, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(512, 256),   nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 128),   nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── Geometric feature extraction (vectorized) ─────────────────────────────
def geometric_features(landmarks: np.ndarray, pairs_arr: np.ndarray) -> np.ndarray:
    """
    landmarks : (478, 3)  normalized [0,1] x,y,z from MediaPipe
    pairs_arr : (N, 2)    int32 index pairs pre-loaded from checkpoint

    Returns float32 (2*N,): distances then angles (Köksal & Gumus Eqs. 1 & 2).
    """
    xy = landmarks[:, :2]
    i, j = pairs_arr[:, 0], pairs_arr[:, 1]
    dx = xy[i, 0] - xy[j, 0]
    dy = xy[i, 1] - xy[j, 1]
    return np.concatenate([np.hypot(dx, dy), np.arctan2(dx, dy + 1e-9)]).astype(np.float32)


# ── Model loading ─────────────────────────────────────────────────────────
def load_model(checkpoint: str, device: torch.device):
    """Returns (model, scaler, pairs_arr) from a geo_static checkpoint."""
    ckpt      = torch.load(checkpoint, map_location=device, weights_only=False)
    pairs     = ckpt["pairs"]
    pairs_arr = np.array(pairs, dtype=np.int32)
    cfg       = ckpt.get("config", {})
    in_dim    = 2 * len(pairs)

    model = GeoMLP(in_dim=in_dim, num_classes=NUM_CLASSES, dropout=cfg.get("dropout", 0.5))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    print(f"[FER] Geo-122 loaded  feature_dim={in_dim}  pairs={len(pairs)}")
    return model, ckpt["scaler"], pairs_arr


# ── MediaPipe face landmarker ─────────────────────────────────────────────
def create_face_landmarker(model_path: str):
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def extract_landmarks(result) -> Optional[np.ndarray]:
    """Return (478, 3) float32 of normalized landmarks, or None if no face."""
    if not result.face_landmarks:
        return None
    arr = np.array([[lm.x, lm.y, lm.z] for lm in result.face_landmarks[0]], dtype=np.float32)
    return arr if arr.shape[0] >= 478 else None


# ── Inference ─────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(features: np.ndarray, model, scaler, device) -> tuple[str, np.ndarray]:
    scaled = scaler.transform(features.reshape(1, -1))
    tensor = torch.tensor(scaled, dtype=torch.float32).to(device)
    probs  = torch.softmax(model(tensor), dim=1).squeeze(0).cpu().numpy()
    return EMOTIONS[int(probs.argmax())], probs


# ── Frame annotation ──────────────────────────────────────────────────────
def annotate_frame(frame, label, probs, conf, conf_thresh, show_bars, face_bbox=None):
    color = EMOTION_COLORS_BGR.get(label, (200, 200, 200))

    if face_bbox is not None:
        x, y, w, h = face_bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if conf >= conf_thresh:
            text = f"{label.upper()}  {conf:.0%}"
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.rectangle(frame, (x, y - th - bl - 8), (x + tw + 8, y), color, -1)
            cv2.putText(frame, text, (x + 4, y - bl - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "? (low conf)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
    else:
        if conf >= conf_thresh:
            cv2.putText(frame, f"{label.upper()} {conf:.0%}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if show_bars and probs is not None:
        _draw_bars(frame, probs, label)
    return frame


def _draw_bars(frame, probs, top_label):
    fh, fw  = frame.shape[:2]
    bar_h   = 18
    bar_max = 160
    pad_x   = 12
    pad_y   = fh - (NUM_CLASSES * (bar_h + 4)) - 10
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (pad_x - 4, pad_y - 4),
                  (pad_x + bar_max + 130, pad_y + NUM_CLASSES * (bar_h + 4) + 4),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, (emotion, prob) in enumerate(zip(EMOTIONS, probs)):
        y0     = pad_y + i * (bar_h + 4)
        color  = EMOTION_COLORS_BGR[emotion]
        bw     = int(prob * bar_max)
        is_top = (emotion == top_label)
        cv2.rectangle(frame, (pad_x + 70, y0), (pad_x + 70 + bw, y0 + bar_h), color, -1)
        cv2.putText(frame, emotion, (pad_x, y0 + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 255, 255) if not is_top else color, 1 + int(is_top))
        cv2.putText(frame, f"{prob:.0%}",
                    (pad_x + 70 + bar_max + 6, y0 + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)


def get_face_bbox(result, frame_w, frame_h):
    if not result.face_landmarks:
        return None
    lms = result.face_landmarks[0]
    xs  = [lm.x * frame_w for lm in lms]
    ys  = [lm.y * frame_h for lm in lms]
    x1, y1 = int(min(xs)), int(min(ys))
    x2, y2 = int(max(xs)), int(max(ys))
    pad = 10
    return (max(0, x1 - pad), max(0, y1 - pad),
            min(frame_w, x2 + pad) - max(0, x1 - pad),
            min(frame_h, y2 + pad) - max(0, y1 - pad))


# ── Main loop ─────────────────────────────────────────────────────────────
def run(args,
        on_emotion: Optional[Callable[[str, float], None]] = None,
        on_frame:   Optional[Callable[[np.ndarray], None]] = None,
        on_cap:     Optional[Callable] = None):

    SMOOTH_FRAMES     = 5
    CALLBACK_COOLDOWN = 15.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FER] Device: {device}")

    model, scaler, pairs_arr = load_model(args.checkpoint, device)

    lm_path = getattr(args, "landmarker", "face_landmarker.task")
    if not Path(lm_path).exists():
        raise FileNotFoundError(
            f"face_landmarker.task not found at '{lm_path}'.\n"
            "Download: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker"
        )
    landmarker = create_face_landmarker(lm_path)
    print("[FER] MediaPipe landmarker ready.")

    use_kinect = getattr(args, "kinect", False)
    if use_kinect:
        from emotionrec.kinect_source import KinectSource
        cap = KinectSource(
            device_index=getattr(args, "kinect_index", 0),
            rgb_camera_index=getattr(args, "camera", 0),
        )
        if on_cap:
            on_cap(cap)
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {args.camera}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    emotion_buf     = []
    last_callback_t = 0.0
    last_sent       = None

    print("[FER] Running. Press Q or ESC to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fh, fw = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)

        landmarks = extract_landmarks(result)
        label, probs, conf, bbox = None, None, 0.0, None

        if landmarks is not None:
            feats        = geometric_features(landmarks, pairs_arr)
            label, probs = predict(feats, model, scaler, device)
            conf         = float(probs.max())
            bbox         = get_face_bbox(result, fw, fh)

        annotated = annotate_frame(
            frame.copy(), label or "neutral", probs,
            conf, args.conf, not args.no_bar, bbox
        )
        if label is None:
            cv2.putText(annotated, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 255), 2)

        if use_kinect:
            src   = cap.active_source
            lum   = cap.last_luminance
            color = (0, 200, 0) if src == "rgb" else (0, 180, 255)
            badge = f"CAM: {'SYS RGB' if src == 'rgb' else 'KINECT IR'}  lum={lum:.0f}"
            (bw, bh), bl = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            fh2, fw2 = annotated.shape[:2]
            bx = fw2 - bw - 14
            cv2.rectangle(annotated, (bx - 4, 10), (bx + bw + 4, 10 + bh + bl + 4), (30, 30, 30), -1)
            cv2.putText(annotated, badge, (bx, 10 + bh),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        if on_frame is not None:
            on_frame(annotated)

        if label is not None and on_emotion is not None:
            emotion_buf.append(label)
            if len(emotion_buf) > SMOOTH_FRAMES:
                emotion_buf.pop(0)

            if len(emotion_buf) == SMOOTH_FRAMES:
                dominant = max(set(emotion_buf), key=emotion_buf.count)
                count    = emotion_buf.count(dominant)
                now      = time.perf_counter()

                if (count >= SMOOTH_FRAMES * 0.6 and
                        (dominant != last_sent or now - last_callback_t > CALLBACK_COOLDOWN)):
                    conf_val = float(probs[EMOTIONS.index(dominant)])
                    if conf_val < 0.70:
                        dominant = "neutral"
                        conf_val = float(probs[EMOTIONS.index("neutral")])
                    on_emotion(dominant, conf_val)
                    last_sent       = dominant
                    last_callback_t = now
        else:
            emotion_buf.clear()

        if on_frame is None:
            cv2.imshow("FER — Geo 122 | Q to quit", annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break

    cap.release()
    if on_frame is None:
        cv2.destroyAllWindows()
    print("[FER] Stopped.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                   default="emotion-reco/runs/geo_static/checkpoints/best.pt")
    p.add_argument("--landmarker",   default="face_landmarker.task")
    p.add_argument("--camera",       type=int,   default=0)
    p.add_argument("--conf",         type=float, default=0.3)
    p.add_argument("--no-bar",       action="store_true")
    p.add_argument("--kinect",       action="store_true")
    p.add_argument("--kinect-index", type=int,   default=0, dest="kinect_index")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
