"""
emotionrec/test_fer.py — Standalone FER test (Geo 122)
=======================================================
Webcam loop to verify the Geo-122 GeoMLP checkpoint.
Shows bounding box, emotion label, confidence, and per-class probability bars.

Usage:
    python -m emotionrec.test_fer
    python -m emotionrec.test_fer --checkpoint emotion-reco/runs/geo_static/checkpoints/best.pt
    python -m emotionrec.test_fer --camera 1 --conf 0.4

Press Q or ESC to quit.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import mediapipe as mp

from emotionrec.realtime_fer import (
    EMOTIONS, EMOTION_COLORS_BGR,
    GeoMLP, load_model,
    create_face_landmarker, extract_landmarks, geometric_features,
    get_face_bbox, predict,
)

NUM_CLASSES = len(EMOTIONS)


def draw_bbox_label(frame, label, conf, bbox, conf_thresh):
    color = EMOTION_COLORS_BGR[label]
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    if conf >= conf_thresh:
        text = f"{label.upper()}  {conf:.0%}"
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
        cv2.rectangle(frame, (x, y - th - bl - 10), (x + tw + 10, y), color, -1)
        cv2.putText(frame, text, (x + 5, y - bl - 5), font, scale, (255, 255, 255), thick)
    else:
        cv2.putText(frame, "? low conf", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 120, 120), 1)


def draw_prob_bars(frame, probs, top_label):
    fh, fw   = frame.shape[:2]
    bar_h    = 22
    bar_max  = 180
    margin_r = 14
    pad_y    = fh - NUM_CLASSES * (bar_h + 6) - 14
    label_w  = 82
    pct_w    = 46
    panel_x  = fw - margin_r - bar_max - label_w - pct_w - 8
    panel_y  = pad_y - 6
    panel_w  = label_w + bar_max + pct_w + 16
    panel_h  = NUM_CLASSES * (bar_h + 6) + 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, (emotion, prob) in enumerate(zip(EMOTIONS, probs)):
        y0     = pad_y + i * (bar_h + 6)
        color  = EMOTION_COLORS_BGR[emotion]
        is_top = (emotion == top_label)
        bw     = int(prob * bar_max)
        bar_x  = panel_x + label_w + 6
        cv2.rectangle(frame, (bar_x, y0), (bar_x + bw, y0 + bar_h), color, -1)
        cv2.rectangle(frame, (bar_x, y0), (bar_x + bar_max, y0 + bar_h), (60, 60, 60), 1)
        cv2.putText(frame, emotion, (panel_x + 4, y0 + bar_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    color if is_top else (220, 220, 220), 2 if is_top else 1)
        cv2.putText(frame, f"{prob:.0%}", (bar_x + bar_max + 5, y0 + bar_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FER-test] Device : {device}")

    model, scaler, pairs_arr = load_model(args.checkpoint, device)

    lm_path = Path(args.landmarker)
    if not lm_path.exists():
        raise FileNotFoundError(f"face_landmarker.task not found: {lm_path}")
    landmarker = create_face_landmarker(str(lm_path))
    print(f"[FER-test] Landmarker : {lm_path}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"[FER-test] Camera : {args.camera}  —  press Q / ESC to quit\n")

    SMOOTH = 5
    buf    = []
    t_prev = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fh, fw = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)

        landmarks = extract_landmarks(result)
        if landmarks is not None:
            feats        = geometric_features(landmarks, pairs_arr)
            label, probs = predict(feats, model, scaler, device)
            conf         = float(probs.max())
            bbox         = get_face_bbox(result, fw, fh)

            buf.append(label)
            if len(buf) > SMOOTH:
                buf.pop(0)
            smooth_label = max(set(buf), key=buf.count) if buf else label

            if bbox:
                draw_bbox_label(frame, smooth_label, conf, bbox, args.conf)
            draw_prob_bars(frame, probs, smooth_label)
        else:
            buf.clear()
            cv2.putText(frame, "No face detected", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 255), 2)

        t_now  = time.perf_counter()
        fps    = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now
        cv2.putText(frame, f"FPS {fps:.1f}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

        cv2.imshow("CalmWheel — FER test (Geo 122)  |  Q to quit", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[FER-test] Stopped.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                   default="emotion-reco/runs/geo_static/checkpoints/best.pt")
    p.add_argument("--landmarker", default="face_landmarker.task")
    p.add_argument("--camera",     type=int,   default=0)
    p.add_argument("--conf",       type=float, default=0.3)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
