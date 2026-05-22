"""Real-time webcam emotion recognition.

Replicates Section 4.3 of the paper:
  * MediaPipe FaceMesh in streaming mode on every webcam frame.
  * Sliding 5-frame buffer (default 400 ms cadence so 5 frames span ~2 s,
    enough to cover the onset → apex window of a macro expression).
  * Build (4, A) feature tensor, scale with the saved StandardScaler, predict.
  * Track the mean absolute distance feature to surface onset / apex / offset.

Usage:
    python -m implementation.realtime \
        --model    runs/ravdess/fold_best.pt \
        --scaler   runs/ravdess/scaler.pkl \
        --labels   runs/ravdess/labels.json \
        [--camera 0] [--interval_ms 400]
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import time
from pathlib import Path
from typing import Deque

import cv2
import numpy as np
import torch

from dataset.features import build_pair_indices, sequence_features
from dataset.landmarks import FaceMeshDetector
from training.model import build_model
from training.config import TrainConfig


def overlay_text(frame: np.ndarray, lines: list[str], origin=(10, 28)) -> None:
    x, y = origin
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (x, y + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )


def classify_phase(mean_dist_history: list[float]) -> str:
    """Crude onset/apex/offset detector from mean |distance feature|.

    The paper notes that mean distance is small during neutral, grows during
    onset, peaks at apex, then decays during offset. We compare the latest
    sample against a short running max.
    """
    if len(mean_dist_history) < 3:
        return "..."
    recent = mean_dist_history[-5:]
    peak = max(mean_dist_history[-15:])
    cur = recent[-1]
    if cur < 0.2 * peak:
        return "neutral"
    if cur >= 0.9 * peak and cur >= recent[-2]:
        return "apex"
    if cur > recent[-2]:
        return "onset"
    return "offset"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--scaler", required=True, help="StandardScaler pickle from training")
    ap.add_argument("--labels", required=True, help="labels.json from training")
    ap.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    ap.add_argument("--interval_ms", type=int, default=400,
                    help="Time between buffered frames (paper uses 400 ms)")
    ap.add_argument("--buffer", type=int, default=5,
                    help="Sliding-window length (paper uses 5)")
    ap.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    labels = json.loads(Path(args.labels).read_text())
    with open(args.scaler, "rb") as f:
        scaler = pickle.load(f)

    # Rebuild the model from the dims stored in the checkpoint, then load weights.
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    cfg = TrainConfig(**ckpt["config"]) if "config" in ckpt else TrainConfig()
    model = build_model(ckpt["time_steps"], ckpt["feature_dim"],
                        ckpt["num_classes"], cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    pair_idx = build_pair_indices()
    feat_dim = 2 * len(pair_idx)
    time_steps = args.buffer - 1

    detector = FaceMeshDetector(
        static_image_mode=False,
        max_num_faces=1,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        return 1

    buf: Deque[np.ndarray] = collections.deque(maxlen=args.buffer)
    mean_dist_history: list[float] = []
    last_sampled = 0.0
    last_pred = "—"
    last_conf = 0.0
    last_phase = "..."

    print("Press 'q' to quit.")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            now = time.time()

            pts = detector.detect_selected(rgb)
            if pts is not None:
                h, w = rgb.shape[:2]
                pts_px = pts.copy()
                pts_px[:, 0] *= w
                pts_px[:, 1] *= h

                if (now - last_sampled) * 1000.0 >= args.interval_ms:
                    buf.append(pts_px)
                    last_sampled = now

                    if len(buf) == args.buffer:
                        seq = np.stack(buf)              # (5, 61, 2)
                        feats = sequence_features(seq, pair_idx)   # (4, A)
                        flat = feats.reshape(-1, feat_dim)
                        scaled = scaler.transform(flat).reshape(1, time_steps, feat_dim)
                        with torch.no_grad():
                            xb = torch.from_numpy(scaled.astype(np.float32)).to(device)
                            probs = torch.softmax(model(xb), dim=1)[0].cpu().numpy()
                        idx = int(np.argmax(probs))
                        last_pred = labels[idx]
                        last_conf = float(probs[idx])

                        # Track onset/apex/offset via mean |distance| feature.
                        # First half of `feats` columns are distances.
                        mean_dist = float(np.mean(np.abs(feats[:, : len(pair_idx)])))
                        mean_dist_history.append(mean_dist)
                        if len(mean_dist_history) > 60:
                            mean_dist_history.pop(0)
                        last_phase = classify_phase(mean_dist_history)

                # Draw the 61 selected landmarks for visual feedback.
                for x, y in pts_px.astype(int):
                    cv2.circle(frame_bgr, (x, y), 1, (255, 200, 0), -1)

            overlay_text(
                frame_bgr,
                [
                    f"Emotion: {last_pred}  ({last_conf:.2f})",
                    f"Phase:   {last_phase}",
                    f"Buffer:  {len(buf)}/{args.buffer}",
                ],
            )
            cv2.imshow("Emotion (q to quit)", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
