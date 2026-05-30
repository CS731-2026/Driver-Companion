"""Real-time webcam inference for any trained model in this project.

Supported methods (pass with --method):

  sequential   ConvLSTM1D (Köksal & Gumus 2025) — buffered 5-frame
               landmark distance+angle features. Needs --scaler and --labels.
  emonet       Dual-stream landmark CNN + geometric MLP (DriveEmo-FL).
  blendshape   MLP / attention on 52 MediaPipe blendshapes
               (Jakhete & Kulkarni 2024). MLP vs attention is detected from
               the checkpoint config.
  geo_static   Landmark distance+angle MLP (Köksal & Gumus 2025, static
               adaptation).
  image        MobileNetV2 (or any torchvision backbone in --backbone) on the
               raw RGB frame.

The sequential predictor needs N frames before its first inference; the four
static predictors classify each frame independently — much faster, no warmup.

Examples:
    python -m implementation.realtime --method sequential \\
        --model runs/ckplus/fold_best.pt \\
        --scaler runs/ckplus/scaler.pkl --labels runs/ckplus/labels.json

    python -m implementation.realtime --method blendshape \\
        --model runs/blendshape_attn_ckplus/checkpoints/best.pt

    python -m implementation.realtime --method emonet \\
        --model runs/emonet_ckplus/checkpoints/best.pt

    python -m implementation.realtime --method geo_static \\
        --model runs/geo_static_ckplus/checkpoints/best.pt

    python -m implementation.realtime --method image \\
        --model runs/image/checkpoints/best.pt --backbone mobilenet_v2
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import time
from pathlib import Path
from typing import Deque, List, Optional

import cv2
import numpy as np
import torch

from dataset.landmarks import FaceMeshDetector


# Shared 7-class Ekman vocabulary the static-image trainers use.
UNIFIED_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def overlay_text(frame: np.ndarray, lines: List[str], origin=(10, 28)) -> None:
    x, y = origin
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 1, cv2.LINE_AA)


def classify_phase(history: List[float]) -> str:
    """Crude onset/apex/offset detector from mean |distance feature|.

    Used only by the sequential predictor. Compares the latest sample against
    a short running max — small = neutral, growing = onset, peak = apex,
    decaying = offset.
    """
    if len(history) < 3:
        return "..."
    recent = history[-5:]
    peak = max(history[-15:])
    cur = recent[-1]
    if cur < 0.2 * peak:
        return "neutral"
    if cur >= 0.9 * peak and cur >= recent[-2]:
        return "apex"
    if cur > recent[-2]:
        return "onset"
    return "offset"


# ─────────────────────────────────────────────────────────────────────────────
# Predictors — one per method.
# Each predict(bgr, rgb) returns the overlay lines to draw on the BGR frame,
# and may annotate `bgr` in place (landmark dots, bounding boxes, etc.).
# ─────────────────────────────────────────────────────────────────────────────


class SequentialPredictor:
    """ConvLSTM1D over a sliding N-frame landmark window."""

    def __init__(self, args: argparse.Namespace, device: torch.device):
        from dataset.features import build_pair_indices, sequence_features
        from training.config import TrainConfig
        from training.model import build_model

        if not args.labels or not args.scaler:
            raise SystemExit("--scaler and --labels are required for sequential")
        self.device = device
        self.labels = json.loads(Path(args.labels).read_text())
        with open(args.scaler, "rb") as f:
            self.scaler = pickle.load(f)

        ckpt = torch.load(args.model, map_location=device, weights_only=False)
        cfg = TrainConfig(**ckpt["config"]) if "config" in ckpt else TrainConfig()
        self.model = build_model(ckpt["time_steps"], ckpt["feature_dim"],
                                 ckpt["num_classes"], cfg).to(device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.selected = ckpt.get("selected_landmarks")
        pair_idx = ckpt.get("pair_idx")
        self.pair_idx = (build_pair_indices() if pair_idx is None
                         else np.asarray(pair_idx))
        self.feat_dim = 2 * len(self.pair_idx)
        self.time_steps = args.buffer - 1
        self.interval_ms = args.interval_ms
        self._sequence_features = sequence_features

        self.detector = FaceMeshDetector(static_image_mode=False, max_num_faces=1)
        self.buf: Deque[np.ndarray] = collections.deque(maxlen=args.buffer)
        self.last_sampled = 0.0
        self.history: List[float] = []
        self.label = "—"
        self.conf = 0.0
        self.phase = "..."

    def predict(self, frame_bgr: np.ndarray, frame_rgb: np.ndarray) -> List[str]:
        now = time.time()
        pts = self.detector.detect_selected(frame_rgb, selected=self.selected)
        if pts is not None:
            h, w = frame_rgb.shape[:2]
            pts_px = pts.copy()
            pts_px[:, 0] *= w
            pts_px[:, 1] *= h

            if (now - self.last_sampled) * 1000.0 >= self.interval_ms:
                self.buf.append(pts_px)
                self.last_sampled = now

                if len(self.buf) == self.buf.maxlen:
                    seq = np.stack(self.buf)
                    feats = self._sequence_features(seq, self.pair_idx)
                    flat = feats.reshape(-1, self.feat_dim)
                    scaled = self.scaler.transform(flat).reshape(
                        1, self.time_steps, self.feat_dim
                    )
                    with torch.no_grad():
                        xb = torch.from_numpy(scaled.astype(np.float32)).to(self.device)
                        probs = torch.softmax(self.model(xb), dim=1)[0].cpu().numpy()
                    idx = int(np.argmax(probs))
                    self.label = self.labels[idx]
                    self.conf = float(probs[idx])

                    mean_dist = float(np.mean(np.abs(feats[:, : len(self.pair_idx)])))
                    self.history.append(mean_dist)
                    if len(self.history) > 60:
                        self.history.pop(0)
                    self.phase = classify_phase(self.history)

            for xp, yp in pts_px.astype(int):
                cv2.circle(frame_bgr, (xp, yp), 1, (255, 200, 0), -1)

        return [
            f"Emotion: {self.label}  ({self.conf:.2f})",
            f"Phase:   {self.phase}",
            f"Buffer:  {len(self.buf)}/{self.buf.maxlen}",
        ]

    def close(self) -> None:
        self.detector.close()


class _StaticBase:
    """Common scaffolding for the four single-frame predictors."""

    def __init__(self, device: torch.device, *, output_blendshapes: bool = False):
        self.device = device
        self.detector = FaceMeshDetector(
            static_image_mode=False,
            max_num_faces=1,
            output_blendshapes=output_blendshapes,
        )
        self.label = "—"
        self.conf = 0.0

    def overlay_lines(self) -> List[str]:
        return [f"Emotion: {self.label}  ({self.conf:.2f})"]

    def close(self) -> None:
        self.detector.close()


class EmoNetPredictor(_StaticBase):
    """Dual-stream EmoNet: landmark grid CNN + geometric MLP."""

    def __init__(self, args: argparse.Namespace, device: torch.device):
        super().__init__(device, output_blendshapes=False)
        from training.train_emonet import EmoNet, extract_geometric_features

        self.model = EmoNet(num_classes=len(UNIFIED_EMOTIONS)).to(device)
        state = torch.load(args.model, map_location=device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self._geo = extract_geometric_features
        self.labels = UNIFIED_EMOTIONS

    @staticmethod
    def _to_spatial_grid(lm: np.ndarray) -> np.ndarray:
        flat = lm.flatten()
        padded = np.zeros(22 * 22 * 3, dtype=np.float32)
        padded[: len(flat)] = flat
        return padded.reshape(3, 22, 22)

    def predict(self, frame_bgr: np.ndarray, frame_rgb: np.ndarray) -> List[str]:
        lm = self.detector.detect(frame_rgb)
        if lm is not None:
            grid = self._to_spatial_grid(lm)
            geo = self._geo(lm)
            with torch.no_grad():
                g = torch.from_numpy(grid).unsqueeze(0).to(self.device)
                v = torch.from_numpy(geo).unsqueeze(0).to(self.device)
                probs = torch.softmax(self.model(g, v), dim=1)[0].cpu().numpy()
            idx = int(np.argmax(probs))
            self.label = self.labels[idx]
            self.conf = float(probs[idx])

            # Light landmark overlay for visual feedback.
            h, w = frame_rgb.shape[:2]
            for x, y, _z in lm[::8]:        # subsample ~60 dots
                cv2.circle(frame_bgr, (int(x * w), int(y * h)), 1, (255, 200, 0), -1)
        return self.overlay_lines()


class BlendshapePredictor(_StaticBase):
    """MLP / attention over the 52 MediaPipe blendshape coefficients."""

    def __init__(self, args: argparse.Namespace, device: torch.device):
        super().__init__(device, output_blendshapes=True)
        from training.train_blendshape import (
            BlendshapeAttentionNet, BlendshapeMLP, NUM_BLENDSHAPES,
        )

        state = torch.load(args.model, map_location=device, weights_only=False)
        self.scaler = state.get("scaler")
        cfg = state.get("config", {})
        kind = cfg.get("model", "mlp")
        cls = BlendshapeAttentionNet if kind == "attention" else BlendshapeMLP
        self.model = cls(num_classes=len(UNIFIED_EMOTIONS),
                         dropout=cfg.get("dropout", 0.4)).to(device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.kind = kind
        self.labels = UNIFIED_EMOTIONS
        self._n_blendshapes = NUM_BLENDSHAPES

    def predict(self, frame_bgr: np.ndarray, frame_rgb: np.ndarray) -> List[str]:
        bs = self.detector.detect_blendshapes(frame_rgb)
        if bs is not None and bs.shape == (self._n_blendshapes,):
            x = bs.reshape(1, -1).astype(np.float32)
            if self.scaler is not None:
                x = self.scaler.transform(x).astype(np.float32)
            with torch.no_grad():
                xb = torch.from_numpy(x).to(self.device)
                probs = torch.softmax(self.model(xb), dim=1)[0].cpu().numpy()
            idx = int(np.argmax(probs))
            self.label = self.labels[idx]
            self.conf = float(probs[idx])
        return self.overlay_lines() + [f"Model:   blendshape_{self.kind}"]


class GeoStaticPredictor(_StaticBase):
    """Static-image landmark distance+angle features → MLP."""

    def __init__(self, args: argparse.Namespace, device: torch.device):
        super().__init__(device, output_blendshapes=False)
        from training.train_geo_static import GeoMLP, geometric_features

        state = torch.load(args.model, map_location=device, weights_only=False)
        self.scaler = state.get("scaler")
        self.pairs = state["pairs"]
        cfg = state.get("config", {})
        in_dim = 2 * len(self.pairs)
        self.model = GeoMLP(in_dim=in_dim,
                            num_classes=len(UNIFIED_EMOTIONS),
                            dropout=cfg.get("dropout", 0.5)).to(device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self._features = geometric_features
        self.labels = UNIFIED_EMOTIONS

    def predict(self, frame_bgr: np.ndarray, frame_rgb: np.ndarray) -> List[str]:
        lm = self.detector.detect(frame_rgb)
        if lm is not None:
            feats = self._features(lm, self.pairs).reshape(1, -1).astype(np.float32)
            if self.scaler is not None:
                feats = self.scaler.transform(feats).astype(np.float32)
            with torch.no_grad():
                xb = torch.from_numpy(feats).to(self.device)
                probs = torch.softmax(self.model(xb), dim=1)[0].cpu().numpy()
            idx = int(np.argmax(probs))
            self.label = self.labels[idx]
            self.conf = float(probs[idx])
            h, w = frame_rgb.shape[:2]
            for x, y, _z in lm[::8]:
                cv2.circle(frame_bgr, (int(x * w), int(y * h)), 1, (255, 200, 0), -1)
        return self.overlay_lines()


class ImagePredictor:
    """ImageNet-pretrained backbone on the raw RGB frame (no MediaPipe)."""

    def __init__(self, args: argparse.Namespace, device: torch.device):
        from torchvision import transforms
        from training.train_image import build_backbone

        self.device = device
        self.model = build_backbone(args.backbone, len(UNIFIED_EMOTIONS)).to(device)
        state = torch.load(args.model, map_location=device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self.labels = UNIFIED_EMOTIONS

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.backbone = args.backbone
        self.label = "—"
        self.conf = 0.0

    def predict(self, frame_bgr: np.ndarray, frame_rgb: np.ndarray) -> List[str]:
        x = self.transform(frame_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        self.label = self.labels[idx]
        self.conf = float(probs[idx])
        return [
            f"Emotion: {self.label}  ({self.conf:.2f})",
            f"Model:   {self.backbone}",
        ]

    def close(self) -> None:  # nothing to release
        pass


PREDICTORS = {
    "sequential": SequentialPredictor,
    "emonet":     EmoNetPredictor,
    "blendshape": BlendshapePredictor,
    "geo_static": GeoStaticPredictor,
    "image":      ImagePredictor,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--method", required=True, choices=list(PREDICTORS),
                    help="Which trained model to run")
    ap.add_argument("--model", required=True, help="Path to the .pt checkpoint")
    # Sequential-only:
    ap.add_argument("--scaler", help="StandardScaler pickle (sequential method only)")
    ap.add_argument("--labels", help="labels.json (sequential method only — static "
                    "methods use the Ekman 7 vocabulary)")
    ap.add_argument("--buffer", type=int, default=5,
                    help="Sliding window length for the sequential predictor")
    ap.add_argument("--interval_ms", type=int, default=400,
                    help="Time between sequential buffer samples")
    # Image-only:
    ap.add_argument("--backbone", default="mobilenet_v2",
                    help="Backbone for --method image (default: mobilenet_v2)")
    ap.add_argument("--image-size", type=int, default=224,
                    help="Input image size for --method image (default: 224)")
    # Shared:
    ap.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    ap.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Method: {args.method}   model: {args.model}   device: {device}")

    predictor = PREDICTORS[args.method](args, device)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        return 1

    print("Press 'q' to quit. Click the window first if it's not focused.")
    win_title = f"Emotion ({args.method}) - q to quit"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)   # named explicitly so it survives Qt sizing quirks

    first_frame_logged = False
    last_log = 0.0
    last_lines: List[str] = []
    frames = 0
    exit_reason = "user-quit"
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                exit_reason = "cap.read returned ok=False"
                break
            frames += 1
            if not first_frame_logged:
                print(f"First frame: shape={frame_bgr.shape} dtype={frame_bgr.dtype}")
                first_frame_logged = True

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            lines = predictor.predict(frame_bgr, frame_rgb)
            overlay_text(frame_bgr, lines)
            cv2.imshow(win_title, frame_bgr)

            # Rate-limited terminal echo so you can confirm predictions are
            # being produced even if the imshow window isn't visible.
            now = time.time()
            if now - last_log >= 1.0:
                print(f"frame {frames:5d}  " + "  ".join(lines))
                last_log = now
            last_lines = lines

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        predictor.close()
    print(f"Exited: {exit_reason}  (rendered {frames} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
