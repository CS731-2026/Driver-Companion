"""Kinect v2 dual-stream demo — system RGB camera + Kinect IR, with optional emotion recognition.

Shows the system webcam RGB feed and the Kinect v2 IR feed side-by-side in a
single window and runs any of the project's trained emotion models on *both*
streams simultaneously, demonstrating that the landmark-based pipeline works
regardless of whether the input is visible-light or near-infrared.

Requirements:
    pip install pylibfreenect2
    (libfreenect2 must be installed system-wide first)

Usage — viewer only (no model):
    python -m implementation.kinect_demo

Usage — with a model:
    python -m implementation.kinect_demo --method geo_static \
        --model runs/geo_static_ckplus/checkpoints/best.pt

    python -m implementation.kinect_demo --method sequential \
        --model runs/ckplus/fold_best.pt \
        --scaler runs/ckplus/scaler.pkl --labels runs/ckplus/labels.json

Supported --method values are the same as realtime.py:
    sequential | emonet | blendshape | geo_static | image

IR pipeline note:
    MediaPipe expects an 8-bit 3-channel image.  The raw Kinect IR frame is
    float32 in [0, 65535].  We normalise to uint8 by clipping at the 99th
    percentile and then stack into a 3-channel grey image.  Face detection and
    landmark extraction work reliably under good lighting.

Key bindings:
    q — quit
    s — save a snapshot of the current composite frame as kinect_snapshot.png
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── Kinect v2 import (graceful fallback to webcam for development) ────────────
try:
    import pylibfreenect2 as fn2
    from pylibfreenect2 import (
        Freenect2,
        SyncMultiFrameListener,
        FrameType,
        Frame,
    )
    _HAS_KINECT = True
except ImportError:
    _HAS_KINECT = False

# ── project imports ───────────────────────────────────────────────────────────
# Reuse the predictor classes from realtime.py unchanged.
from implementation.realtime import (
    PREDICTORS,
    overlay_text,
    UNIFIED_EMOTIONS,
)

# Display geometry
_DISPLAY_W = 640   # width of each pane
_DISPLAY_H = 480   # height of each pane
_GAP = 4           # pixel gap between the two panes


# ─────────────────────────────────────────────────────────────────────────────
# IR normalisation
# ─────────────────────────────────────────────────────────────────────────────

def ir_to_uint8_bgr(ir_frame: np.ndarray) -> np.ndarray:
    """Convert a float32 IR frame [0, 65535] to a displayable BGR uint8 image."""
    p99 = float(np.percentile(ir_frame, 99)) or 1.0
    clipped = np.clip(ir_frame / p99, 0.0, 1.0)
    grey8 = (clipped * 255).astype(np.uint8)
    return cv2.cvtColor(grey8, cv2.COLOR_GRAY2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# Kinect v2 capture context
# ─────────────────────────────────────────────────────────────────────────────

class KinectIRCapture:
    """Pulls only the IR stream from the Kinect v2 (no Kinect RGB camera used)."""

    def __init__(self) -> None:
        self._freenect = Freenect2()
        n = self._freenect.enumerateDevices()
        if n == 0:
            raise RuntimeError("No Kinect v2 device found — is it plugged in?")

        serial = self._freenect.getDeviceSerialNumber(0)
        self._device = self._freenect.openDevice(serial)

        self._listener = SyncMultiFrameListener(FrameType.Ir)
        self._device.setIrAndDepthFrameListener(self._listener)
        self._device.start()
        print(f"Kinect v2 opened (serial {serial}) — IR stream only")

    def read_ir(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return (ok, ir_bgr)."""
        frames = self._listener.waitForNewFrame()
        try:
            ir_raw = frames["ir"].asarray(np.float32)   # (424, 512) float32
            ir_bgr = cv2.flip(ir_to_uint8_bgr(ir_raw), 1)
        except Exception:
            return False, None
        finally:
            self._listener.release(frames)
        return True, ir_bgr

    def close(self) -> None:
        self._device.stop()
        self._device.close()


class SystemCameraCapture:
    """System webcam — provides the RGB stream."""

    def __init__(self, index: int) -> None:
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open system camera index {index}")
        print(f"System camera opened (index {index})")

    def read_rgb(self) -> Tuple[bool, Optional[np.ndarray]]:
        ok, frame = self._cap.read()
        if not ok:
            return False, None
        return True, frame

    def close(self) -> None:
        self._cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: single webcam simulates both streams (no Kinect present)
# ─────────────────────────────────────────────────────────────────────────────

class WebcamFallbackCapture:
    """Single webcam used when no Kinect is connected (development/testing only).

    The same frame serves as 'RGB'; a greyscale conversion of it stands in for
    the missing IR feed so the rest of the pipeline can still be exercised.
    """

    def __init__(self, index: int) -> None:
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open webcam index {index}")
        print(f"[fallback] Using webcam index {index} for both streams (no Kinect detected)")

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        ok, frame = self._cap.read()
        if not ok:
            return False, None, None
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ir_bgr = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
        return True, frame, ir_bgr

    def close(self) -> None:
        self._cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# Composite frame builder
# ─────────────────────────────────────────────────────────────────────────────

def make_composite(left: np.ndarray, right: np.ndarray,
                   left_label: str, right_label: str) -> np.ndarray:
    """Resize both panes to _DISPLAY_W x _DISPLAY_H and stitch side-by-side."""
    l = cv2.resize(left,  (_DISPLAY_W, _DISPLAY_H))
    r = cv2.resize(right, (_DISPLAY_W, _DISPLAY_H))
    gap = np.zeros((_DISPLAY_H, _GAP, 3), dtype=np.uint8)
    composite = np.concatenate([l, gap, r], axis=1)

    # Pane header labels
    for i, txt in enumerate([left_label, right_label]):
        x_off = i * (_DISPLAY_W + _GAP)
        cv2.rectangle(composite, (x_off, 0), (x_off + _DISPLAY_W, 24), (30, 30, 30), -1)
        cv2.putText(composite, txt, (x_off + 6, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return composite


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--method", choices=list(PREDICTORS),
                    help="Emotion model to run on both streams (optional)")
    ap.add_argument("--model", help="Path to the .pt checkpoint")
    # sequential-only:
    ap.add_argument("--scaler", help="StandardScaler pickle (sequential only)")
    ap.add_argument("--labels", help="labels.json (sequential only)")
    ap.add_argument("--buffer", type=int, default=5)
    ap.add_argument("--interval_ms", type=int, default=400)
    # image-only:
    ap.add_argument("--backbone", default="mobilenet_v2")
    ap.add_argument("--image-size", type=int, default=224)
    # Kinect / fallback:
    ap.add_argument("--camera", type=int, default=0,
                    help="Webcam index used when no Kinect is detected")
    ap.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = ap.parse_args()

    if args.method and not args.model:
        ap.error("--model is required when --method is given")

    # ── open camera sources ──────────────────────────────────────────────────
    # RGB always comes from the system camera; IR from the Kinect v2.
    rgb_source = SystemCameraCapture(args.camera)
    ir_source: Optional[object] = None

    if _HAS_KINECT:
        try:
            ir_source = KinectIRCapture()
            source_name = f"System camera (RGB) + Kinect v2 (IR)"
        except RuntimeError as e:
            print(f"[warn] {e} — IR pane will show greyscale from system camera")
            source_name = f"System camera (RGB, IR fallback)"
    else:
        print("[warn] pylibfreenect2 not installed — IR pane will show greyscale from system camera")
        source_name = f"System camera (RGB, IR fallback)"

    # ── load predictors (one instance per stream) ────────────────────────────
    import torch
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    predictor_rgb: Optional[object] = None
    predictor_ir:  Optional[object] = None

    if args.method:
        print(f"Loading '{args.method}' model from {args.model} on {device} ...")
        predictor_rgb = PREDICTORS[args.method](args, device)
        predictor_ir  = PREDICTORS[args.method](args, device)
        print("Models ready.")

    # ── display loop ─────────────────────────────────────────────────────────
    win = "Kinect v2  |  RGB (left)   IR (right)   |   q=quit  s=snapshot"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, _DISPLAY_W * 2 + _GAP, _DISPLAY_H)

    frames = 0
    last_log = 0.0
    print(f"Source: {source_name}")
    print("Press 'q' to quit, 's' to save a snapshot.")

    try:
        while True:
            ok_rgb, color_bgr = rgb_source.read_rgb()
            if not ok_rgb:
                print("RGB stream ended or read error — exiting.")
                break

            if ir_source is not None:
                ok_ir, ir_bgr = ir_source.read_ir()
                if not ok_ir:
                    print("IR stream read error — exiting.")
                    break
            else:
                # Fallback: derive IR-like frame from the RGB feed
                grey = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
                ir_bgr = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

            frames += 1

            # Run models on their respective streams.
            rgb_lines: List[str] = []
            ir_lines:  List[str] = []

            if predictor_rgb is not None:
                color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
                rgb_lines = predictor_rgb.predict(color_bgr, color_rgb)

            if predictor_ir is not None:
                # IR is already BGR (grey), convert to RGB for MediaPipe.
                ir_rgb = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2RGB)
                ir_lines = predictor_ir.predict(ir_bgr, ir_rgb)

            overlay_text(color_bgr, rgb_lines, origin=(10, 38))
            overlay_text(ir_bgr,    ir_lines,  origin=(10, 38))

            left_header  = "RGB" + (f"  —  {rgb_lines[0]}" if rgb_lines else "")
            right_header = "IR"  + (f"  —  {ir_lines[0]}"  if ir_lines  else "")
            composite = make_composite(color_bgr, ir_bgr, left_header, right_header)
            cv2.imshow(win, composite)

            now = time.time()
            if now - last_log >= 2.0:
                info = f"frame {frames:5d}"
                if rgb_lines:
                    info += f"  RGB→ {rgb_lines[0]}"
                if ir_lines:
                    info += f"  IR→ {ir_lines[0]}"
                print(info)
                last_log = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                out = "kinect_snapshot.png"
                cv2.imwrite(out, composite)
                print(f"Saved snapshot to {out}")

    finally:
        rgb_source.close()
        if ir_source is not None:
            ir_source.close()
        if predictor_rgb is not None:
            predictor_rgb.close()
        if predictor_ir is not None:
            predictor_ir.close()
        cv2.destroyAllWindows()

    print(f"Done — rendered {frames} frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
