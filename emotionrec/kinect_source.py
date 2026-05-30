"""
kinect_source.py — Kinect v2 IR + system webcam RGB hybrid source

Drop-in replacement for cv2.VideoCapture that switches between:
  - RGB from the system webcam (cv2.VideoCapture)
  - IR  from the Kinect v2 (active illumination)

Switching is either automatic (hysteresis on system-webcam luminance)
or manual via toggle_source().

Requires: pylibfreenect2  (Python bindings for libfreenect2)
  pip install pylibfreenect2
  or build from: https://github.com/r9y9/pylibfreenect2
"""

import numpy as np
import cv2
from typing import Optional, Tuple

# Luminance hysteresis thresholds (0-255 mean-pixel scale)
_LUX_LOW  = 50   # switch System RGB → Kinect IR when mean brightness drops below this
_LUX_HIGH = 70   # switch Kinect IR → System RGB when mean brightness rises above this

# Kinect v2 IR values are float32 in [0, ~65535].
# Active illumination scenes typically stay below 4096; clamp here.
_IR_CLIP = 4096.0


class KinectSource:
    """
    Hybrid frame source: system webcam RGB  ↔  Kinect v2 IR.

    Exposes the same read() / isOpened() / set() / release() interface as
    cv2.VideoCapture so it can be used as a drop-in replacement in
    realtime_fer.py without modifying the main capture loop.

    The `active_source` property reports which stream ("rgb" or "ir") was
    used for the most recent frame.
    """

    def __init__(self,
                 device_index: int = 0,
                 rgb_camera_index: int = 0,
                 lux_low:  float = _LUX_LOW,
                 lux_high: float = _LUX_HIGH):
        try:
            import pylibfreenect2 as freenect2
            self._fk2 = freenect2
        except ImportError as exc:
            raise ImportError(
                "pylibfreenect2 not installed.\n"
                "  pip install pylibfreenect2\n"
                "  or build from https://github.com/r9y9/pylibfreenect2"
            ) from exc

        # ── Kinect IR stream ───────────────────────────────────
        self._fn = freenect2.Freenect2()
        if self._fn.enumerateDevices() == 0:
            raise RuntimeError("No Kinect v2 device found. Check USB3 connection.")

        serial = (self._fn.getDefaultDeviceSerialNumber() if device_index == 0
                  else self._fn.getDeviceSerialNumber(device_index))
        self._device = self._fn.openDevice(serial)

        # Subscribe to IR only — RGB comes from the system webcam
        self._listener = freenect2.SyncMultiFrameListener(freenect2.FrameType.Ir)
        self._device.setIrAndDepthFrameListener(self._listener)
        self._device.start()

        # ── System webcam RGB stream ───────────────────────────
        self._rgb_cap = cv2.VideoCapture(rgb_camera_index)
        if not self._rgb_cap.isOpened():
            self._device.stop()
            self._device.close()
            raise RuntimeError(f"Cannot open system camera {rgb_camera_index}")
        self._rgb_cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self._rgb_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self._lux_low     = lux_low
        self._lux_high    = lux_high
        self._using_ir    = False
        self._manual_mode = False   # True = button override, skip auto-switch
        self._opened      = True
        self._last_lum: float = 128.0

        print(f"[Kinect] Opened device {serial} — IR source")
        print(f"[Kinect] System camera {rgb_camera_index} — RGB source")

    # ── Public ─────────────────────────────────────────────────

    @property
    def active_source(self) -> str:
        return "ir" if self._using_ir else "rgb"

    def toggle_source(self):
        """Manually flip between system RGB and Kinect IR, disabling auto-switch."""
        self._manual_mode = True
        self._using_ir = not self._using_ir
        mode = "IR" if self._using_ir else "RGB"
        src  = "Kinect IR" if self._using_ir else "System RGB"
        print(f"[Kinect] Manual override → {src}")
        return mode

    @property
    def last_luminance(self) -> float:
        return self._last_lum

    # ── cv2.VideoCapture-compatible interface ──────────────────

    def isOpened(self) -> bool:
        return self._opened

    def set(self, prop_id, value):
        # Kinect resolution is fixed — silently ignore CAP_PROP_FRAME_WIDTH etc.
        pass

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._opened:
            return False, None

        fk2 = self._fk2

        # Read system webcam (RGB)
        ret_rgb, rgb_bgr = self._rgb_cap.read()

        # Read Kinect IR
        frames = fk2.FrameMap()
        ok = self._listener.waitForNewFrame(frames, 5000)
        if not ok:
            return False, None

        try:
            ir_frame = frames[fk2.FrameType.Ir]
            ir_raw   = ir_frame.asarray(np.float32).copy()
        finally:
            self._listener.release(frames)

        # Measure ambient light from the system camera frame
        if ret_rgb and rgb_bgr is not None:
            lum = float(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY).mean())
        else:
            lum = self._last_lum
        self._last_lum = lum

        # Hysteresis switch (skipped when manually overridden via button)
        if not self._manual_mode:
            if self._using_ir:
                if lum > self._lux_high:
                    self._using_ir = False
                    print(f"[Kinect] Light OK ({lum:.0f}) → System RGB camera")
            else:
                if lum < self._lux_low:
                    self._using_ir = True
                    print(f"[Kinect] Low light ({lum:.0f}) → Kinect IR camera")

        if self._using_ir:
            return True, _ir_to_bgr(ir_raw)
        else:
            if not ret_rgb or rgb_bgr is None:
                return False, None
            return True, rgb_bgr

    def release(self):
        if self._opened:
            self._device.stop()
            self._device.close()
            self._rgb_cap.release()
            self._opened = False
            print("[Kinect] Device and system camera closed.")


# ── Helpers ────────────────────────────────────────────────────

def _ir_to_bgr(ir_raw: np.ndarray) -> np.ndarray:
    """Convert Kinect v2 IR float32 array to 8-bit BGR for MediaPipe."""
    clipped = np.clip(ir_raw, 0.0, _IR_CLIP)
    norm    = (clipped / _IR_CLIP * 255.0).astype(np.uint8)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
