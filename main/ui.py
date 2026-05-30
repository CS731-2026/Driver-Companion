# ui.py
# CalmWheel — Main UI
# Layout: left = live camera feed | center = GIF avatar | right = conversation log

import tkinter as tk
from tkinter import font as tkfont
import threading
import time
import os
from PIL import Image, ImageTk, ImageSequence
import cv2
import numpy as np

GIF_DIR = r"./gifs"

GIF_PATHS = {
    "neutral":  os.path.join(GIF_DIR, "neutral.gif"),
    "happy":    os.path.join(GIF_DIR, "happy.gif"),
    "angry":    os.path.join(GIF_DIR, "angry.gif"),
    "sad":      os.path.join(GIF_DIR, "sad.gif"),
    "fear":     os.path.join(GIF_DIR, "fear.gif"),
    "disgust":  os.path.join(GIF_DIR, "disgust.gif"),
    "surprise": os.path.join(GIF_DIR, "surprise.gif"),
}

EMOTION_COLORS = {
    "neutral":  "#4A90D9",
    "happy":    "#F5C542",
    "angry":    "#E84040",
    "sad":      "#6B8CBA",
    "fear":     "#9B59B6",
    "disgust":  "#27AE60",
    "surprise": "#F39C12",
}

EMOTION_LABELS = {
    "neutral":  "NEUTRAL",
    "happy":    "HAPPY",
    "angry":    "ANGRY",
    "sad":      "SAD",
    "fear":     "FEAR",
    "disgust":  "DISGUST",
    "surprise": "SURPRISE",
}

# Models available for switching
AVAILABLE_MODELS = {
    "Claude Haiku":  "claude-haiku",
    "GPT-4o Mini":   "gpt-4o-mini",
    "Llama 70B":     "llama-70b",
    "Gemma 26B":     "gemma-26b",
}

WIN_W, WIN_H = 1200, 720
CAM_W, CAM_H = 480, 360
AVATAR_W     = 300
AVATAR_H     = 300
LOG_W        = 300


class GifPlayer:
    def __init__(self, label: tk.Label, path: str, size=(AVATAR_W, AVATAR_H)):
        self.label  = label
        self.frames = []
        self.delays = []
        self.idx    = 0
        self._job   = None
        self._load(path, size)

    def _load(self, path: str, size):
        try:
            gif = Image.open(path)
            for frame in ImageSequence.Iterator(gif):
                img = frame.convert("RGBA").resize(size, Image.LANCZOS)
                self.frames.append(ImageTk.PhotoImage(img))
                self.delays.append(frame.info.get("duration", 80))
            print(f"[GIF] Loaded {os.path.basename(path)} ({len(self.frames)} frames)")
        except Exception as e:
            print(f"[GIF] Could not load {path}: {e}")

    def play(self):
        if not self.frames:
            return
        self.label.configure(image=self.frames[self.idx])
        delay = self.delays[self.idx]
        self.idx = (self.idx + 1) % len(self.frames)
        self._job = self.label.after(delay, self.play)

    def stop(self):
        if self._job:
            self.label.after_cancel(self._job)
            self._job = None
        self.idx = 0


class CalmWheelUI:
    def __init__(self, root: tk.Tk):
        self.root            = root
        self.fer_emotion     = "neutral"
        self.display_emotion = "neutral"
        self.gif_players     = {}
        self.active_player   = None
        self.cap             = None
        self._cam_running    = False
        self.on_model_change = None   # callback: fn(model_key: str)
        self.on_start        = None   # callback: fn()

        self._build_window()
        self._build_layout()
        self._load_gifs()
        self._start_camera()
        self._switch_avatar("neutral")

    def _build_window(self):
        self.root.title("CalmWheel — Driver Companion")
        self.root.geometry(f"{WIN_W}x{WIN_H}")
        self.root.configure(bg="#0D1117")
        self.root.resizable(False, False)

        self.font_title   = tkfont.Font(family="Courier New", size=11, weight="bold")
        self.font_label   = tkfont.Font(family="Courier New", size=9)
        self.font_emotion = tkfont.Font(family="Courier New", size=16, weight="bold")
        self.font_log     = tkfont.Font(family="Courier New", size=9)
        self.font_small   = tkfont.Font(family="Courier New", size=8)

    def _build_layout(self):
        BG    = "#0D1117"
        PANEL = "#161B22"
        BORDER= "#30363D"
        TEAL  = "#1D9E75"
        TEXT  = "#E6EDF3"
        MUTED = "#8B949E"

        # ── Top bar ──
        top = tk.Frame(self.root, bg=BG, height=44)
        top.pack(fill="x")
        top.pack_propagate(False)
        tk.Label(top, text="● CALMWHEEL  //  DRIVER EMOTION COMPANION",
                 bg=BG, fg=TEAL, font=self.font_title).pack(side="left", padx=20, pady=10)
        self.status_dot = tk.Label(top, text="◉ LIVE", bg=BG, fg=TEAL, font=self.font_small)
        self.status_dot.pack(side="right", padx=20)
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=12, pady=12)

        # ── LEFT: camera ──
        left = tk.Frame(main, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        left.pack(side="left", fill="y", padx=(0, 8))

        tk.Label(left, text="CAMERA FEED", bg=PANEL, fg=MUTED,
                 font=self.font_small).pack(pady=(8, 4))
        self.cam_label = tk.Label(left, bg="#000000", width=CAM_W, height=CAM_H)
        self.cam_label.pack(padx=8, pady=(0, 8))

        fer_frame = tk.Frame(left, bg=PANEL)
        fer_frame.pack(pady=(0, 4))
        tk.Label(fer_frame, text="DETECTED:", bg=PANEL, fg=MUTED,
                 font=self.font_small).pack(side="left", padx=(8, 4))
        self.fer_badge = tk.Label(fer_frame, text="NEUTRAL",
                                  bg=EMOTION_COLORS["neutral"], fg="#FFFFFF",
                                  font=self.font_label, padx=8, pady=2)
        self.fer_badge.pack(side="left")

        conf_frame = tk.Frame(left, bg=PANEL)
        conf_frame.pack(pady=(0, 8), padx=8, fill="x")
        tk.Label(conf_frame, text="CONFIDENCE", bg=PANEL, fg=MUTED,
                 font=self.font_small).pack(anchor="w")
        bar_bg = tk.Frame(conf_frame, bg=BORDER, height=6)
        bar_bg.pack(fill="x", pady=2)
        self.conf_bar = tk.Frame(bar_bg, bg=TEAL, height=6)
        self.conf_bar.place(x=0, y=0, relwidth=0.85, height=6)
        self.conf_label = tk.Label(conf_frame, text="85%", bg=PANEL, fg=TEAL,
                                   font=self.font_small)
        self.conf_label.pack(anchor="e")

        # Kinect RGB/IR toggle (hidden until kinect is active)
        self._kinect_toggle_cb  = None
        self._kinect_btn_text   = tk.StringVar(value="CAM: SYS RGB")
        self._kinect_btn_frame  = tk.Frame(left, bg=PANEL)
        self._kinect_btn        = tk.Button(
            self._kinect_btn_frame,
            textvariable=self._kinect_btn_text,
            font=self.font_small,
            bg=BORDER, fg="#FFFFFF",
            activebackground=TEAL, activeforeground="#FFFFFF",
            relief="flat", padx=8, pady=4, cursor="hand2",
            command=self._on_kinect_toggle,
        )
        self._kinect_btn.pack(fill="x")
        # not packed yet — shown only when kinect is connected

        # ── CENTER: avatar ──
        center = tk.Frame(main, bg=BG)
        center.pack(side="left", fill="both", expand=True, padx=8)

        tk.Label(center, text="COMPANION", bg=BG, fg=MUTED,
                 font=self.font_small).pack(pady=(0, 6))

        avatar_outer = tk.Frame(center, bg=PANEL,
                                highlightthickness=1, highlightbackground=BORDER)
        avatar_outer.pack()
        self.avatar_label = tk.Label(avatar_outer, bg="#000000",
                                     width=AVATAR_W, height=AVATAR_H)
        self.avatar_label.pack(padx=4, pady=4)

        self.avatar_emotion_text = tk.Label(center, text="NEUTRAL",
                                            bg=BG, fg=TEAL, font=self.font_emotion)
        self.avatar_emotion_text.pack(pady=(10, 2))

        self.source_label = tk.Label(center, text="— FER", bg=BG, fg=MUTED,
                                     font=self.font_small)
        self.source_label.pack(pady=(0, 4))

        resp_outer = tk.Frame(center, bg=PANEL,
                              highlightthickness=1, highlightbackground=BORDER)
        resp_outer.pack(fill="x", pady=(4, 0), padx=4)
        tk.Label(resp_outer, text="CALMWHEEL SAYS", bg=PANEL, fg=MUTED,
                 font=self.font_small).pack(anchor="w", padx=8, pady=(6, 2))
        self.response_label = tk.Label(resp_outer, text="Press START to begin session.",
                                       bg=PANEL, fg=TEXT, font=self.font_log,
                                       wraplength=360, justify="left", anchor="w")
        self.response_label.pack(fill="x", padx=8, pady=(0, 10))

        # ── Start button ──
        self._start_btn = tk.Button(
            center,
            text="▶  START SESSION",
            font=self.font_title,
            bg=TEAL, fg="#FFFFFF",
            activebackground="#157A5C",
            activeforeground="#FFFFFF",
            relief="flat",
            padx=20, pady=10,
            cursor="hand2",
            command=self._on_start_click,
        )
        self._start_btn.pack(pady=(12, 0))

        # ── RIGHT: log + model switcher ──
        right = tk.Frame(main, bg=PANEL, width=LOG_W,
                         highlightthickness=1, highlightbackground=BORDER)
        right.pack(side="right", fill="y", padx=(8, 0))
        right.pack_propagate(False)

        tk.Label(right, text="CONVERSATION LOG", bg=PANEL, fg=MUTED,
                 font=self.font_small).pack(pady=(8, 4), padx=8, anchor="w")
        tk.Frame(right, bg=BORDER, height=1).pack(fill="x", padx=8)

        self.log_text = tk.Text(right, bg=PANEL, fg=TEXT, font=self.font_log,
                                wrap="word", state="disabled", bd=0,
                                highlightthickness=0, padx=8, pady=6)
        self.log_text.pack(fill="both", expand=True, pady=(4, 0))
        self.log_text.tag_configure("emotion", foreground=TEAL,      font=self.font_small)
        self.log_text.tag_configure("driver",  foreground="#8B949E",  font=self.font_log)
        self.log_text.tag_configure("ai",      foreground=TEXT,       font=self.font_log)
        self.log_text.tag_configure("time",    foreground="#484F58",   font=self.font_small)

        # ── Model switcher (bottom of right panel) ──
        tk.Frame(right, bg=BORDER, height=1).pack(fill="x", padx=8, pady=(4, 0))
        model_frame = tk.Frame(right, bg=PANEL)
        model_frame.pack(fill="x", padx=8, pady=6)

        tk.Label(model_frame, text="MODEL", bg=PANEL, fg=MUTED,
                 font=self.font_small).pack(anchor="w")

        self._selected_model = tk.StringVar(value="Claude Haiku")
        self._model_buttons  = {}

        btn_frame = tk.Frame(model_frame, bg=PANEL)
        btn_frame.pack(fill="x", pady=(4, 0))

        for i, (label, key) in enumerate(AVAILABLE_MODELS.items()):
            btn = tk.Button(
                btn_frame,
                text=label,
                font=self.font_small,
                bg=TEAL if label == "Claude Haiku" else BORDER,
                fg="#FFFFFF",
                activebackground=TEAL,
                activeforeground="#FFFFFF",
                relief="flat",
                padx=4, pady=3,
                cursor="hand2",
                command=lambda l=label, k=key: self._on_model_click(l, k)
            )
            btn.grid(row=i//2, column=i%2, padx=2, pady=2, sticky="ew")
            btn_frame.grid_columnconfigure(i%2, weight=1)
            self._model_buttons[label] = btn

        # ── Bottom status bar ──
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")
        bottom = tk.Frame(self.root, bg=BG, height=28)
        bottom.pack(fill="x")
        bottom.pack_propagate(False)
        self.bottom_status = tk.Label(
            bottom,
            text="System ready  |  Model: Claude Haiku  |  Mode: online",
            bg=BG, fg="#484F58", font=self.font_small
        )
        self.bottom_status.pack(side="left", padx=16, pady=6)

    def _on_model_click(self, label: str, key: str):
        """Handle model button click — update UI and notify run.py."""
        TEAL   = "#1D9E75"
        BORDER = "#30363D"

        # Update button highlights
        for lbl, btn in self._model_buttons.items():
            btn.configure(bg=TEAL if lbl == label else BORDER)

        self._selected_model.set(label)
        self.bottom_status.configure(
            text=f"System ready  |  Model: {label}  |  Mode: online"
        )
        self._log_model_switch(label)

        # Notify run.py via callback
        if self.on_model_change:
            self.on_model_change(key)

    def _load_gifs(self):
        os.makedirs(GIF_DIR, exist_ok=True)
        for emotion, path in GIF_PATHS.items():
            if os.path.exists(path):
                player = GifPlayer(self.avatar_label, path)
                if player.frames:
                    self.gif_players[emotion] = player
            else:
                print(f"[GIF] Missing: {path}")

    def _switch_avatar(self, emotion: str):
        if self.active_player:
            self.active_player.stop()
            self.active_player = None

        color = EMOTION_COLORS.get(emotion, "#4A90D9")
        label = EMOTION_LABELS.get(emotion, emotion.upper())

        if emotion in self.gif_players:
            self.active_player = self.gif_players[emotion]
            self.active_player.idx = 0
            self.avatar_label.configure(text="", bg="#000000")
            self.active_player.play()
        else:
            self.avatar_label.configure(
                image="", bg=color, text=label, fg="white", font=self.font_emotion
            )

        self.avatar_emotion_text.configure(text=label, fg=color)

    def show_kinect_toggle(self, toggle_cb):
        """Show the RGB/IR toggle button and wire it to toggle_cb."""
        self._kinect_toggle_cb = toggle_cb
        self._kinect_btn_frame.pack(pady=(0, 8), padx=8, fill="x")

    def _on_kinect_toggle(self):
        if not self._kinect_toggle_cb:
            return
        TEAL   = "#1D9E75"
        BORDER = "#30363D"
        mode = self._kinect_toggle_cb()          # returns "RGB" or "IR"
        label = "SYS RGB" if mode == "RGB" else "KINECT IR"
        self._kinect_btn_text.set(f"CAM: {label}")
        self._kinect_btn.configure(bg=TEAL if mode == "IR" else BORDER)

    # ── Public API ───────────────────────────────────────────
    def set_fer_emotion(self, emotion: str, confidence: float = 1.0):
        emotion = emotion.lower()
        if emotion not in GIF_PATHS:
            emotion = "neutral"
        self.fer_emotion = emotion
        color = EMOTION_COLORS.get(emotion, "#4A90D9")
        self.fer_badge.configure(
            text=EMOTION_LABELS.get(emotion, emotion.upper()), bg=color)
        self.conf_bar.place(relwidth=confidence)
        self.conf_label.configure(text=f"{int(confidence * 100)}%")
        self._log_emotion(emotion, source="FER")

    def set_llm_response(self, response_text: str,
                         suggested_emotion: str = None,
                         fer_emotion: str = None):
        display = (suggested_emotion or fer_emotion or self.fer_emotion).lower()
        if display not in GIF_PATHS:
            display = "neutral"
        self.display_emotion = display
        source = "LLM" if suggested_emotion else "FER"
        self._switch_avatar(display)
        self.source_label.configure(text=f"— {source}")
        self.response_label.configure(text=response_text)
        self._log_ai(response_text, display)

    def set_driver_speech(self, text: str):
        self._log_driver(text)

    def set_status(self, text: str):
        self.bottom_status.configure(text=text)

    # ── Camera ───────────────────────────────────────────────
    def _start_camera(self):
        """Camera feed comes from FER via set_fer_frame(). No fallback camera opened."""
        self._fer_providing_frames = False
        self._cam_running          = False
        self.cap                   = None
        self.cam_label.configure(text="Waiting for\ncamera feed...",
                                 fg="#484F58", font=self.font_label)

    def disable_fallback_camera(self):
        """No-op — kept for compatibility."""
        pass

    def _camera_loop(self):
        pass

    def set_fer_frame(self, bgr_frame):
        """
        Called by FER every frame with annotated BGR image.
        Displays it in the UI camera panel (includes face boxes + bars).
        Letterboxes to preserve aspect ratio.
        """
        self._fer_providing_frames = True
        h, w = bgr_frame.shape[:2]
        scale   = min(CAM_W / w, CAM_H / h)
        new_w   = int(w * scale)
        new_h   = int(h * scale)
        resized = cv2.resize(bgr_frame, (new_w, new_h))
        canvas  = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
        x_off   = (CAM_W - new_w) // 2
        y_off   = (CAM_H - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        frame = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img   = ImageTk.PhotoImage(Image.fromarray(frame))
        self.cam_label.configure(image=img)
        self.cam_label.image = img

    # ── Logging ──────────────────────────────────────────────
    def _log_emotion(self, emotion: str, source: str = "FER"):
        ts = time.strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"\n[{ts}] ", "time")
        self.log_text.insert("end", f"{source} → {emotion.upper()}\n", "emotion")
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    def _log_driver(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{ts}] ", "time")
        self.log_text.insert("end", f"Driver: {text}\n", "driver")
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    def _log_ai(self, text: str, emotion: str = ""):
        ts = time.strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{ts}] ", "time")
        self.log_text.insert("end", f"AI [{emotion}]: {text}\n\n", "ai")
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    def _log_model_switch(self, model_name: str):
        ts = time.strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{ts}] ", "time")
        self.log_text.insert("end", f"Model → {model_name}\n", "emotion")
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    def _on_start_click(self):
        self._start_btn.configure(text="⏳  WARMING UP...", bg="#484F58", state="disabled")
        self.response_label.configure(text="Loading model into memory...")
        if self.on_start:
            self.on_start()

    def set_ready(self):
        self._start_btn.configure(text="◉  RUNNING", bg="#27AE60")
        self.response_label.configure(text="Ready.")

    def on_close(self):
        self._cam_running = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()
