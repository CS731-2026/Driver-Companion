# run.py — CalmWheel entry point
import tkinter as tk
import threading
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from main.ui import CalmWheelUI
from main.main import CalmWheelLLM
from llm.interface import LLMInput

TIMELINE_START_DELAY = 10000
_current_emotion = ["neutral"]
_session_started = threading.Event()


def schedule_timeline(app, timeline):
    for i, item in enumerate(timeline):
        t       = TIMELINE_START_DELAY + int(item["t"]) if i > 0 else TIMELINE_START_DELAY
        emotion = item["emotion"]
        app.root.after(t, lambda e=emotion: app._switch_avatar(e))


def handle_output(app, output, fer_emotion):
    if output is None:
        return
    timeline = output.emotion_timeline
    if timeline:
        app.root.after(0, lambda e=timeline[0]["emotion"]: app._switch_avatar(e))
    app.root.after(0, lambda r=output.response_text, f=fer_emotion:
                   app.set_llm_response(r, fer_emotion=f))
    if len(timeline) > 1:
        app.root.after(0, lambda tl=timeline: schedule_timeline(app, tl))


# ── FER loop ─────────────────────────────────────────────────
def fer_loop(app: CalmWheelUI, llm: CalmWheelLLM, fer_args):
    from emotionrec.realtime_fer import run as fer_run

    app.root.after(0, app.disable_fallback_camera)

    def on_emotion(label: str, confidence: float):
        _current_emotion[0] = label
        app.root.after(0, lambda e=label, c=confidence: app.set_fer_emotion(e, c))
        print(f"[FER] {label} ({confidence:.0%})")

    def on_frame(frame):
        app.set_fer_frame(frame)

    def on_cap(cap):
        app.root.after(0, lambda c=cap: app.show_kinect_toggle(c.toggle_source))

    try:
        fer_run(fer_args, on_emotion=on_emotion, on_frame=on_frame, on_cap=on_cap)
    except FileNotFoundError as e:
        print(f"[FER] {e}")
        print("[FER] Disabled — use CLI to simulate.")


# ── Voice loop ───────────────────────────────────────────────
def voice_loop(app: CalmWheelUI, llm: CalmWheelLLM):
    from main.stt import listen_and_transcribe

    _session_started.wait()
    print("[Voice] Ready.")
    sleeping = False

    while True:
        llm._speaking.wait()

        try:
            text = listen_and_transcribe(language=None)
        except Exception as e:
            print(f"[Voice] Error: {e}"); continue

        if not text:
            continue

        cmd = text.strip().lower()

        if cmd == "mute":
            if not sleeping:
                sleeping = True
                print("[Voice] Sleeping — say 'unmute' to wake.\n")
                app.root.after(0, lambda: app.set_llm_response("[Sleeping]"))
            continue

        if cmd == "unmute":
            if sleeping:
                sleeping = False
                print("[Voice] Awake.\n")
                app.root.after(0, lambda: app.set_llm_response("[Awake]"))
            continue

        if sleeping:
            print(f"[Voice] Sleeping — ignored: '{text}'\n")
            continue

        emotion = _current_emotion[0]
        print(f"[Voice] '{text}'  +  emotion: {emotion}")
        app.root.after(0, lambda t=text: app.set_driver_speech(t))

        output = llm.process(LLMInput(emotion=emotion, speech_text=text))
        if output:
            print(f"  → {output.response_text}")
            print(f"  Timeline: {output.emotion_timeline}\n")
            handle_output(app, output, emotion)
        else:
            print("[Voice] Cooldown active\n")


# ── CLI loop ─────────────────────────────────────────────────
def cli_loop(app: CalmWheelUI, llm: CalmWheelLLM):
    _session_started.wait()
    print("\n" + "="*55)
    print("CalmWheel CLI")
    print("  <emotion>         → simulate FER emotion (no LLM)")
    print("  <emotion> <text>  → simulate speech + emotion → LLM")
    print("  reset | mute | quit")
    print("="*55 + "\n")

    VALID = {"angry","sad","fear","happy","neutral","disgust","surprise"}

    while True:
        try:
            raw = input(">> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not raw: continue
        if raw == "quit":   app.root.after(0, app.on_close); break
        if raw == "reset":  llm.new_trip(); continue
        if raw == "mute":
            llm.tts = not llm.tts
            print(f"[Voice {'ON' if llm.tts else 'OFF'}]\n"); continue

        parts   = raw.split(maxsplit=1)
        emotion = parts[0].lower()
        speech  = parts[1] if len(parts) > 1 else None

        if emotion not in VALID:
            print(f"Invalid. Choose: {VALID}\n"); continue

        _current_emotion[0] = emotion
        app.root.after(0, lambda e=emotion: app.set_fer_emotion(e, confidence=0.95))

        if not speech:
            print(f"[FER] Emotion updated: {emotion}\n")
            continue

        app.root.after(0, lambda s=speech: app.set_driver_speech(s))
        output = llm.process(LLMInput(emotion=emotion, speech_text=speech))
        if output is None:
            print("[Skipped] Cooldown\n")
        else:
            print(f"\nCalmWheel [{output.triggered_by} | {output.model_used} | {output.latency_ms}ms]:")
            print(f"  {output.response_text}")
            print(f"  Timeline: {output.emotion_timeline}\n")
            handle_output(app, output, emotion)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                   default="emotion-reco/runs/geo_static/checkpoints/best.pt",
                   help="FER model checkpoint (default: geo_static 122-landmark)")
    p.add_argument("--camera",       type=int,   default=0)
    p.add_argument("--conf",         type=float, default=0.3)
    p.add_argument("--no-bar",       action="store_true")
    p.add_argument("--no-fer",       action="store_true")
    p.add_argument("--landmarker",   default="face_landmarker.task")
    p.add_argument("--kinect",       action="store_true")
    p.add_argument("--kinect-index", type=int,   default=0, dest="kinect_index")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = tk.Tk()
    app  = CalmWheelUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    llm = CalmWheelLLM(mode="online", tts=True)

    def on_model_change(key):
        llm.model_key = key
        print(f"[Model] → {key}")
    app.on_model_change = on_model_change

    def on_start():
        _session_started.set()
        app.root.after(0, app.set_ready)
        app.root.after(0, lambda: app.set_status(
            "System ready  |  Model: Claude Haiku  |  Mode: online"))
    app.on_start = on_start

    if not args.no_fer:
        threading.Thread(target=fer_loop, args=(app, llm, args), daemon=True).start()
    else:
        print("[FER] Disabled.")

    threading.Thread(target=voice_loop, args=(app, llm), daemon=True).start()
    threading.Thread(target=cli_loop,   args=(app, llm), daemon=True).start()
    root.mainloop()
