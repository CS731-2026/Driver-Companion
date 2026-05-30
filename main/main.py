# main/main.py
import re
import json
import time
import asyncio
import os
import threading
from typing import Optional, List, Dict

import edge_tts
import pygame

from llm.interface import LLMInput, LLMOutput
from llm.prompt_builder import build_messages
from llm.llm_client import call_llm
from llm.trigger_control import TriggerController
from llm.conversation import ConversationMemory

VOICE    = "en-US-AriaNeural"
TTS_FILE = "response.mp3"
VALID_EMOTIONS = {"neutral","happy","sad","angry","fear","disgust","surprise"}

async def _synthesize(text: str):
    communicate = edge_tts.Communicate(text, voice=VOICE)
    await communicate.save(TTS_FILE)

def speak(text: str):
    try:
        asyncio.run(_synthesize(text))
        pygame.mixer.init()
        pygame.mixer.music.load(TTS_FILE)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.unload()
    except Exception as e:
        print(f"[TTS] Error: {e}")

def speak_async(text: str):
    threading.Thread(target=speak, args=(text,), daemon=True).start()


def parse_response(raw: str) -> tuple[str, Optional[List[Dict]]]:
    json_match = re.search(r'\{.*"emotion_timeline".*\}', raw, re.DOTALL)
    if json_match:
        text = raw[:json_match.start()].strip()
        try:
            data     = json.loads(json_match.group())
            timeline = data.get("emotion_timeline", [])
            valid_timeline = [
                item for item in timeline
                if isinstance(item.get("t"), (int, float))
                and item.get("emotion") in VALID_EMOTIONS
            ]
            if valid_timeline:
                return text, valid_timeline
        except json.JSONDecodeError:
            pass
    return raw.strip(), None


class CalmWheelLLM:
    def __init__(self, mode: str = "online", model_key: str = None, tts: bool = True):
        self.mode      = mode
        self.model_key = model_key
        self.tts       = tts
        self.trigger   = TriggerController()
        self.memory    = ConversationMemory(max_turns=6)
        self._speaking = threading.Event()
        self._speaking.set()

    def speak_async(self, text: str):
        def _run():
            self._speaking.clear()
            speak(text)
            self._speaking.set()
        threading.Thread(target=_run, daemon=True).start()

    def process(self, inp: LLMInput) -> Optional[LLMOutput]:
        should, reason = self.trigger.should_trigger(inp.emotion, inp.speech_text)
        if not should:
            return None

        messages = build_messages(
            emotion=inp.emotion,
            speech_text=inp.speech_text,
            trigger_reason=reason,
            conversation_history=self.memory.get_history(),
        )

        start = time.time()
        try:
            raw_response, model_used = call_llm(
                messages=messages,
                model_key=self.model_key,
                mode=self.mode,
            )
        except Exception as e:
            print(f"[CalmWheelLLM] Call failed: {e}")
            return None

        latency_ms = (time.time() - start) * 1000
        response_text, emotion_timeline = parse_response(raw_response)

        if not emotion_timeline:
            emotion_timeline = [{"t": 0, "emotion": inp.emotion}]

        self.trigger.record_trigger(inp.emotion)
        if inp.speech_text:
            self.memory.add_user(inp.speech_text, emotion=inp.emotion)
        self.memory.add_assistant(response_text)

        if self.tts:
            self.speak_async(response_text)

        return LLMOutput(
            response_text=response_text,
            emotion_in=inp.emotion,
            suggested_emotion=emotion_timeline[0]["emotion"] if emotion_timeline else None,
            triggered_by=reason,
            model_used=model_used,
            latency_ms=round(latency_ms, 1),
            emotion_timeline=emotion_timeline,
        )

    def new_trip(self):
        self.trigger.reset()
        self.memory.reset()
        print("[CalmWheelLLM] New trip started — memory cleared")


if __name__ == "__main__":
    print("="*55)
    print("CalmWheel LLM — Interactive Test")
    print("="*55)
    print("Input: <emotion> [text]  |  reset  |  mute  |  quit\n")

    llm = CalmWheelLLM(mode="online", tts=True)

    while True:
        try:
            raw = input(">> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if raw == "quit":   break
        if raw == "reset":  llm.new_trip(); continue
        if raw == "mute":
            llm.tts = not llm.tts
            print(f"[Voice {'ON' if llm.tts else 'OFF'}]\n")
            continue
        if not raw: continue

        parts   = raw.split(maxsplit=1)
        emotion = parts[0].lower()
        speech  = parts[1] if len(parts) > 1 else None

        if emotion not in VALID_EMOTIONS:
            print(f"Invalid. Choose: {VALID_EMOTIONS}\n"); continue

        output = llm.process(LLMInput(emotion=emotion, speech_text=speech))
        if output is None:
            print("[Skipped] Cooldown active\n")
        else:
            print(f"\nCalmWheel [{output.triggered_by} | {output.model_used} | {output.latency_ms}ms]:")
            print(f"  {output.response_text}")
            print(f"  Timeline: {output.emotion_timeline}\n")
