# stt.py
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE       = 16000
SILENCE_THRESHOLD = 0.015  # volume below this = silence
SPEECH_THRESHOLD  = 0.025  # min avg volume to accept a recording as real speech
SILENCE_DURATION  = 1.5    # seconds of silence to stop recording
MAX_DURATION      = 10.0
MIN_DURATION      = 0.8    # ignore clips shorter than this

# Whisper sometimes transcribes silence/noise as these — discard them
NOISE_PHRASES = {
    "", ".", "..", "...", "you", "the", "a", "an",
    "thank you", "thanks", "bye", "goodbye",
    "uh", "um", "hmm", "hm", "ah", "oh",
    "okay", "ok", "yes", "no",
    "subtitles by", "subscribe", "www.",
}

# Single-word commands that bypass the noise/word-count filter
COMMAND_WORDS = {"mute", "unmute"}

MIN_WORD_COUNT = 2   # responses with fewer real words are discarded

print("Loading Whisper model (first time ~30s)...")
_model = WhisperModel("base", device="cpu", compute_type="int8")
print("Whisper ready.")


def _is_noise(text: str) -> bool:
    """Return True if text looks like a noise transcription."""
    clean = text.strip().lower().rstrip(".,!?")
    if clean in COMMAND_WORDS:
        return False
    if clean in NOISE_PHRASES:
        return True
    words = [w for w in clean.split() if len(w) > 1]
    if len(words) < MIN_WORD_COUNT:
        return True
    return False


def listen_and_transcribe(language=None) -> str:
    """
    Listen from microphone until silence, then transcribe.
    Returns transcribed text, or empty string if noise/silence detected.
    """
    recorded    = []
    is_speaking = False
    silence_sec = 0.0
    total_sec   = 0.0
    chunk_sec   = 0.032
    chunk_size  = int(SAMPLE_RATE * chunk_sec)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32", blocksize=chunk_size) as stream:
        while total_sec < MAX_DURATION:
            chunk, _ = stream.read(chunk_size)
            audio     = chunk[:, 0]
            volume    = np.sqrt(np.mean(audio ** 2))
            total_sec += chunk_sec

            if volume > SILENCE_THRESHOLD:
                if not is_speaking:
                    is_speaking = True
                    silence_sec = 0.0
                recorded.append(audio)
                silence_sec = 0.0
            else:
                if is_speaking:
                    recorded.append(audio)
                    silence_sec += chunk_sec
                    if silence_sec >= SILENCE_DURATION:
                        break

    if not recorded:
        return ""

    audio_np = np.concatenate(recorded)

    # Energy check — reject if recording is too quiet overall
    avg_volume = float(np.sqrt(np.mean(audio_np ** 2)))
    if avg_volume < SPEECH_THRESHOLD:
        print(f"[STT] Rejected: low energy ({avg_volume:.4f})")
        return ""

    # Duration check
    duration = len(audio_np) / SAMPLE_RATE
    if duration < MIN_DURATION:
        print(f"[STT] Rejected: too short ({duration:.2f}s)")
        return ""

    # Transcribe
    segments, _ = _model.transcribe(audio_np, language=language, beam_size=5)
    text = " ".join(s.text for s in segments).strip()

    # Noise phrase check
    if _is_noise(text):
        print(f"[STT] Rejected noise: '{text}'")
        return ""

    print(f"[STT] Transcribed: '{text}'")
    return text
