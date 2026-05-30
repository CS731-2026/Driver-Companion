# interface.py
from dataclasses import dataclass, field
from typing import Optional, Literal, List, Dict

EmotionLabel = Literal["angry", "sad", "fear", "happy", "neutral", "disgust", "surprise"]
TriggerReason = Literal["speech", "emotion_intervention", "companionship"]

@dataclass
class LLMInput:
    emotion: EmotionLabel
    speech_text: Optional[str] = None
    confidence: float = 1.0

@dataclass
class LLMOutput:
    response_text: str
    emotion_in: EmotionLabel
    suggested_emotion: Optional[EmotionLabel]
    triggered_by: TriggerReason
    model_used: str
    latency_ms: float
    emotion_timeline: List[Dict] = field(default_factory=list)
    # emotion_timeline format:
    # [{"t": 0, "emotion": "angry"}, {"t": 2500, "emotion": "neutral"}]
    # t = milliseconds from start of TTS playback
