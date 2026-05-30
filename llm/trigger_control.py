# trigger_control.py
# 决定什么时候触发LLM，防止过于频繁打扰司机

import time
from typing import Optional


# 不同情绪下的冷却时间（秒）
# 危险情绪冷却短（需要更及时介入），平稳情绪冷却长
COOLDOWN_BY_EMOTION = {
    "angry":    20,
    "fear":     15,
    "sad":      30,
    "disgust":  25,
    "surprise": 15,
    "happy":    60,
    "neutral":  90,   # 平稳状态只作为陪伴，不要太频繁
}

# 需要主动介入的情绪（司机没说话也触发）
INTERVENTION_EMOTIONS = {"angry", "fear", "sad", "disgust", "surprise"}

# 司机主动说话时，不受冷却时间限制，直接响应
SPEECH_ALWAYS_TRIGGERS = True


class TriggerController:

    def __init__(self):
        self.last_trigger_time: float = 0
        self.last_emotion: Optional[str] = None
        self.trigger_count: int = 0   # 统计用

    def should_trigger(self, emotion: str, speech_text: Optional[str]) -> tuple[bool, str]:
        """
        决定是否触发LLM，以及触发原因。

        Returns:
            (should_trigger: bool, reason: str)
            reason: "speech" | "emotion_intervention" | "companionship" | "skip"
        """
        now = time.time()

        # 1. 司机主动说话 → 必须响应
        if speech_text and SPEECH_ALWAYS_TRIGGERS:
            return True, "speech"

        # 2. 计算这个情绪的冷却时间
        cooldown = COOLDOWN_BY_EMOTION.get(emotion, 60)
        elapsed = now - self.last_trigger_time
        still_cooling = elapsed < cooldown

        # 3. 情绪变化时重置冷却（情绪突变，需要立即响应）
        emotion_changed = emotion != self.last_emotion
        if emotion_changed and emotion in INTERVENTION_EMOTIONS:
            return True, "emotion_intervention"

        # 4. 冷却中，跳过
        if still_cooling:
            return False, "skip"

        # 5. 冷却过了，危险情绪 → 主动介入
        if emotion in INTERVENTION_EMOTIONS:
            return True, "emotion_intervention"

        # 6. 冷却过了，平稳情绪 → 陪伴
        return True, "companionship"

    def record_trigger(self, emotion: str):
        """触发后记录状态"""
        self.last_trigger_time = time.time()
        self.last_emotion = emotion
        self.trigger_count += 1

    def reset(self):
        """新旅程开始时调用"""
        self.last_trigger_time = 0
        self.last_emotion = None
        self.trigger_count = 0
