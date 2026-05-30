# prompt_builder.py
# 把情绪策略 + 用户语音 融合成发给LLM的消息列表

from typing import Optional, List, Dict
from llm.emotion_strategy import build_system_prompt


def build_messages(
    emotion: str,
    speech_text: Optional[str],
    trigger_reason: str,
    conversation_history: Optional[List[Dict]] = None
) -> List[Dict]:
    """
    构建发给LLM的完整messages列表

    Args:
        emotion:              情绪标签
        speech_text:          STT转录文字，没有则传None
        trigger_reason:       触发原因 "speech" | "emotion_intervention" | "companionship"
        conversation_history: 历史对话，用于多轮记忆（可选）

    Returns:
        OpenAI格式的messages列表
    """
    system_prompt = build_system_prompt(emotion)

    # 根据触发原因构建user消息
    if trigger_reason == "speech" and speech_text:
        user_content = f'司机说："{speech_text}"'

    elif trigger_reason == "emotion_intervention":
        # 情绪异常但司机没说话，主动介入
        intervention_hints = {
            "angry":   "司机没有说话，但检测到明显愤怒情绪，请主动给出一句关怀。",
            "sad":     "司机没有说话，但情绪低落，请轻柔地表示陪伴。",
            "fear":    "司机没有说话，但检测到紧张焦虑，请给出稳定感。",
            "disgust": "司机没有说话，但检测到烦躁情绪，请主动给出一句缓和。",
            "surprise":"司机没有说话，但情绪有突然波动，请确认他的状态。",
        }
        user_content = intervention_hints.get(
            emotion,
            f"司机没有说话，情绪为{emotion}，请根据策略主动给出一句回应。"
        )

    else:
        # companionship - 平稳状态下的陪伴，防被动疲劳
        user_content = "司机情绪平稳，已驾驶一段时间，请给出一句自然的陪伴或轻松话题。"

    # 组装messages：system + 历史 + 当前
    messages = [{"role": "system", "content": system_prompt}]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": user_content})
    return messages
