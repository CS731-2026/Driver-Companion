# conversation.py
# 多轮对话记忆，让AI记住这次驾驶中的对话上下文

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Turn:
    role: str       # "user" | "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    emotion: Optional[str] = None   # 记录这一轮的情绪，方便分析


class ConversationMemory:
    """
    保存单次驾驶旅程中的对话历史。
    启动新旅程时调用 reset() 清空。
    """

    def __init__(self, max_turns: int = 6):
        """
        Args:
            max_turns: 保留最近多少轮对话（太长会超token限制，且驾驶场景不需要太长记忆）
        """
        self.max_turns = max_turns
        self.turns: List[Turn] = []

    def add_user(self, content: str, emotion: str = None):
        self.turns.append(Turn(role="user", content=content, emotion=emotion))
        self._trim()

    def add_assistant(self, content: str):
        self.turns.append(Turn(role="assistant", content=content))
        self._trim()

    def get_history(self) -> List[Dict]:
        """返回OpenAI格式的历史，注入到messages里"""
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def reset(self):
        """新旅程开始时调用"""
        self.turns = []

    def _trim(self):
        """只保留最近 max_turns 轮（每轮=user+assistant各一条）"""
        max_messages = self.max_turns * 2
        if len(self.turns) > max_messages:
            self.turns = self.turns[-max_messages:]

    def summary(self) -> str:
        """简单打印当前记忆状态，调试用"""
        lines = [f"[对话记忆] 共{len(self.turns)}条"]
        for t in self.turns:
            preview = t.content[:30] + "..." if len(t.content) > 30 else t.content
            emotion_tag = f" [{t.emotion}]" if t.emotion else ""
            lines.append(f"  {t.role}{emotion_tag}: {preview}")
        return "\n".join(lines)
