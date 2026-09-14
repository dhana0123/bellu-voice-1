from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock


@dataclass
class ChatSession:
    brain: object
    asr: object | None = None
    tts: object | None = None
    history: list[dict] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock)

    def reply(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        with self._lock:
            answer = self.brain.chat(text, list(self.history))
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": answer})
            if len(self.history) > 32:
                self.history = self.history[-32:]
            return answer
