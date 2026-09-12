from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import time
from typing import Any


@dataclass
class MemoryEvent:
    timestamp: float
    kind: str
    text: str


class TemporalMemory:
    """Three-level memory so the LLM never sees raw audio history."""

    def __init__(self, recent_limit: int = 40, long_limit: int = 12) -> None:
        self.recent: deque[MemoryEvent] = deque(maxlen=recent_limit)
        self.long: deque[str] = deque(maxlen=long_limit)
        self._last_topic: str = ""

    def add(self, kind: str, text: str, timestamp: float | None = None) -> None:
        event = MemoryEvent(timestamp=timestamp or time(), kind=kind, text=text.strip())
        if not event.text:
            return
        self.recent.append(event)
        if kind in {"user_final", "assistant_say"} and len(event.text) > 40:
            self.long.append(f"{kind}: {event.text[:240]}")

    def set_topic(self, topic: str) -> None:
        topic = topic.strip()
        if topic and topic != self._last_topic:
            self._last_topic = topic
            self.long.append(f"topic: {topic}")

    def prompt_block(self, active: dict[str, Any]) -> str:
        recent_lines = [f"- {item.kind}: {item.text}" for item in list(self.recent)[-12:]]
        long_lines = [f"- {item}" for item in self.long]
        return (
            "ACTIVE STATE\n"
            f"{active}\n\n"
            "RECENT TRACE\n"
            + ("\n".join(recent_lines) or "- none")
            + "\n\nLONG MEMORY\n"
            + ("\n".join(long_lines) or "- none")
        )
