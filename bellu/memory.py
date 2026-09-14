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
    """Global context: CURRENT + RECENT + HISTORY. LLM never sees raw audio."""

    def __init__(self, recent_limit: int = 40, long_limit: int = 12) -> None:
        self.recent: deque[MemoryEvent] = deque(maxlen=recent_limit)
        self.long: deque[str] = deque(maxlen=long_limit)
        self._last_topic: str = ""

    def add(self, kind: str, text: str, timestamp: float | None = None) -> None:
        event = MemoryEvent(timestamp=timestamp or time(), kind=kind, text=text.strip())
        if not event.text:
            return
        self.recent.append(event)
        if kind in {"user_final", "assistant_say"} and len(event.text) > 24:
            self.long.append(f"{kind}: {event.text[:180]}")

    def set_topic(self, topic: str) -> None:
        topic = topic.strip()
        if topic and topic != self._last_topic:
            self._last_topic = topic
            self.long.append(f"topic: {topic}")

    def current_block(self, state: Any) -> str:
        asr = (getattr(state.asr, "text", "") or "").strip() or "<no voice>"
        sta = state.sta
        asst = state.assistant
        conv = state.conversation
        return (
            "CURRENT\n"
            f"vad_speaking={sta.user_speaking} rms={getattr(state, 'rms', 0):.4f} pause_ms={sta.pause_ms}\n"
            f"asr={asr} lang={state.asr.language or 'te'} final={state.asr.is_final}\n"
            f"sta turn={sta.turn_state.value} complete={sta.turn_completion:.2f} "
            f"irq={sta.interruption_probability:.2f} bc={sta.backchannel_opportunity:.2f} "
            f"overlap={sta.overlap} emotion={sta.emotion}\n"
            f"assistant speaking={asst.speaking} action={asst.current_action} last={asst.last_text[:80]}\n"
            f"topic={conv.topic or '-'} intent={conv.user_intent or '-'}"
        )

    def prompt_block(self, state: Any) -> str:
        recent_lines = [f"- {item.kind}: {item.text[:120]}" for item in list(self.recent)[-10:]]
        long_lines = [f"- {item}" for item in list(self.long)[-8:]]
        return (
            self.current_block(state)
            + "\n\nRECENT\n"
            + ("\n".join(recent_lines) or "- none")
            + "\n\nHISTORY\n"
            + ("\n".join(long_lines) or "- none")
        )
