from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from time import time


_IDS = count(1)


def merge_partial(prev: str, incoming: str) -> str:
    """Keep one growing utterance from overlapping ASR windows."""

    prev = (prev or "").strip()
    incoming = (incoming or "").strip()
    if not incoming:
        return prev
    if not prev:
        return incoming
    if incoming == prev:
        return prev
    if incoming in prev:
        return prev
    if prev in incoming:
        return incoming
    prev_w = prev.split()
    new_w = incoming.split()
    max_k = min(len(prev_w), len(new_w))
    for k in range(max_k, 0, -1):
        if prev_w[-k:] == new_w[:k]:
            return " ".join(prev_w + new_w[k:])
    if abs(len(incoming) - len(prev)) <= 8 or len(incoming) >= len(prev):
        return incoming
    return prev


@dataclass
class Utterance:
    utterance_id: int
    text: str = ""
    frozen: bool = False
    final_transcript: str = ""
    audio_start: int = 0
    audio_end: int | None = None
    last_asr_text: str = ""
    last_asr_cursor: int = -1
    started_s: float = field(default_factory=time)


class UtteranceTracker:
    def __init__(self) -> None:
        self.current = Utterance(utterance_id=next(_IDS), frozen=True, audio_end=0)
        self.handled_utterance_id = self.current.utterance_id

    def open(self) -> bool:
        return not self.current.frozen

    def ingest_asr(self, text: str, cursor: int) -> bool:
        """Return True if the hypothesis is new for this utterance+cursor."""

        text = (text or "").strip()
        if not text or self.current.frozen:
            return False
        if text == self.current.last_asr_text:
            return False
        self.current.last_asr_text = text
        self.current.last_asr_cursor = cursor
        self.current.text = merge_partial(self.current.text, text)
        return True

    def begin_speech(self, cursor: int, preroll: int = 0) -> bool:
        """Start a new utterance after freeze/handle. Returns True if id changed."""

        u = self.current
        if u.frozen or self.handled_utterance_id == u.utterance_id:
            start = u.audio_end if u.audio_end is not None else max(0, cursor - preroll)
            self.current = Utterance(utterance_id=next(_IDS), audio_start=start)
            return True
        if not u.text and u.audio_start == 0:
            u.audio_start = max(0, cursor - preroll)
        return False

    def freeze(self, cursor: int) -> bool:
        if self.current.frozen:
            return False
        text = (self.current.text or "").strip()
        if not text:
            return False
        self.current.frozen = True
        self.current.final_transcript = text
        self.current.audio_end = cursor
        return True

    def mark_handled(self) -> None:
        self.handled_utterance_id = self.current.utterance_id

    def ready_for_llm(self) -> bool:
        u = self.current
        return bool(
            u.frozen
            and u.final_transcript
            and u.utterance_id != self.handled_utterance_id
        )
