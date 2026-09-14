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
    # Distinct decode of the same clip: keep the longer recent hypothesis.
    if abs(len(incoming) - len(prev)) <= 8 or len(incoming) >= len(prev):
        return incoming
    return prev


@dataclass
class Utterance:
    utterance_id: int
    text: str = ""
    frozen: bool = False
    final_transcript: str = ""
    started_s: float = field(default_factory=time)


class UtteranceTracker:
    def __init__(self) -> None:
        self.current = Utterance(utterance_id=next(_IDS))
        self.handled_utterance_id: int | None = None

    def ingest_asr(self, text: str) -> None:
        text = (text or "").strip()
        if not text or self.current.frozen:
            return
        merged = merge_partial(self.current.text, text)
        self.current.text = merged

    def begin_speech(self) -> None:
        if self.current.frozen or (
            self.handled_utterance_id == self.current.utterance_id and self.current.text
        ):
            self.current = Utterance(utterance_id=next(_IDS))

    def freeze(self) -> bool:
        if self.current.frozen:
            return False
        text = (self.current.text or "").strip()
        if not text:
            return False
        self.current.frozen = True
        self.current.final_transcript = text
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
