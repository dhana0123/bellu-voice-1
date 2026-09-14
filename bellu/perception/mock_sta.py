from __future__ import annotations

from threading import Event

import numpy as np

from bellu.protocol import SpeechCommand, spoken_text
from bellu.types import STAState, TurnState


class MockSTA:
    def infer(self, audio: np.ndarray, sample_rate: int, timestamp: float, assistant_speaking: bool) -> STAState:
        speaking = float(np.sqrt(np.mean(np.square(audio))) + 1e-9) > 0.015
        return STAState(
            user_speaking=speaking,
            turn_completion=0.2 if speaking else 0.92,
            backchannel_opportunity=0.8 if speaking else 0.1,
            interruption_probability=0.7 if (assistant_speaking and speaking) else 0.05,
            overlap=bool(assistant_speaking and speaking),
            timestamp=timestamp,
            turn_state=TurnState.INCOMPLETE if speaking else TurnState.COMPLETE,
        )


class MockTTS:
    def __init__(self, cfg: dict | None = None) -> None:
        self.cancel = Event()
        self.busy = Event()
        self.paused = Event()
        self.last_text = ""

    def load(self) -> None:
        return

    def cancel_playback(self) -> None:
        self.cancel.set()

    def speak(self, command: SpeechCommand, sink) -> None:
        from bellu.perception.audio import placeholder_speech

        text = spoken_text(command)
        self.last_text = text
        if not text:
            return
        self.busy.set()
        try:
            sink(placeholder_speech(text, 16000), 16000)
        finally:
            self.busy.clear()
