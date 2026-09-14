from __future__ import annotations

from threading import Event, Thread

import numpy as np

from bellu.protocol import SpeechCommand, spoken_text
from bellu.types import STAEvent, STAState, TurnState


def _tail_rms(audio: np.ndarray, sample_rate: int, tail_s: float = 0.22) -> float:
    n = max(1, int(sample_rate * tail_s))
    tail = np.asarray(audio, dtype=np.float32).reshape(-1)[-n:]
    if tail.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(tail))) + 1e-9)


def _trailing_silence_ms(audio: np.ndarray, sample_rate: int, thresh: float = 0.015) -> tuple[int, bool]:
    wav = np.asarray(audio, dtype=np.float32).reshape(-1)
    if wav.size == 0:
        return 10_000, False
    hop = max(1, int(sample_rate * 0.02))
    last_voice = -1
    had = False
    for i in range(0, len(wav), hop):
        sl = wav[i : i + hop]
        rms = float(np.sqrt(np.mean(np.square(sl))) + 1e-9)
        if rms > thresh:
            had = True
            last_voice = i + len(sl)
    if not had:
        return int(len(wav) / sample_rate * 1000), False
    return int(max(0, len(wav) - last_voice) / sample_rate * 1000), True


class MockSTA:
    """Energy endpointing until Easy-Turn is loaded. Owns EOT; orchestrator has no pause gate."""

    EOT_SILENCE_MS = 550

    def infer(self, audio: np.ndarray, sample_rate: int, timestamp: float, assistant_speaking: bool) -> STAState:
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
        if wav.size < int(sample_rate * 0.08):
            return STAState(
                user_speaking=False,
                turn_completion=0.10,
                timestamp=timestamp,
                turn_state=TurnState.WAIT,
                event=STAEvent.HOLD,
            )
        tail = _tail_rms(wav, sample_rate)
        speaking_now = tail > 0.015
        silence_ms, had_speech = _trailing_silence_ms(audio, sample_rate)
        overlap = bool(assistant_speaking and speaking_now)

        if overlap and tail > 0.03:
            event = STAEvent.INTERRUPTION
            turn = TurnState.INCOMPLETE
            complete = 0.15
            irq = 0.85
        elif speaking_now:
            event = STAEvent.SPEAKING
            turn = TurnState.INCOMPLETE
            complete = 0.22
            irq = 0.08
        elif had_speech and silence_ms >= self.EOT_SILENCE_MS:
            event = STAEvent.END_OF_TURN
            turn = TurnState.COMPLETE
            complete = 0.93
            irq = 0.05
        elif had_speech:
            event = STAEvent.HOLD
            turn = TurnState.INCOMPLETE
            complete = 0.45
            irq = 0.05
        else:
            event = STAEvent.HOLD
            turn = TurnState.WAIT
            complete = 0.10
            irq = 0.0

        return STAState(
            user_speaking=speaking_now,
            turn_completion=complete,
            backchannel_opportunity=0.75 if speaking_now and not assistant_speaking else 0.1,
            interruption_probability=irq,
            overlap=overlap,
            pause_ms=silence_ms if not speaking_now else 0,
            timestamp=timestamp,
            turn_state=turn,
            event=event,
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
        self.speak_async(command, sink)

    def speak_text(self, text: str, sink) -> None:
        self.speak_async(SpeechCommand.say(text), sink)

    def speak_async(self, command: SpeechCommand, sink) -> None:
        Thread(target=self._run, args=(command, sink), daemon=True).start()

    def _run(self, command: SpeechCommand, sink) -> None:
        from bellu.perception.audio import placeholder_speech

        text = spoken_text(command)
        self.last_text = text
        if not text:
            return
        self.cancel.clear()
        self.busy.set()
        try:
            sink(placeholder_speech(text, 16000), 16000)
        finally:
            self.busy.clear()
