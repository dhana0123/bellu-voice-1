from __future__ import annotations

from threading import Event, Thread
from time import sleep, time
from typing import Any

from bellu.gating import needs_llm
from bellu.memory import TemporalMemory
from bellu.perception.audio import AudioRing
from bellu.protocol import SpeechCommand, spoken_text
from bellu.types import Action, GlobalState


class DuplexRuntime:
    """80 ms orchestrator. Perception always runs. The LLM only runs on gated events."""

    def __init__(
        self,
        cfg: dict[str, Any],
        asr,
        sta,
        brain,
        tts,
        speaker,
        microphone_factory,
    ) -> None:
        self.cfg = cfg
        self.asr = asr
        self.sta = sta
        self.brain = brain
        self.tts = tts
        self.speaker = speaker
        self.microphone_factory = microphone_factory
        self.state = GlobalState()
        self.memory = TemporalMemory()
        self.ring = AudioRing(cfg["sample_rate"], cfg["ring_buffer_s"])
        self.stop = Event()
        self.last_asr = 0.0
        self.last_sta = 0.0
        self.last_llm = 0.0
        self._mic = None

    def start(self) -> None:
        self._mic = self.microphone_factory(self.cfg["sample_rate"], self.ring.push)
        self._mic.start()
        Thread(target=self._loop, daemon=True).start()

    def join(self) -> None:
        try:
            while not self.stop.is_set():
                sleep(0.2)
        except KeyboardInterrupt:
            self.stop.set()
        if self._mic is not None:
            self._mic.stop()

    def _loop(self) -> None:
        tick = self.cfg["event_loop_ms"] / 1000.0
        asr_every = self.cfg["asr_interval_ms"] / 1000.0
        sta_every = self.cfg["sta_interval_ms"] / 1000.0
        while not self.stop.is_set():
            now = time()
            self.state.time = now
            self.state.sta.user_speaking = self.ring.speaking
            self.state.sta.pause_ms = self.ring.pause_ms()
            self.state.assistant.speaking = bool(getattr(self.tts, "busy", Event()).is_set())

            if now - self.last_asr >= asr_every:
                self.last_asr = now
                audio = self.ring.window(self.cfg["asr"]["window_s"])
                asr_state = self.asr.transcribe(audio, self.cfg["sample_rate"], now)
                if asr_state.text:
                    self.state.asr = asr_state
                    kind = "user_final" if asr_state.is_final else "user_partial"
                    self.memory.add(kind, asr_state.text, now)

            if now - self.last_sta >= sta_every:
                self.last_sta = now
                audio = self.ring.window(self.cfg["sta"]["window_s"])
                sta_state = self.sta.infer(
                    audio,
                    self.cfg["sample_rate"],
                    now,
                    assistant_speaking=self.state.assistant.speaking,
                )
                sta_state.pause_ms = self.ring.pause_ms()
                self.state.sta = sta_state
                self.memory.add("sta", f"{sta_state.turn_state.value} {sta_state.easy_turn_transcript}", now)

            gating = {**self.cfg.get("gating", {}), "min_decision_interval_ms": self.cfg["llm"].get("min_decision_interval_ms", 320)}
            decision = needs_llm(self.state, gating, self.last_llm)
            if decision.needed:
                self.last_llm = now
                command = self.brain.decide(self.state, self.memory.prompt_block(self.state.snapshot()), decision.reason)
                self._apply(command)
            sleep(tick)

    def _apply(self, command: SpeechCommand) -> None:
        self.state.assistant.current_action = command.action.value.lower()
        if command.action in {Action.STOP, Action.INTERRUPT}:
            self.tts.cancel_playback()
            self.state.assistant.speaking = False
        if command.action == Action.WAIT:
            return
        if command.action == Action.CONTINUE:
            getattr(self.tts, "paused", Event()).clear()
            return

        text = spoken_text(command)
        if not text:
            return
        self.memory.add("assistant_say", text)
        self.state.assistant.last_text = text

        def sink(audio, sr):
            self.speaker.play(audio, sr)

        Thread(target=self.tts.speak, args=(command, sink), daemon=True).start()
