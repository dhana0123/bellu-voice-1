from __future__ import annotations

from collections import deque
from threading import Event, Thread
from time import sleep, time
from typing import Any, Callable

from bellu.gating import needs_llm
from bellu.memory import TemporalMemory
from bellu.perception.audio import AudioRing
from bellu.protocol import SpeechCommand, spoken_text
from bellu.types import ASRState, Action, GlobalState, TurnState


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
        self.last_command: SpeechCommand | None = None
        self.last_trigger = ""
        self.last_handled_transcript = ""
        self.ui_log: deque[dict[str, Any]] = deque(maxlen=80)
        self._mic = None
        self._loop_thread: Thread | None = None
        self.mode = "live"
        self.on_tts: Callable | None = None

    def start(self, use_microphone: bool = True) -> None:
        if self._loop_thread and self._loop_thread.is_alive():
            return
        self.stop.clear()
        if use_microphone and self.microphone_factory is not None:
            self._mic = self.microphone_factory(self.cfg["sample_rate"], self.ring.push)
            self._mic.start()
        self._loop_thread = Thread(target=self._loop, daemon=True)
        self._loop_thread.start()
        self._note("system", "event loop started")

    def halt(self) -> None:
        self.stop.set()
        if self._mic is not None:
            self._mic.stop()
            self._mic = None
        self._note("system", "event loop stopped")

    def join(self) -> None:
        try:
            while not self.stop.is_set():
                sleep(0.2)
        except KeyboardInterrupt:
            self.stop.set()
        self.halt()

    def push_audio(self, chunk) -> None:
        self.ring.push(chunk)

    def interrupt(self) -> None:
        self.tts.cancel_playback()
        self.state.assistant.speaking = False
        self.state.assistant.current_action = "interrupted"
        self._note("system", "interrupt")

    def view(self) -> dict[str, Any]:
        return {
            "running": bool(self._loop_thread and self._loop_thread.is_alive() and not self.stop.is_set()),
            "mode": self.mode,
            "rms": round(self.ring.rms, 4),
            "snapshot": self.state.snapshot(),
            "last_command": self.last_command.to_dict() if self.last_command else None,
            "last_trigger": self.last_trigger,
            "log": list(self.ui_log),
        }

    def _note(self, kind: str, text: str) -> None:
        self.ui_log.append({"t": time(), "kind": kind, "text": text[:240]})

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
                    self._note(kind, asr_state.text)

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
                transcript = (self.state.asr.text or "").strip()
                if decision.reason in {"turn_complete", "asr_final", "backchannel_opportunity"}:
                    if transcript and transcript == self.last_handled_transcript:
                        sleep(tick)
                        continue
                self.last_llm = now
                self.last_trigger = decision.reason
                if transcript and decision.reason in {"turn_complete", "asr_final", "typed_turn"}:
                    self.last_handled_transcript = transcript
                try:
                    command = self.brain.decide(
                        self.state,
                        self.memory.prompt_block(self.state.snapshot()),
                        decision.reason,
                    )
                except Exception as exc:
                    self._note("error", f"llm: {exc}")
                    command = SpeechCommand.wait(reason=f"llm_error:{type(exc).__name__}")
                self._apply(command)
            sleep(tick)

    def _apply(self, command: SpeechCommand) -> None:
        self.last_command = command
        self.state.assistant.current_action = command.action.value.lower()
        self._note("protocol", f"{command.action.value} {spoken_text(command) or command.reason}".strip())
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
            if self.on_tts is not None:
                self.on_tts(audio, sr)

        Thread(target=self.tts.speak, args=(command, sink), daemon=True).start()

    def ingest_text(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        now = time()
        self.state.asr = ASRState(text=text, is_final=True, timestamp=now)
        self.state.sta.user_speaking = False
        self.state.sta.turn_completion = 0.95
        self.state.sta.turn_state = TurnState.COMPLETE
        self.memory.add("user_final", text, now)
        self._note("user_final", text)
        command = self.brain.decide(
            self.state,
            self.memory.prompt_block(self.state.snapshot()),
            "typed_turn",
        )
        self.last_trigger = "typed_turn"
        self.last_llm = now
        self.last_handled_transcript = text
        self._apply(command)
