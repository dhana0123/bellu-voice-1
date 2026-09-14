from __future__ import annotations

from collections import deque
from queue import Empty, Queue
from threading import Event, Thread
from time import sleep, time
from typing import Any, Callable

from bellu.log import clip, clog
from bellu.memory import TemporalMemory
from bellu.perception.audio import AudioRing
from bellu.protocol import SpeechCommand, spoken_text
from bellu.types import ASRState, Action, GlobalState, TurnState


class DuplexRuntime:
    """Two parallel streams, never stopping each other.

    IN  : mic ring → ASR / STA → memory  (always)
    OUT : speak queue → TTS → speakers   (always)
    LLM : reads memory, enqueues SAY; never cancels OUT except UI Stop.
    """

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
        self.last_trigger = "in"
        self.last_handled_transcript = ""
        self._last_asr_log = ""
        self._last_sta_log = ""
        self._asr_busy = Event()
        self._sta_busy = Event()
        self._llm_busy = Event()
        self._speak_q: Queue[SpeechCommand] = Queue(maxsize=8)
        self.ui_log: deque[dict[str, Any]] = deque(maxlen=80)
        self._mic = None
        self._in_thread: Thread | None = None
        self._out_thread: Thread | None = None
        self._loop_thread: Thread | None = None
        self.mode = "live"
        self.on_tts: Callable | None = None

    def start(self, use_microphone: bool = True) -> None:
        if self._in_thread and self._in_thread.is_alive():
            return
        self.stop.clear()
        if use_microphone and self.microphone_factory is not None:
            self._mic = self.microphone_factory(self.cfg["sample_rate"], self.ring.push)
            self._mic.start()
        self._in_thread = Thread(target=self._in_loop, name="bellu-in", daemon=True)
        self._out_thread = Thread(target=self._out_loop, name="bellu-out", daemon=True)
        self._loop_thread = self._in_thread
        self._in_thread.start()
        self._out_thread.start()
        clog("loop", f"IN+OUT parallel mic={'on' if self._mic else 'browser'}")
        self._note("system", "duplex streams started")

    def halt(self) -> None:
        self.stop.set()
        self.tts.cancel_playback()
        if self._mic is not None:
            self._mic.stop()
            self._mic = None
        clog("loop", "stopped")
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
        while True:
            try:
                self._speak_q.get_nowait()
            except Empty:
                break
        self.tts.cancel_playback()
        self.state.assistant.speaking = False
        self.state.assistant.current_action = "interrupted"
        clog("out", "stop (user)")
        self._note("system", "interrupt")

    def view(self) -> dict[str, Any]:
        return {
            "running": bool(self._in_thread and self._in_thread.is_alive() and not self.stop.is_set()),
            "mode": self.mode,
            "rms": round(self.ring.rms, 4),
            "snapshot": self.state.snapshot(),
            "last_command": self.last_command.to_dict() if self.last_command else None,
            "last_trigger": self.last_trigger,
            "context": self.memory.prompt_block(self.state),
            "log": list(self.ui_log),
        }

    def _note(self, kind: str, text: str) -> None:
        self.ui_log.append({"t": time(), "kind": kind, "text": text[:240]})
        if kind == "error":
            clog("error", text)

    def _in_loop(self) -> None:
        tick = self.cfg["event_loop_ms"] / 1000.0
        asr_every = self.cfg["asr_interval_ms"] / 1000.0
        sta_every = self.cfg["sta_interval_ms"] / 1000.0
        while not self.stop.is_set():
            now = time()
            self.state.time = now
            self.state.rms = self.ring.rms
            self.state.sta.user_speaking = self.ring.speaking
            self.state.sta.pause_ms = self.ring.pause_ms()
            tts_busy = bool(getattr(self.tts, "busy", Event()).is_set())
            self.state.assistant.speaking = tts_busy
            if not self._asr_busy.is_set() and now - self.last_asr >= asr_every:
                self._asr_busy.set()
                Thread(target=self._asr_once, daemon=True).start()
            if not self._sta_busy.is_set() and now - self.last_sta >= sta_every:
                self._sta_busy.set()
                Thread(target=self._sta_once, daemon=True).start()
            if not self._llm_busy.is_set() and not tts_busy:
                self._llm_busy.set()
                Thread(target=self._llm_once, daemon=True).start()
            sleep(tick)

    def _out_loop(self) -> None:
        while not self.stop.is_set():
            try:
                command = self._speak_q.get(timeout=0.08)
            except Empty:
                continue
            text = spoken_text(command)
            if not text:
                continue
            self.state.assistant.last_text = text
            self.memory.add("assistant_say", text)

            def sink(audio, sr, _cmd=command):
                self.speaker.play(audio, sr)
                if self.on_tts is not None:
                    self.on_tts(audio, sr)

            clog("out", f"speak {clip(text)}")
            try:
                self.tts.speak(command, sink)
            except Exception as exc:
                clog("tts", f"FAIL {type(exc).__name__}: {exc}")
                self._note("error", f"tts: {exc}")

    def _asr_once(self) -> None:
        now = time()
        self.last_asr = now
        audio = self.ring.window(self.cfg["asr"]["window_s"])
        t0 = time()
        try:
            asr_state = self.asr.transcribe(audio, self.cfg["sample_rate"], now)
        except Exception as exc:
            clog("in", f"asr FAIL {type(exc).__name__}: {exc}")
            self._note("error", f"asr: {exc}")
            self._asr_busy.clear()
            return
        ms = (time() - t0) * 1000
        shown = clip(asr_state.text) or "(silence)"
        line = f"{ms:.0f}ms {asr_state.language or '-'} {shown}"
        if line != self._last_asr_log:
            self._last_asr_log = line
            clog("in", f"asr {line}")
        if asr_state.text:
            self.state.asr = asr_state
            kind = "user_final" if asr_state.is_final else "user_partial"
            self.memory.add(kind, asr_state.text, now)
            self._note(kind, asr_state.text)
        self._asr_busy.clear()

    def _sta_once(self) -> None:
        now = time()
        self.last_sta = now
        audio = self.ring.window(self.cfg["sta"]["window_s"])
        try:
            sta_state = self.sta.infer(
                audio,
                self.cfg["sample_rate"],
                now,
                assistant_speaking=self.state.assistant.speaking,
            )
        except Exception as exc:
            clog("in", f"sta FAIL {type(exc).__name__}: {exc}")
            self._sta_busy.clear()
            return
        sta_state.pause_ms = self.ring.pause_ms()
        sta_state.interruption_probability = 0.0
        sta_state.overlap = False
        self.state.sta = sta_state
        sta_line = f"{sta_state.turn_state.value} complete={sta_state.turn_completion:.2f}"
        if sta_line != self._last_sta_log:
            self._last_sta_log = sta_line
            clog("in", f"sta {sta_line}")
        self._sta_busy.clear()

    def _llm_once(self) -> None:
        now = time()
        self.last_llm = now
        self.last_trigger = "in"
        try:
            command = self.brain.decide(
                self.state,
                self.memory.prompt_block(self.state),
                "in",
            )
        except Exception as exc:
            self._note("error", f"llm: {exc}")
            command = SpeechCommand.wait(reason=f"llm_error:{type(exc).__name__}")
        self._apply(command)
        self._llm_busy.clear()

    def _apply(self, command: SpeechCommand) -> None:
        if command.action in {Action.STOP, Action.INTERRUPT}:
            command = SpeechCommand.wait(reason="barge_in_ignored_use_stop")
        self.last_command = command
        self.state.assistant.current_action = command.action.value.lower()
        spoken = spoken_text(command) or command.reason
        clog("proto", f"{command.action.value} {clip(spoken)}")
        if command.action != Action.WAIT:
            self._note("protocol", f"{command.action.value} {spoken}".strip())

        if command.action in {Action.WAIT, Action.CONTINUE}:
            return
        text = spoken_text(command)
        if not text:
            return
        try:
            self._speak_q.put_nowait(command)
            clog("out", f"queued {clip(text)}")
        except Exception:
            clog("out", "queue full — keep playing")

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
        clog("in", f"text {clip(text)}")
        self._note("user_final", text)
        self.last_handled_transcript = text
        if not self._llm_busy.is_set():
            self._llm_busy.set()
            Thread(target=self._llm_once, daemon=True).start()
