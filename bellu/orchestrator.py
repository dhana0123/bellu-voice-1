from __future__ import annotations

from collections import deque
from threading import Event, Thread
from time import sleep, time
from typing import Any, Callable

from bellu.expression.playout import PcmQueue
from bellu.language import clean_spoken
from bellu.log import clip, clog
from bellu.memory import TemporalMemory
from bellu.perception.audio import AudioRing
from bellu.protocol import SpeechCommand
from bellu.types import ASRState, GlobalState, STAEvent, TurnState
from bellu.utterance import UtteranceTracker


class DuplexRuntime:
    """IN (ASR+STA) always runs. One frozen utterance → one LLM. TTS generate ≠ playback."""

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
        self.utterance = UtteranceTracker()
        self.pcm = PcmQueue(int(cfg.get("playback", {}).get("output_sample_rate") or 16000))
        self.stop = Event()
        self.barge_in = Event()
        self.last_asr = 0.0
        self.last_sta = 0.0
        self.last_llm = 0.0
        self.last_command: SpeechCommand | None = None
        self.last_trigger = "in"
        self._last_asr_log = ""
        self._last_sta_log = ""
        self._asr_busy = Event()
        self._sta_busy = Event()
        self._llm_busy = Event()
        self.ui_log: deque[dict[str, Any]] = deque(maxlen=80)
        self._mic = None
        self._in_thread: Thread | None = None
        self._loop_thread: Thread | None = None
        self.mode = "live"
        self.on_tts: Callable | None = None
        self.on_audio_reset: Callable | None = None

    def start(self, use_microphone: bool = True) -> None:
        if self._in_thread and self._in_thread.is_alive():
            return
        self.stop.clear()
        self.barge_in.clear()
        if use_microphone and self.microphone_factory is not None:
            self._mic = self.microphone_factory(self.cfg["sample_rate"], self.ring.push)
            self._mic.start()
        self._in_thread = Thread(target=self._in_loop, name="bellu-in", daemon=True)
        self._loop_thread = self._in_thread
        self._in_thread.start()
        clog("loop", "IN always; STA EOT → one LLM; TTS queue")
        self._note("system", "duplex streams started")

    def halt(self) -> None:
        self.stop.set()
        self.interrupt()
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
        self.barge_in.set()
        self.tts.cancel_playback()
        self.pcm.clear()
        self.state.assistant.speaking = False
        self.state.assistant.current_action = "interrupted"
        clog("out", "stop (interrupt)")
        self._note("system", "interrupt")
        if self.on_audio_reset is not None:
            self.on_audio_reset()

    def view(self) -> dict[str, Any]:
        u = self.utterance.current
        return {
            "running": bool(self._in_thread and self._in_thread.is_alive() and not self.stop.is_set()),
            "mode": self.mode,
            "rms": round(self.ring.rms, 4),
            "snapshot": self.state.snapshot(),
            "last_command": self.last_command.to_dict() if self.last_command else None,
            "last_trigger": self.last_trigger,
            "context": self.memory.prompt_block(self.state),
            "log": list(self.ui_log),
            "utterance_id": u.utterance_id,
            "utterance_frozen": u.frozen,
            "pcm_pending": self.pcm.pending(),
        }

    def _note(self, kind: str, text: str) -> None:
        self.ui_log.append({"t": time(), "kind": kind, "text": text[:240]})
        if kind == "error":
            clog("error", text)

    def _assistant_live(self) -> bool:
        generating = bool(getattr(self.tts, "busy", Event()).is_set())
        return generating or self.pcm.pending() > 0

    def _audio_sink(self, audio, sr) -> None:
        epoch = self.pcm.epoch
        if self.barge_in.is_set():
            return
        raw = self.pcm.push(audio, sr, epoch)
        if raw is None:
            return
        self.speaker.play(audio, sr)
        if self.on_tts is not None:
            self.on_tts(raw, self.pcm.play_sr)

    def _in_loop(self) -> None:
        tick = self.cfg["event_loop_ms"] / 1000.0
        asr_every = self.cfg["asr_interval_ms"] / 1000.0
        sta_every = self.cfg["sta_interval_ms"] / 1000.0
        while not self.stop.is_set():
            now = time()
            self.state.time = now
            self.state.rms = self.ring.rms
            live = self._assistant_live()
            self.state.assistant.speaking = live
            if self.barge_in.is_set() and not bool(getattr(self.tts, "busy", Event()).is_set()):
                self.barge_in.clear()
            if not self._asr_busy.is_set() and now - self.last_asr >= asr_every:
                self._asr_busy.set()
                Thread(target=self._asr_once, daemon=True).start()
            if not self._sta_busy.is_set() and now - self.last_sta >= sta_every:
                self._sta_busy.set()
                Thread(target=self._sta_once, daemon=True).start()
            event = self.state.sta.event
            if event == STAEvent.INTERRUPTION and live:
                self.interrupt()
                self.utterance.begin_speech()
            elif event == STAEvent.SPEAKING:
                self.utterance.begin_speech()
            elif event == STAEvent.END_OF_TURN:
                if self.utterance.freeze():
                    frozen = self.utterance.current.final_transcript
                    self.state.asr.is_final = True
                    self.state.asr.text = frozen
                    clog("in", f"eot utt={self.utterance.current.utterance_id} {clip(frozen)}")
            if (
                event == STAEvent.END_OF_TURN
                and self.utterance.ready_for_llm()
                and not self._llm_busy.is_set()
            ):
                self._llm_busy.set()
                self.utterance.mark_handled()
                Thread(target=self._llm_once, daemon=True).start()
            sleep(tick)

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
            self.utterance.ingest_asr(asr_state.text)
            asr_state.text = self.utterance.current.final_transcript or self.utterance.current.text
            asr_state.is_final = self.utterance.current.frozen
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
                assistant_speaking=self._assistant_live(),
            )
        except Exception as exc:
            clog("in", f"sta FAIL {type(exc).__name__}: {exc}")
            self._sta_busy.clear()
            return
        if not sta_state.pause_ms:
            sta_state.pause_ms = self.ring.pause_ms()
        self.state.sta = sta_state
        sta_line = f"{sta_state.event.value} {sta_state.turn_state.value} complete={sta_state.turn_completion:.2f}"
        if sta_line != self._last_sta_log:
            self._last_sta_log = sta_line
            clog("in", f"sta {sta_line}")
        self._sta_busy.clear()

    def _on_llm_tokens(self, piece: str) -> None:
        piece = clean_spoken(piece or "")
        if not piece or self.stop.is_set() or self.barge_in.is_set():
            clog("out", "skip phrase")
            return
        cmd = SpeechCommand.say(piece, reason="eot_reply")
        self.last_command = cmd
        self.state.assistant.current_action = "say"
        self.state.assistant.last_text = piece
        self.memory.add("assistant_say", piece)
        self._note("protocol", f"SAY emotion={cmd.style.emotion} pace={cmd.style.pace} {clip(piece)}")
        clog("out", f"phrase→tts {clip(piece)}")
        try:
            speak = getattr(self.tts, "speak_async", None) or getattr(self.tts, "speak")
            speak(cmd, self._audio_sink)
        except Exception as exc:
            clog("tts", f"FAIL {type(exc).__name__}: {exc}")

    def _llm_once(self) -> None:
        self.barge_in.clear()
        asr = (self.utterance.current.final_transcript or self.state.asr.text or "").strip()
        self.last_llm = time()
        self.last_trigger = "eot"
        clog("llm", f"utt={self.utterance.current.utterance_id} asr={clip(asr)!r}")
        try:
            self.brain.stream_reply(asr, self._on_llm_tokens)
        except Exception as exc:
            self._note("error", f"llm: {exc}")
            clog("llm", f"FAIL {exc}")
        self._llm_busy.clear()

    def ingest_text(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        now = time()
        self.utterance.begin_speech()
        self.utterance.ingest_asr(text)
        self.utterance.freeze()
        self.state.asr = ASRState(text=text, is_final=True, timestamp=now)
        self.state.sta.user_speaking = False
        self.state.sta.turn_completion = 0.95
        self.state.sta.turn_state = TurnState.COMPLETE
        self.state.sta.event = STAEvent.END_OF_TURN
        self.memory.add("user_final", text, now)
        clog("in", f"text {clip(text)}")
        self._note("user_final", text)
        if self.utterance.ready_for_llm() and not self._llm_busy.is_set():
            self._llm_busy.set()
            self.utterance.mark_handled()
            Thread(target=self._llm_once, daemon=True).start()
