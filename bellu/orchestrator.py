from __future__ import annotations

from collections import deque
from threading import Event, Thread
from time import sleep, time
from typing import Any, Callable

import numpy as np

from bellu.expression.playout import PcmQueue
from bellu.language import clean_spoken
from bellu.log import clip, clog
from bellu.memory import TemporalMemory
from bellu.perception.audio import AudioRing
from bellu.protocol import SpeechCommand
from bellu.types import ASRState, GlobalState, STAEvent, STAState, SystemState, TurnState, UserState
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
        self._sta_fail_log = ""
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
            "audio_start": u.audio_start,
            "audio_end": u.audio_end,
            "pcm_pending": self.pcm.pending(),
            "user_state": self.state.user_state.value,
            "system_state": self.state.assistant.system_state.value,
        }

    def _note(self, kind: str, text: str) -> None:
        self.ui_log.append({"t": time(), "kind": kind, "text": text[:240]})
        if kind == "error":
            clog("error", text)

    def _assistant_live(self) -> bool:
        generating = bool(getattr(self.tts, "busy", Event()).is_set())
        return generating or self.pcm.pending() > 0

    def _sync_duplex_state(self) -> None:
        if self.barge_in.is_set():
            self.state.assistant.system_state = SystemState.STOPPING
        elif self._assistant_live():
            self.state.assistant.system_state = SystemState.SPEAKING
        else:
            self.state.assistant.system_state = SystemState.IDLE
        self.state.assistant.speaking = self.state.assistant.system_state == SystemState.SPEAKING
        ev = self.state.sta.event
        if ev in {STAEvent.SPEAKING, STAEvent.INTERRUPTION}:
            self.state.user_state = UserState.SPEAKING
        elif ev == STAEvent.END_OF_TURN:
            self.state.user_state = UserState.END_OF_TURN
        elif ev == STAEvent.HOLD and self.state.sta.turn_state == TurnState.WAIT:
            self.state.user_state = UserState.SILENT
        elif ev == STAEvent.HOLD:
            self.state.user_state = UserState.HOLD
        else:
            self.state.user_state = UserState.SILENT

    def _preroll(self) -> int:
        return int(0.35 * int(self.cfg["sample_rate"]))

    def _turn_audio(self):
        u = self.utterance.current
        if u.frozen and u.audio_end is not None:
            return self.ring.slice_from(u.audio_end)
        return self.ring.slice_from(u.audio_start)

    def _audio_energy(self, audio) -> float:
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
        if wav.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(wav))) + 1e-9)

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
            self._sync_duplex_state()
            if self.barge_in.is_set() and not bool(getattr(self.tts, "busy", Event()).is_set()):
                self.barge_in.clear()
            if (
                self.utterance.open()
                and not self._asr_busy.is_set()
                and now - self.last_asr >= asr_every
            ):
                self._asr_busy.set()
                Thread(target=self._asr_once, daemon=True).start()
            if not self._sta_busy.is_set() and now - self.last_sta >= sta_every:
                self._sta_busy.set()
                Thread(target=self._sta_once, daemon=True).start()
            event = self.state.sta.event
            cursor = self.ring.written
            if event == STAEvent.INTERRUPTION and live:
                self.interrupt()
                if self.utterance.begin_speech(cursor, self._preroll()):
                    clog("in", f"utt={self.utterance.current.utterance_id} barge-in")
            elif event == STAEvent.SPEAKING:
                post = self._turn_audio() if self.utterance.current.frozen else None
                if not self.utterance.current.frozen or self._audio_energy(post) > 0.015:
                    if self.utterance.begin_speech(cursor, self._preroll()):
                        clog("in", f"utt={self.utterance.current.utterance_id} start cur={self.utterance.current.audio_start}")
            elif event == STAEvent.END_OF_TURN:
                if self.utterance.freeze(cursor):
                    frozen = self.utterance.current.final_transcript
                    self.state.asr.is_final = True
                    self.state.asr.text = frozen
                    self.state.user_state = UserState.END_OF_TURN
                    clog(
                        "in",
                        f"eot utt={self.utterance.current.utterance_id} "
                        f"cur={self.utterance.current.audio_start}:{self.utterance.current.audio_end} {clip(frozen)}",
                    )
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
        if not self.utterance.open():
            self._asr_busy.clear()
            return
        u = self.utterance.current
        audio = self.ring.slice_from(u.audio_start)
        min_n = int(0.25 * int(self.cfg["sample_rate"]))
        if audio.size < min_n:
            self._asr_busy.clear()
            return
        cursor = self.ring.written
        t0 = time()
        try:
            asr_state = self.asr.transcribe(audio, self.cfg["sample_rate"], now)
        except Exception as exc:
            clog("in", f"asr FAIL {type(exc).__name__}: {exc}")
            self._note("error", f"asr: {exc}")
            self._asr_busy.clear()
            return
        ms = (time() - t0) * 1000
        text = (asr_state.text or "").strip()
        if not self.utterance.ingest_asr(text, cursor):
            self._asr_busy.clear()
            return
        shown = clip(self.utterance.current.text) or "(silence)"
        clog("in", f"asr utt={self.utterance.current.utterance_id} {ms:.0f}ms {asr_state.language or '-'} {shown}")
        asr_state.text = self.utterance.current.text
        asr_state.is_final = False
        self.state.asr = asr_state
        self.memory.add("user_partial", asr_state.text, now)
        self._note("user_partial", asr_state.text)
        self._asr_busy.clear()

    def _sta_once(self) -> None:
        now = time()
        self.last_sta = now
        audio = self._turn_audio()
        min_sta = int(0.08 * int(self.cfg["sample_rate"]))
        if audio.size < min_sta:
            sta_state = STAState(
                user_speaking=False,
                turn_completion=0.10,
                timestamp=now,
                turn_state=TurnState.WAIT,
                event=STAEvent.HOLD,
            )
            self.state.sta = sta_state
            self._sync_duplex_state()
            sta_line = (
                f"user={self.state.user_state.value} sys={self.state.assistant.system_state.value} "
                f"{sta_state.event.value} complete={sta_state.turn_completion:.2f}"
            )
            if sta_line != self._last_sta_log:
                self._last_sta_log = sta_line
                clog("in", f"sta {sta_line}")
            self._sta_busy.clear()
            return
        try:
            sta_state = self.sta.infer(
                audio,
                self.cfg["sample_rate"],
                now,
                assistant_speaking=self._assistant_live(),
            )
        except Exception as exc:
            if self._sta_fail_log != str(exc):
                self._sta_fail_log = str(exc)
                clog("in", f"sta FAIL {type(exc).__name__}: {exc}")
                clog("boot", "STA falling back to MockSTA")
            from bellu.perception.mock_sta import MockSTA

            self.sta = MockSTA()
            try:
                sta_state = self.sta.infer(
                    audio,
                    self.cfg["sample_rate"],
                    now,
                    assistant_speaking=self._assistant_live(),
                )
            except Exception:
                self._sta_busy.clear()
                return
            self.state.sta = sta_state
            self._sync_duplex_state()
            sta_line = (
                f"user={self.state.user_state.value} sys={self.state.assistant.system_state.value} "
                f"{sta_state.event.value} complete={sta_state.turn_completion:.2f}"
            )
            clog("in", f"sta {sta_line}")
            self._sta_busy.clear()
            return
        if not sta_state.pause_ms:
            sta_state.pause_ms = self.ring.pause_ms()
        self.state.sta = sta_state
        self._sync_duplex_state()
        sta_line = (
            f"user={self.state.user_state.value} sys={self.state.assistant.system_state.value} "
            f"{sta_state.event.value} complete={sta_state.turn_completion:.2f}"
        )
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
        self.utterance.begin_speech(self.ring.written, self._preroll())
        self.utterance.ingest_asr(text, self.ring.written)
        self.utterance.freeze(self.ring.written)
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
