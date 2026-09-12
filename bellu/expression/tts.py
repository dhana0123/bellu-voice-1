from __future__ import annotations

from threading import Event, Thread
from typing import Callable

import numpy as np
import torch

from bellu.protocol import SpeechCommand, spoken_text

AudioSink = Callable[[np.ndarray, int], None]


def description_from_protocol(command: SpeechCommand, speaker: str = "Laura") -> str:
    style = command.style
    emotion = style.emotion.replace("_", " ")
    if style.pace >= 1.15:
        pace = "quickly"
    elif style.pace <= 0.85:
        pace = "slowly"
    else:
        pace = "at a moderate pace"
    laugh = " with a light laugh" if style.laugh >= 0.4 else ""
    smile = " a slight smile in the voice" if style.smile >= 0.4 else ""
    return (
        f"{speaker} speaks {pace} with a {emotion} delivery{laugh}{smile}. "
        f"Clear audio, close-mic recording, intensity {style.intensity:.2f}."
    )


class ParlerExpression:
    """Streaming ParlerTTS. Understands START / STREAM / PAUSE / RESUME / CANCEL via the protocol."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.device = cfg.get("device", "cuda")
        self.model = None
        self.desc_tokenizer = None
        self.prompt_tokenizer = None
        self.cancel = Event()
        self.paused = Event()
        self.busy = Event()
        self.sample_rate = int(cfg.get("sample_rate", 44100))

    def load(self) -> None:
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        name = self.cfg["model_id"]
        dtype = torch.float16 if str(self.device).startswith("cuda") else torch.float32
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(name, torch_dtype=dtype).to(self.device)
        self.desc_tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        self.prompt_tokenizer = AutoTokenizer.from_pretrained(name)
        self.sample_rate = self.model.audio_encoder.config.sampling_rate

    def cancel_playback(self) -> None:
        self.cancel.set()

    def speak(self, command: SpeechCommand, sink: AudioSink) -> None:
        text = spoken_text(command)
        if command.action.value in {"WAIT", "STOP", "CONTINUE"} or not text:
            return
        if self.model is None:
            self.load()
        self.cancel.clear()
        self.busy.set()
        try:
            self._stream(text, description_from_protocol(command, self.cfg.get("speaker", "Laura")), sink)
        finally:
            self.busy.clear()

    def _stream(self, text: str, description: str, sink: AudioSink) -> None:
        from parler_tts import ParlerTTSStreamer

        frame_rate = self.model.audio_encoder.config.frame_rate
        play_steps = int(frame_rate * float(self.cfg.get("play_steps_s", 0.5)))
        streamer = ParlerTTSStreamer(self.model, device=self.device, play_steps=play_steps)
        desc = self.desc_tokenizer(description, return_tensors="pt").to(self.device)
        prompt = self.prompt_tokenizer(text, return_tensors="pt").to(self.device)
        kwargs = dict(
            input_ids=desc.input_ids,
            prompt_input_ids=prompt.input_ids,
            attention_mask=desc.attention_mask,
            prompt_attention_mask=prompt.attention_mask,
            streamer=streamer,
            do_sample=True,
            temperature=1.0,
            min_new_tokens=10,
        )
        thread = Thread(target=self.model.generate, kwargs=kwargs, daemon=True)
        thread.start()
        for chunk in streamer:
            if self.cancel.is_set():
                break
            if chunk is None or np.asarray(chunk).size == 0:
                break
            while self.paused.is_set() and not self.cancel.is_set():
                pass
            audio = np.asarray(chunk, dtype=np.float32)
            sink(audio, self.sample_rate)
        thread.join(timeout=0.1)


class MockTTS:
    def __init__(self, cfg: dict | None = None) -> None:
        self.cancel = Event()
        self.busy = Event()
        self.last_text = ""

    def load(self) -> None:
        return

    def cancel_playback(self) -> None:
        self.cancel.set()

    def speak(self, command: SpeechCommand, sink: AudioSink) -> None:
        text = spoken_text(command)
        self.last_text = text
        if not text:
            return
        sink(np.zeros(1600, dtype=np.float32), 16000)
