from __future__ import annotations

import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from bellu.perception.audio import resample_mono
from bellu.types import ASRState


class ASREngine(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int, timestamp: float) -> ASRState:
        raise NotImplementedError


class IndicTranscribeASR(ASREngine):
    """Streaming-window wrapper around Indic-Transcribe-core (Bodhan AI / AI4Bharat)."""

    def __init__(
        self,
        model_id: str = "bodhan-ai/indic-transcribe-core",
        language: Optional[str] = None,
        device: str = "cuda",
    ) -> None:
        self.model_id = model_id
        self.language = language
        self.device = device
        self._model = None
        self._last_text = ""

    def load(self) -> None:
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(self.model_id)
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        from indic_transcribe import IndicTranscribe  # type: ignore

        self._model = IndicTranscribe.from_pretrained(model_dir)
        if hasattr(self._model, "to") and self.device:
            try:
                self._model.to(self.device)
            except Exception:
                pass

    def transcribe(self, audio: np.ndarray, sample_rate: int, timestamp: float) -> ASRState:
        if self._model is None:
            self.load()
        wav = resample_mono(audio, sample_rate, 16000)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        try:
            sf.write(path, wav, 16000)
            kwargs = {}
            if self.language:
                kwargs["lang"] = self.language
            text, lid = self._model.transcribe(path, return_lid=True, **kwargs)
        finally:
            Path(path).unlink(missing_ok=True)

        text = (text or "").strip()
        lang = None
        if isinstance(lid, dict):
            lang = lid.get("lang")
        is_final = bool(text) and text == self._last_text
        self._last_text = text
        return ASRState(text=text, is_final=is_final, language=lang, timestamp=timestamp)


class MockASR(ASREngine):
    def transcribe(self, audio: np.ndarray, sample_rate: int, timestamp: float) -> ASRState:
        energy = float(np.sqrt(np.mean(np.square(audio))) + 1e-9)
        if energy < 0.02:
            return ASRState(text="", is_final=False, timestamp=timestamp)
        return ASRState(text="(mock transcript)", is_final=False, language="en", timestamp=timestamp)
