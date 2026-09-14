from __future__ import annotations

import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from bellu.perception.audio import resample_mono
from bellu.types import ASRState


class ASREngine(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int, timestamp: float) -> ASRState:
        raise NotImplementedError


def _patch_generation_config() -> None:
    from transformers.generation.configuration_utils import GenerationConfig

    if getattr(GenerationConfig, "_bellu_patched", False):
        return

    orig_fp = GenerationConfig.from_pretrained.__func__

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        kwargs.pop("dtype", None)
        kwargs.pop("torch_dtype", None)
        return orig_fp(cls, *args, **kwargs)

    orig_to_dict = GenerationConfig.to_dict

    def to_dict(self, *args, **kwargs):
        data = orig_to_dict(self, *args, **kwargs)
        for key, value in list(data.items()):
            # torch.dtype is not JSON-serializable; logging GenerationConfig needs strings.
            if type(value).__name__ == "dtype":
                data[key] = str(value)
        return data

    GenerationConfig.from_pretrained = from_pretrained
    GenerationConfig.to_dict = to_dict
    GenerationConfig._bellu_patched = True


def _patch_indic_canary(model_dir: str) -> None:
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    import modeling_indic_canary as mic  # type: ignore

    cls = mic.IndicCanaryForConditionalGeneration
    if getattr(cls, "_bellu_patched", False):
        return

    orig_init = cls.__init__
    orig_fp = cls.from_pretrained.__func__

    def init(self, *args, **kwargs):
        kwargs.pop("dtype", None)
        kwargs.pop("torch_dtype", None)
        return orig_init(self, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        kwargs.pop("dtype", None)
        kwargs.pop("torch_dtype", None)
        return orig_fp(cls, *args, **kwargs)

    cls.__init__ = init  # type: ignore[method-assign]
    cls.from_pretrained = from_pretrained
    cls._bellu_patched = True


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

        _patch_generation_config()
        _patch_indic_canary(model_dir)

        from indic_transcribe import IndicTranscribe  # type: ignore

        self._model = IndicTranscribe.from_pretrained(model_dir)

        target = self.device or "cpu"
        try:
            import torch

            model = getattr(self._model, "model", self._model)
            model.to(target)
            if str(target).startswith("cuda") and torch.cuda.is_available():
                model.to(dtype=torch.bfloat16)
            if hasattr(self._model, "device"):
                self._model.device = target
        except Exception:
            pass

    def transcribe(self, audio: np.ndarray, sample_rate: int, timestamp: float) -> ASRState:
        if self._model is None:
            self.load()
        import soundfile as sf

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
