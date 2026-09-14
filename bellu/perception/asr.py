from __future__ import annotations

import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from bellu.language import CODE as LANG_CODE
from bellu.language import clean_user_text
from bellu.perception.audio import resample_mono
from bellu.types import ASRState


class ASREngine(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int, timestamp: float) -> ASRState:
        raise NotImplementedError


class _DynamicCacheLayers:
    """Bridge Indic-Canary (`.layers[i].keys`) and both transformers cache layouts.

    Newer transformers assign `self.layers` in Cache.__init__. Older ones only
    store `key_cache` / `value_cache`. A getter-only property breaks the new API.
    """

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        stored = obj.__dict__.get("layers")
        if stored is not None:
            return stored
        key_cache = obj.__dict__.get("key_cache")
        value_cache = obj.__dict__.get("value_cache")
        if key_cache is None or value_cache is None:
            raise AttributeError("DynamicCache has no layers or key_cache")
        from types import SimpleNamespace

        return [SimpleNamespace(keys=k, values=v) for k, v in zip(key_cache, value_cache)]

    def __set__(self, obj, value):
        obj.__dict__["layers"] = value


def _patch_dynamic_cache_layers() -> None:
    """Indic-Canary reads cross_cache.layers[i].keys; older transformers only expose key_cache."""

    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        return

    existing = getattr(DynamicCache, "layers", None)
    already = getattr(DynamicCache, "_bellu_layers_patched", False)
    if already and isinstance(existing, _DynamicCacheLayers):
        return
    # Getter-only property from the previous shim has no setter and must be replaced.
    if already and isinstance(existing, property) and existing.fset is not None:
        return

    DynamicCache.layers = _DynamicCacheLayers()  # type: ignore[method-assign]
    DynamicCache._bellu_layers_patched = True


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
        language: Optional[str] = LANG_CODE,
        device: str = "cuda",
        min_rms: float = 0.02,
    ) -> None:
        self.model_id = model_id
        self.language = language or LANG_CODE
        self.device = device
        self.min_rms = min_rms
        self._model = None
        self._last_text = ""

    def load(self) -> None:
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(self.model_id)
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        _patch_generation_config()
        _patch_dynamic_cache_layers()
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

        wav = resample_mono(audio, sample_rate, 16000)
        rms = float(np.sqrt(np.mean(np.square(wav))) + 1e-9)
        if rms < self.min_rms:
            return ASRState(text="", is_final=False, language=self.language, timestamp=timestamp)

        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        try:
            sf.write(path, wav, 16000)
            kwargs = {"lang": self.language or LANG_CODE}
            text, lid = self._model.transcribe(path, return_lid=True, **kwargs)
        finally:
            Path(path).unlink(missing_ok=True)

        text = clean_user_text(text or "")
        lang = self.language or LANG_CODE
        is_final = bool(text) and text == self._last_text
        self._last_text = text
        return ASRState(text=text, is_final=is_final, language=lang, timestamp=timestamp)
