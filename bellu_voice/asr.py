"""Utterance ASR: Whisper (default) or IndicConformer for hi/te/ta/kn."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np

from .audio import ASR_RATE, resample_mono
from .lang import INDIC_CONFORMER, asr_backend_for, normalize_lang

logger = logging.getLogger("bellu_voice.asr")


class AsrEngine:
    def __init__(self, device: str, preference: str = "auto", mock: bool = False):
        self.device = device
        self.preference = preference
        self.mock = mock
        self._whisper = None
        self._conformer: dict[str, object] = {}

    def transcribe(self, pcm: np.ndarray, sample_rate: int, lang: str) -> str:
        if self.mock:
            return "Hello"
        lang = normalize_lang(lang)
        backend = asr_backend_for(lang, self.preference)
        audio = resample_mono(pcm, sample_rate, ASR_RATE)
        if audio.size < ASR_RATE * 0.2:
            return ""
        if backend == "conformer":
            try:
                return self._transcribe_conformer(audio, lang)
            except Exception as exc:
                logger.warning("IndicConformer failed (%s); falling back to Whisper", exc)
        return self._transcribe_whisper(audio, lang)

    def _transcribe_whisper(self, audio_16k: np.ndarray, lang: str) -> str:
        model = self._load_whisper()
        language = None if lang in {"", "auto"} else lang
        if hasattr(model, "transcribe") and self._whisper_kind == "faster":
            segs, _info = model.transcribe(audio_16k, language=language, vad_filter=True)
            return " ".join(s.text.strip() for s in segs).strip()
        result = model.transcribe(audio_16k, language=language, fp16=self.device != "cpu")
        return str(result.get("text", "")).strip()

    def _load_whisper(self):
        if self._whisper is not None:
            return self._whisper
        device = self.device
        try:
            from faster_whisper import WhisperModel

            logger.info("Loading faster-whisper large-v3 …")
            self._whisper = WhisperModel(
                "large-v3",
                device="cuda" if device.startswith("cuda") else "cpu",
                compute_type="float16" if device.startswith("cuda") else "int8",
            )
            self._whisper_kind = "faster"
            return self._whisper
        except Exception as exc:
            logger.warning("faster-whisper unavailable (%s); using openai-whisper", exc)
        import whisper

        logger.info("Loading openai-whisper large-v3 …")
        self._whisper = whisper.load_model("large-v3", device=device)
        self._whisper_kind = "openai"
        return self._whisper

    def _transcribe_conformer(self, audio_16k: np.ndarray, lang: str) -> str:
        repo = INDIC_CONFORMER[lang]
        model = self._load_conformer(repo)
        path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        try:
            import wave

            pcm = np.clip(audio_16k, -1.0, 1.0)
            pcm_i = (pcm * 32767.0).astype(np.int16)
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(ASR_RATE)
                wf.writeframes(pcm_i.tobytes())
            try:
                out = model.transcribe([path], language_id=lang, batch_size=1)
            except TypeError:
                out = model.transcribe([path], batch_size=1)
        finally:
            Path(path).unlink(missing_ok=True)
        return _extract_transcript(out).strip()

    def _load_conformer(self, repo: str):
        if repo in self._conformer:
            return self._conformer[repo]
        import torch
        import nemo.collections.asr as nemo_asr
        from huggingface_hub import hf_hub_download, list_repo_files

        files = [f for f in list_repo_files(repo) if f.endswith(".nemo")]
        if not files:
            raise RuntimeError(f"No .nemo file in {repo}")
        nemo_path = hf_hub_download(repo, files[0])
        Hybrid = getattr(nemo_asr.models, "EncDecHybridRNNTCTCBPEModel", None)
        restore_cls = Hybrid or nemo_asr.models.ASRModel
        map_location = "cuda" if self.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        try:
            model = restore_cls.restore_from(restore_path=str(nemo_path), map_location=map_location)
        except TypeError:
            model = restore_cls.restore_from(str(nemo_path))
        if hasattr(model, "cur_decoder"):
            model.cur_decoder = "ctc"
        model.freeze()
        model.eval()
        self._conformer[repo] = model
        logger.info("IndicConformer ready %s", repo)
        return model


def _extract_transcript(out) -> str:
    if out is None:
        return ""
    if isinstance(out, str):
        return out
    if isinstance(out, (list, tuple)) and out:
        first = out[0]
        if isinstance(first, str):
            return first
        if hasattr(first, "text"):
            return str(first.text)
        if isinstance(first, (list, tuple)) and first:
            return str(first[0])
    if hasattr(out, "text"):
        return str(out.text)
    return str(out)
