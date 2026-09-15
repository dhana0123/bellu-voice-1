"""Indic Parler-TTS."""

from __future__ import annotations

import logging
import numpy as np

from .audio import SAMPLE_RATE, resample_mono
from .lang import tts_description

logger = logging.getLogger("bellu_voice.tts")

PARLER_REPO = "ai4bharat/indic-parler-tts"


class TtsEngine:
    def __init__(self, device: str, mock: bool = False):
        self.device = device
        self.mock = mock
        self.model = None
        self.tok = None
        self.desc_tok = None
        self.sr = SAMPLE_RATE
        if not mock:
            self._load()

    def _load(self) -> None:
        from .tf_compat import patch_transformers_for_parler

        patch_transformers_for_parler()
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        logger.info("Loading TTS %s …", PARLER_REPO)
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(PARLER_REPO).to(self.device)
        self.tok = AutoTokenizer.from_pretrained(PARLER_REPO)
        self.desc_tok = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        self.sr = int(self.model.config.sampling_rate)
        logger.info("TTS ready (sr=%s)", self.sr)

    def synthesize(self, text: str, lang: str) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        if self.mock:
            t = np.arange(int(0.35 * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
            return (0.08 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        import torch

        desc = self.desc_tok(tts_description(lang), return_tensors="pt")
        prompt = self.tok(text, return_tensors="pt")
        device = self.device
        desc = {k: v.to(device) for k, v in desc.items()}
        prompt = {k: v.to(device) for k, v in prompt.items()}
        with torch.inference_mode():
            gen = self.model.generate(
                input_ids=desc["input_ids"],
                attention_mask=desc.get("attention_mask"),
                prompt_input_ids=prompt["input_ids"],
                prompt_attention_mask=prompt.get("attention_mask"),
            )
        audio = gen.cpu().numpy().squeeze().astype(np.float32)
        return resample_mono(audio, self.sr, SAMPLE_RATE)
