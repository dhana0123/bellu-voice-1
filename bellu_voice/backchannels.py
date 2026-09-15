"""Pre-rendered backchannel clips."""

from __future__ import annotations

import logging

import numpy as np

from .lang import backchannel_texts, normalize_lang

logger = logging.getLogger("bellu_voice.backchannels")


class Backchannels:
    def __init__(self, tts, lang: str, mock: bool = False):
        self.clips: list[np.ndarray] = []
        self._i = 0
        texts = backchannel_texts(lang)
        if mock:
            t = np.arange(int(0.2 * 24000), dtype=np.float32) / 24000
            self.clips = [(0.05 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)]
            return
        for phrase in texts:
            try:
                wav = tts.synthesize(phrase, normalize_lang(lang))
                if wav.size:
                    self.clips.append(wav)
            except Exception as exc:
                logger.warning("Backchannel %r failed: %s", phrase, exc)
        if not self.clips:
            t = np.arange(int(0.15 * 24000), dtype=np.float32) / 24000
            self.clips = [(0.04 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)]

    def next(self) -> np.ndarray:
        clip = self.clips[self._i % len(self.clips)]
        self._i += 1
        return clip
