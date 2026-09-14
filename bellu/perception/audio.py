from __future__ import annotations

from collections import deque
from threading import Lock
from time import time

import numpy as np


class AudioRing:
    def __init__(self, sample_rate: int, seconds: float) -> None:
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * seconds)
        self._buf: deque[np.ndarray] = deque()
        self._n = 0
        self._written = 0
        self._lock = Lock()
        self.last_voice_s = 0.0
        self.speaking = False
        self.rms = 0.0

    def push(self, chunk: np.ndarray) -> None:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        rms = float(np.sqrt(np.mean(np.square(chunk))) + 1e-9)
        self.rms = rms
        self.speaking = rms > 0.015
        if self.speaking:
            self.last_voice_s = time()
        with self._lock:
            self._buf.append(chunk)
            self._n += len(chunk)
            self._written += len(chunk)
            while self._n > self.max_samples and self._buf:
                dropped = self._buf.popleft()
                self._n -= len(dropped)

    def window(self, seconds: float) -> np.ndarray:
        n = int(self.sample_rate * seconds)
        with self._lock:
            if not self._buf:
                return np.zeros(n, dtype=np.float32)
            audio = np.concatenate(list(self._buf))
        if len(audio) >= n:
            return audio[-n:]
        pad = np.zeros(n - len(audio), dtype=np.float32)
        return np.concatenate([pad, audio])

    def pause_ms(self) -> int:
        if self.speaking:
            return 0
        return int(max(0.0, time() - self.last_voice_s) * 1000)

    @property
    def written(self) -> int:
        with self._lock:
            return self._written

    def slice_from(self, start: int, end: int | None = None) -> np.ndarray:
        """Samples in [start, end) of the monotonic write cursor (clipped to the ring)."""

        with self._lock:
            written = self._written
            if not self._buf:
                return np.zeros(0, dtype=np.float32)
            audio = np.concatenate(list(self._buf))
        ring_start = written - len(audio)
        stop = written if end is None else min(int(end), written)
        begin = max(int(start), ring_start)
        if stop <= begin:
            return np.zeros(0, dtype=np.float32)
        return audio[begin - ring_start : stop - ring_start].copy()


def resample_mono(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if src_sr == dst_sr or audio.size == 0:
        return audio
    duration = audio.size / float(src_sr)
    n = max(1, int(duration * dst_sr))
    x_old = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def placeholder_speech(text: str, sample_rate: int = 16000) -> np.ndarray:
    """Audible stand-in when ParlerTTS is missing. Varying tones, not silence."""

    text = (text or "").strip()
    if not text:
        return np.zeros(0, dtype=np.float32)
    n = int(sample_rate * min(2.6, 0.08 * max(len(text), 6)))
    audio = np.zeros(n, dtype=np.float32)
    pos = 0
    hop = int(0.07 * sample_rate)
    seg = int(0.09 * sample_rate)
    for ch in text[:28]:
        end = min(n, pos + seg)
        sl = end - pos
        if sl < 16:
            break
        freq = 240.0 + (ord(ch) % 28) * 16.0
        tt = np.arange(sl, dtype=np.float32) / float(sample_rate)
        env = np.hanning(sl).astype(np.float32)
        audio[pos:end] += 0.22 * env * np.sin(2.0 * np.pi * freq * tt)
        pos += hop
    return np.clip(audio, -0.45, 0.45).astype(np.float32)


def speech_rate(audio: np.ndarray, sample_rate: int) -> float:
    if audio.size < sample_rate // 4:
        return 1.0
    energy = np.abs(audio)
    thresh = max(0.02, float(energy.mean()) * 1.5)
    voiced = energy > thresh
    rate = float(voiced.mean()) * 2.0
    return float(np.clip(rate, 0.6, 1.8))
