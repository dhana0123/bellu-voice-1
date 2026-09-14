from __future__ import annotations

from collections import deque
from threading import Lock

import numpy as np

from bellu.perception.audio import resample_mono


class PcmQueue:
    """TTS generation writes here; WebSocket playback reads independently."""

    def __init__(self, play_sr: int = 16000, maxlen: int = 256) -> None:
        self.play_sr = play_sr
        self._q: deque[bytes] = deque(maxlen=maxlen)
        self._lock = Lock()
        self.epoch = 0

    def push(self, audio: np.ndarray, sr: int, epoch: int) -> bytes | None:
        if epoch != self.epoch:
            return None
        pcm = resample_mono(np.asarray(audio, dtype=np.float32).reshape(-1), sr, self.play_sr)
        if pcm.size == 0:
            return None
        raw = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        with self._lock:
            self._q.append(raw)
        return raw

    def pop(self) -> bytes | None:
        with self._lock:
            if not self._q:
                return None
            return self._q.popleft()

    def clear(self) -> None:
        with self._lock:
            self._q.clear()
        self.epoch += 1

    def pending(self) -> int:
        with self._lock:
            return len(self._q)
