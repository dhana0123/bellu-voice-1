from __future__ import annotations

import numpy as np

SAMPLE_RATE = 24_000
ASR_RATE = 16_000
FRAME = 1920  # 80 ms at 24 kHz
DUALTURN_HOP = FRAME * 3  # 240 ms


def resample_mono(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if src_sr == dst_sr:
        return x
    n_src = x.shape[0]
    n_dst = int(round(n_src * dst_sr / src_sr))
    if n_dst <= 1:
        return np.zeros(max(n_dst, 0), dtype=np.float32)
    src_t = np.linspace(0.0, 1.0, n_src, endpoint=False)
    dst_t = np.linspace(0.0, 1.0, n_dst, endpoint=False)
    return np.interp(dst_t, src_t, x).astype(np.float32)


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def split_sentences(text: str) -> list[str]:
    import re

    parts = re.split(r"(?<=[\.!?।?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]
