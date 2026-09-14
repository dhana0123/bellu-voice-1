from __future__ import annotations

import numpy as np

from bellu.types import ASRState


class MockASR:
    def transcribe(self, audio: np.ndarray, sample_rate: int, timestamp: float) -> ASRState:
        energy = float(np.sqrt(np.mean(np.square(audio))) + 1e-9)
        if energy < 0.02:
            return ASRState(text="", is_final=False, timestamp=timestamp)
        return ASRState(text="(mock transcript)", is_final=True, language="en", timestamp=timestamp)
