from __future__ import annotations

import numpy as np
import sounddevice as sd

from bellu.perception.audio import resample_mono


class Speaker:
    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate

    def play(self, audio: np.ndarray, src_sr: int) -> None:
        wav = resample_mono(audio, src_sr, self.sample_rate)
        try:
            sd.play(wav, self.sample_rate, blocking=False)
        except Exception:
            pass


class NullSpeaker:
    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate

    def play(self, audio: np.ndarray, src_sr: int) -> None:
        return


class Microphone:
    def __init__(self, sample_rate: int, on_chunk) -> None:
        self.sample_rate = sample_rate
        self.on_chunk = on_chunk
        self.stream = None

    def start(self) -> None:
        def callback(indata, frames, time_info, status):
            self.on_chunk(np.asarray(indata[:, 0], dtype=np.float32))

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=int(self.sample_rate * 0.08),
            callback=callback,
        )
        self.stream.start()

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
