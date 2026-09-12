from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from bellu.perception.audio import resample_mono, speech_rate
from bellu.perception.turn_tags import parse_turn
from bellu.types import STAState, TurnState

TURN_MAP = {
    TurnState.COMPLETE: dict(turn_completion=0.93, backchannel_opportunity=0.12, interruption_probability=0.05),
    TurnState.INCOMPLETE: dict(turn_completion=0.28, backchannel_opportunity=0.78, interruption_probability=0.08),
    TurnState.BACKCHANNEL: dict(turn_completion=0.15, backchannel_opportunity=0.10, interruption_probability=0.05),
    TurnState.WAIT: dict(turn_completion=0.88, backchannel_opportunity=0.05, interruption_probability=0.20),
    TurnState.UNKNOWN: dict(turn_completion=0.0, backchannel_opportunity=0.0, interruption_probability=0.0),
}

DEFAULT_PROMPT = (
    "Please transcribe the audio, then append exactly one turn tag: "
    "<COMPLETE>, <INCOMPLETE>, <BACKCHANNEL>, or <WAIT>."
)


def log_mel_spectrogram(wav: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
    """Match Easy-Turn / Wenet whisper log-mel (80 bins, hop 160, n_fft 400)."""

    waveform = torch.from_numpy(resample_mono(wav, sample_rate, 16000)).float()
    n_fft, hop, n_mels = 400, 160, 80
    window = torch.hann_window(n_fft)
    stft = torch.stft(waveform, n_fft, hop, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    import librosa

    filters = torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=n_fft, n_mels=n_mels)).float()
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.transpose(0, 1)


class EasyTurnSTA:
    """STA backend: ASLP-lab/Easy-Turn (complete / incomplete / backchannel / wait)."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        self.prompt = cfg.get("prompt") or DEFAULT_PROMPT
        self.model = None

    def load(self) -> None:
        root = Path(self.cfg["easy_turn_root"]).resolve()
        if not root.exists():
            raise FileNotFoundError(
                f"Easy-Turn root not found: {root}. Run `python -m bellu.cli setup-sta`."
            )
        wenet_parent = root
        if str(wenet_parent) not in sys.path:
            sys.path.insert(0, str(wenet_parent))

        from wenet.utils.init_model import init_model  # type: ignore

        train_yaml = Path(self.cfg["train_yaml"]).resolve()
        with train_yaml.open("r", encoding="utf-8") as handle:
            configs = yaml.safe_load(handle)
        args = SimpleNamespace(
            checkpoint=str(Path(self.cfg["checkpoint"]).resolve()),
            jit=False,
            use_lora=False,
            lora_ckpt_path=None,
        )
        loaded = init_model(args, configs)
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        if isinstance(model, tuple):
            model = model[0]
        self.model = model.to(self.device)
        self.model.eval()

    def infer(self, audio: np.ndarray, sample_rate: int, timestamp: float, assistant_speaking: bool) -> STAState:
        feats = log_mel_spectrogram(audio, sample_rate).unsqueeze(0).to(self.device)
        lengths = torch.tensor([feats.size(1)], device=self.device)
        if self.model is None:
            self.load()
        with torch.no_grad():
            raw = self.model.generate(wavs=feats, wavs_len=lengths, prompt=self.prompt)
        if isinstance(raw, list):
            raw = raw[0]
        turn, tag, transcript = parse_turn(str(raw))
        mapped = TURN_MAP[turn]
        speaking = float(np.sqrt(np.mean(np.square(audio))) + 1e-9) > 0.015
        overlap = bool(assistant_speaking and speaking)
        irq = mapped["interruption_probability"]
        if overlap:
            irq = max(irq, 0.65)
        return STAState(
            user_speaking=speaking,
            speech_rate=speech_rate(audio, sample_rate),
            emotion="neutral",
            emotion_intensity=0.0,
            prosody="falling" if turn == TurnState.COMPLETE else "flat",
            pause_ms=0,
            turn_completion=mapped["turn_completion"],
            backchannel_opportunity=mapped["backchannel_opportunity"],
            interruption_probability=irq,
            overlap=overlap,
            turn_state=turn,
            raw_tag=tag,
            easy_turn_transcript=transcript,
            timestamp=timestamp,
        )


class MockSTA:
    def infer(self, audio: np.ndarray, sample_rate: int, timestamp: float, assistant_speaking: bool) -> STAState:
        speaking = float(np.sqrt(np.mean(np.square(audio))) + 1e-9) > 0.015
        return STAState(
            user_speaking=speaking,
            turn_completion=0.2 if speaking else 0.9,
            backchannel_opportunity=0.8 if speaking else 0.1,
            timestamp=timestamp,
            turn_state=TurnState.INCOMPLETE if speaking else TurnState.COMPLETE,
        )
