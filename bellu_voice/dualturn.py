"""DualTurn streaming policy (rolling 5 s window, 240 ms hop)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .audio import DUALTURN_HOP, SAMPLE_RATE, rms

logger = logging.getLogger("bellu_voice.dualturn")

Action = Literal["ST", "CL", "SL", "CT", "BC", "NONE"]

DUALTURN_REPO = "anyreach-ai/dualturn-qwen2.5-mimi-0.5B"
WINDOW_SEC = 5.0


@dataclass
class DualTurnSignals:
    vad_user: float
    vad_agent: float
    eot_user: float
    bot_user: float
    bot_agent: float
    hold_user: float
    bc_agent: float


def heuristic(sig: DualTurnSignals, speaking: bool) -> Action:
    if speaking:
        if sig.vad_user > 0.55 and sig.bot_user > 0.35:
            return "SL"
        return "CT"
    if sig.bc_agent > 0.55 and sig.vad_user > 0.45:
        return "BC"
    if sig.eot_user > 0.5 and (sig.bot_agent > 0.45 or sig.vad_user < 0.35):
        return "ST"
    if sig.hold_user > 0.5:
        return "CL"
    return "NONE"


def energy_signals(user: np.ndarray, agent: np.ndarray, speaking: bool) -> DualTurnSignals:
    u = rms(user[-DUALTURN_HOP:]) if user.size else 0.0
    a = rms(agent[-DUALTURN_HOP:]) if agent.size else 0.0
    u_on = 1.0 if u > 0.02 else 0.0
    a_on = 1.0 if a > 0.02 else 0.0
    # Silence after speech looks like EOT.
    u_prev = rms(user[-2 * DUALTURN_HOP : -DUALTURN_HOP]) if user.size > DUALTURN_HOP else 0.0
    eot_u = 1.0 if u_prev > 0.02 and u_on < 0.5 else 0.0
    return DualTurnSignals(
        vad_user=u_on,
        vad_agent=a_on,
        eot_user=eot_u,
        bot_user=u_on,
        bot_agent=0.6 if (eot_u and not speaking) else a_on,
        hold_user=0.0,
        bc_agent=0.0,
    )


class DualTurnEngine:
    def __init__(self, device: str, mock: bool = False):
        self.device = device
        self.mock = mock
        self.model = None
        self.window = int(SAMPLE_RATE * WINDOW_SEC)
        self.user = np.zeros(0, dtype=np.float32)
        self.agent = np.zeros(0, dtype=np.float32)
        self._since_infer = 0
        if not mock:
            self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoModel

        logger.info("Loading DualTurn %s …", DUALTURN_REPO)
        self.model = AutoModel.from_pretrained(
            DUALTURN_REPO,
            trust_remote_code=True,
            device_map=self.device if self.device != "cpu" else "cpu",
        )
        self.model.eval()
        if hasattr(self.model, "to"):
            try:
                self.model.to(self.device)
            except Exception:
                pass
        logger.info("DualTurn ready")

    def reset(self) -> None:
        self.user = np.zeros(0, dtype=np.float32)
        self.agent = np.zeros(0, dtype=np.float32)
        self._since_infer = 0

    def _trim(self) -> None:
        n = self.window
        if self.user.size > n:
            self.user = self.user[-n:]
        if self.agent.size > n:
            self.agent = self.agent[-n:]

    def push_user(self, pcm: np.ndarray) -> None:
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        self.user = np.concatenate([self.user, pcm])
        pad = self.user.size - self.agent.size
        if pad > 0:
            self.agent = np.concatenate([self.agent, np.zeros(pad, dtype=np.float32)])
        self._since_infer += pcm.size
        self._trim()

    def push_agent(self, pcm: np.ndarray) -> None:
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        self.agent = np.concatenate([self.agent, pcm])
        pad = self.agent.size - self.user.size
        if pad > 0:
            self.user = np.concatenate([self.user, np.zeros(pad, dtype=np.float32)])
        self._trim()

    def ready(self) -> bool:
        return self._since_infer >= DUALTURN_HOP and self.user.size >= DUALTURN_HOP

    def infer(self, speaking: bool) -> Action:
        self._since_infer = 0
        n = min(self.user.size, self.agent.size, self.window)
        if n < DUALTURN_HOP:
            return "NONE"
        user = self.user[-n:]
        agent = self.agent[-n:]
        if self.mock or self.model is None:
            sig = energy_signals(user, agent, speaking)
            return heuristic(sig, speaking)
        import torch

        stereo = np.stack([user, agent], axis=0)
        wav = torch.from_numpy(stereo).to(dtype=torch.float32)
        try:
            wav = wav.to(next(self.model.parameters()).device)
        except Exception:
            pass
        with torch.inference_mode():
            out = self.model(wav, sr=SAMPLE_RATE)

        def last2(t, i: int) -> float:
            v = t[0, -1, i].detach().float().cpu().item()
            return float(v)

        sig = DualTurnSignals(
            vad_user=last2(out.vad_probs, 0),
            vad_agent=last2(out.vad_probs, 1),
            eot_user=last2(out.eot_probs, 0),
            bot_user=last2(out.bot_probs, 0),
            bot_agent=last2(out.bot_probs, 1),
            hold_user=last2(out.hold_probs, 0),
            bc_agent=last2(out.bc_probs, 1),
        )
        return heuristic(sig, speaking)
