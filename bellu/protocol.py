from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from bellu.types import Action

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Style:
    emotion: str = "neutral"
    intensity: float = 0.5
    energy: float = 0.5
    pace: float = 1.0
    confidence: float = 0.7
    smile: float = 0.0
    laugh: float = 0.0


@dataclass
class Timing:
    pause_before_ms: int = 0
    pause_after_ms: int = 0
    duration_ms: Optional[int] = None
    overlap: bool = False


@dataclass
class SpeechCommand:
    """Stable LLM → TTS contract. The LLM never sets F0/energy/spectral tilt."""

    action: Action = Action.WAIT
    text: str = ""
    nonverbal: str = ""
    style: Style = field(default_factory=Style)
    timing: Timing = field(default_factory=Timing)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["action"] = self.action.value
        return payload

    @classmethod
    def wait(cls, reason: str = "") -> "SpeechCommand":
        return cls(action=Action.WAIT, reason=reason)

    @classmethod
    def from_llm_text(cls, raw: str) -> "SpeechCommand":
        text = raw.strip()
        match = _JSON_RE.search(text)
        if not match:
            if not text:
                return cls.wait("empty llm output")
            return cls(action=Action.SAY, text=text, reason="unstructured llm text")
        data = json.loads(match.group(0))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpeechCommand":
        action = Action(str(data.get("action", "WAIT")).upper())
        style_in = data.get("style") or {}
        timing_in = data.get("timing") or {}
        return cls(
            action=action,
            text=str(data.get("text") or ""),
            nonverbal=str(data.get("nonverbal") or data.get("type") or ""),
            style=Style(
                emotion=str(style_in.get("emotion", "neutral")),
                intensity=float(style_in.get("intensity", 0.5)),
                energy=float(style_in.get("energy", 0.5)),
                pace=float(style_in.get("pace", 1.0)),
                confidence=float(style_in.get("confidence", 0.7)),
                smile=float(style_in.get("smile", 0.0)),
                laugh=float(style_in.get("laugh", 0.0)),
            ),
            timing=Timing(
                pause_before_ms=int(timing_in.get("pause_before_ms", 0)),
                pause_after_ms=int(timing_in.get("pause_after_ms", 0)),
                duration_ms=timing_in.get("duration_ms"),
                overlap=bool(timing_in.get("overlap", False)),
            ),
            reason=str(data.get("reason") or ""),
        )


BACKCHANNEL_PHRASES = {
    "MM_HMM": "mm-hmm",
    "HMM": "hmm",
    "UHH": "uhh",
    "OK": "okay",
    "HAAN": "haan",
    "ACHA": "acha",
}


def spoken_text(command: SpeechCommand) -> str:
    if command.action == Action.BACKCHANNEL:
        key = (command.nonverbal or "MM_HMM").upper()
        return BACKCHANNEL_PHRASES.get(key, command.nonverbal or "mm-hmm")
    return command.text.strip()
