from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from bellu.types import Action

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_ACTION_RE = re.compile(r"""['"]?action['"]?\s*[:=]\s*['"]?([A-Za-z_]+)""", re.IGNORECASE)
_TEXT_RE = re.compile(r"""['"]text['"]\s*[:=]\s*['"]([^'"]*)['"]""", re.IGNORECASE)


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
        text = (raw or "").strip()
        if not text:
            return cls.wait("empty llm output")
        upper = text.upper()
        if any(key.upper() in upper for key in ("ACTIVE STATE", "TURNING POINTS", "TRIGGER:", "RECENT TRACE")):
            return cls.wait("llm_dumped_context")

        fenced = _FENCE_RE.search(text)
        if fenced:
            text = fenced.group(1).strip()

        for blob in _json_blobs(text):
            data = _loads_object(blob)
            if data is not None:
                try:
                    return cls.from_dict(data)
                except Exception:
                    continue

        action_m = _ACTION_RE.search(text)
        text_m = _TEXT_RE.search(text)
        if action_m:
            spoken = (text_m.group(1) if text_m else "").strip()
            try:
                action = Action(action_m.group(1).upper())
            except ValueError:
                action = Action.SAY if spoken else Action.WAIT
            if action == Action.SAY and not spoken:
                spoken = _plain_speech(text)
            return cls(action=action, text=spoken, reason="partial llm protocol")

        spoken = _plain_speech(text)
        if not spoken:
            return cls.wait("unparseable llm output")
        return cls(action=Action.SAY, text=spoken, reason="unstructured llm text")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpeechCommand":
        try:
            action = Action(str(data.get("action", "WAIT")).upper())
        except ValueError:
            action = Action.SAY if data.get("text") else Action.WAIT
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
    "MM_HMM": "హ్మ్",
    "HMM": "హ్మ్",
    "UHH": "ఉమ్",
    "OK": "సరే",
    "HAAN": "అవును",
    "ACHA": "సరే",
}


def _json_blobs(text: str) -> list[str]:
    blobs: list[str] = []
    start = None
    depth = 0
    in_str = False
    quote = ""
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_str = False
            continue
        if ch in {'"', "'"}:
            in_str = True
            quote = ch
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                blobs.append(text[start : i + 1])
                start = None
    return blobs


def _loads_object(blob: str) -> dict[str, Any] | None:
    for candidate in (blob, blob.replace("'", '"')):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return data
    try:
        data = ast.literal_eval(blob)
    except (SyntaxError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _plain_speech(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("{") or cleaned.startswith("["):
        return ""
    return cleaned.split("\n", 1)[0].strip()[:400]


def spoken_text(command: SpeechCommand) -> str:
    if command.action == Action.BACKCHANNEL:
        key = (command.nonverbal or "MM_HMM").upper()
        return BACKCHANNEL_PHRASES.get(key, command.nonverbal or "mm-hmm")
    return command.text.strip()
