from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bellu.types import GlobalState


@dataclass
class DecisionNeed:
    needed: bool
    reason: str
    priority: int = 0


def needs_llm(state: GlobalState, gating: dict[str, Any], last_decision_s: float) -> DecisionNeed:
    """The 80 ms loop always updates state. The LLM only runs on meaningful events."""

    dt = state.time - last_decision_s
    min_interval = gating.get("min_decision_interval_ms", 320) / 1000.0
    if dt < min_interval:
        return DecisionNeed(False, "cooldown")

    sta = state.sta
    asr = state.asr
    speaking = state.assistant.speaking
    bc = float(gating.get("backchannel_threshold", 0.75))
    irq = float(gating.get("interruption_threshold", 0.55))
    complete = float(gating.get("complete_threshold", 0.80))
    wait_t = float(gating.get("wait_threshold", 0.70))

    if speaking and (sta.interruption_probability >= irq or (sta.user_speaking and asr.text)):
        return DecisionNeed(True, "user_interrupt", priority=100)

    if sta.turn_state.value == "WAIT" or (
        not sta.user_speaking and sta.turn_completion >= wait_t and "wait" in (asr.text or "").lower()
    ):
        return DecisionNeed(True, "user_wait", priority=80)

    if not speaking and sta.turn_completion >= complete and asr.text:
        return DecisionNeed(True, "turn_complete", priority=70)

    if (
        not speaking
        and sta.user_speaking
        and sta.backchannel_opportunity >= bc
        and sta.turn_completion < 0.55
    ):
        return DecisionNeed(True, "backchannel_opportunity", priority=40)

    if asr.is_final and asr.text and not speaking:
        return DecisionNeed(True, "asr_final", priority=60)

    return DecisionNeed(False, "monitor")
