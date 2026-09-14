from __future__ import annotations

from bellu.protocol import SpeechCommand
from bellu.types import GlobalState


class MockBrain:
    def stream_reply(self, asr: str, on_chunk) -> None:
        text = (asr or "").strip()
        if text:
            on_chunk(f"అవును, {text}")

    def decide(self, state: GlobalState, memory_block: str, trigger: str) -> SpeechCommand:
        if trigger == "backchannel_opportunity":
            return SpeechCommand.from_dict(
                {"action": "BACKCHANNEL", "nonverbal": "MM_HMM", "style": {"intensity": 0.3}, "reason": trigger}
            )
        if trigger == "user_interrupt":
            return SpeechCommand.from_dict(
                {
                    "action": "INTERRUPT",
                    "text": "Sorry — go ahead.",
                    "style": {"emotion": "acknowledging", "intensity": 0.5},
                    "reason": trigger,
                }
            )
        if trigger in {"turn_complete", "asr_final", "typed_turn"} and state.asr.text:
            return SpeechCommand.from_dict(
                {
                    "action": "SAY",
                    "text": f"I hear you: {state.asr.text}",
                    "style": {"emotion": "empathetic", "intensity": 0.6, "pace": 0.95},
                    "timing": {"pause_before_ms": 80},
                    "reason": trigger,
                }
            )
        return SpeechCommand.wait(trigger)

    def chat(self, user_text: str, history: list[dict] | None = None) -> str:
        return f"I heard you: {user_text}"
