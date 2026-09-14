from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Action(str, Enum):
    SAY = "SAY"
    BACKCHANNEL = "BACKCHANNEL"
    WAIT = "WAIT"
    STOP = "STOP"
    INTERRUPT = "INTERRUPT"
    CONTINUE = "CONTINUE"


class TurnState(str, Enum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    BACKCHANNEL = "BACKCHANNEL"
    WAIT = "WAIT"
    UNKNOWN = "UNKNOWN"


class STAEvent(str, Enum):
    SPEAKING = "SPEAKING"
    HOLD = "HOLD"
    END_OF_TURN = "END_OF_TURN"
    INTERRUPTION = "INTERRUPTION"
    BACKCHANNEL_OPPORTUNITY = "BACKCHANNEL_OPPORTUNITY"


class UserState(str, Enum):
    SILENT = "SILENT"
    SPEAKING = "SPEAKING"
    HOLD = "HOLD"
    END_OF_TURN = "END_OF_TURN"


class SystemState(str, Enum):
    IDLE = "IDLE"
    SPEAKING = "SPEAKING"
    STOPPING = "STOPPING"


@dataclass
class ASRState:
    text: str = ""
    is_final: bool = False
    language: Optional[str] = None
    timestamp: float = 0.0


@dataclass
class STAState:
    """Conversational knowledge from Easy-Turn plus light timing features.

    Raw acoustics (F0, MFCC, codec tokens) never leave this module.
    """

    user_speaking: bool = False
    speech_rate: float = 1.0
    emotion: str = "neutral"
    emotion_intensity: float = 0.0
    prosody: str = "flat"
    pause_ms: int = 0
    turn_completion: float = 0.0
    backchannel_opportunity: float = 0.0
    interruption_probability: float = 0.0
    overlap: bool = False
    turn_state: TurnState = TurnState.UNKNOWN
    event: STAEvent = STAEvent.HOLD
    raw_tag: str = ""
    easy_turn_transcript: str = ""
    timestamp: float = 0.0


@dataclass
class AssistantRuntime:
    speaking: bool = False
    current_action: str = "waiting"
    last_text: str = ""
    system_state: SystemState = SystemState.IDLE


@dataclass
class ConversationFacts:
    topic: str = ""
    user_intent: str = ""
    summary: str = ""


@dataclass
class GlobalState:
    time: float = 0.0
    rms: float = 0.0
    asr: ASRState = field(default_factory=ASRState)
    sta: STAState = field(default_factory=STAState)
    assistant: AssistantRuntime = field(default_factory=AssistantRuntime)
    conversation: ConversationFacts = field(default_factory=ConversationFacts)
    user_state: UserState = UserState.SILENT

    def snapshot(self) -> dict[str, Any]:
        return {
            "time": round(self.time, 3),
            "vad": {
                "speaking": self.sta.user_speaking,
                "rms": round(self.rms, 4),
                "pause_ms": self.sta.pause_ms,
            },
            "user": {
                "speaking": self.sta.user_speaking,
                "user_state": self.user_state.value,
                "transcript": self.asr.text,
                "is_final": self.asr.is_final,
                "language": self.asr.language,
                "emotion": self.sta.emotion,
                "emotion_intensity": self.sta.emotion_intensity,
                "speech_rate": self.sta.speech_rate,
                "turn_state": self.sta.turn_state.value,
                "sta_event": self.sta.event.value,
                "turn_completion": self.sta.turn_completion,
                "backchannel_opportunity": self.sta.backchannel_opportunity,
                "interruption_probability": self.sta.interruption_probability,
                "pause_ms": self.sta.pause_ms,
                "overlap": self.sta.overlap,
            },
            "assistant": {
                "speaking": self.assistant.speaking,
                "system_state": self.assistant.system_state.value,
                "current_action": self.assistant.current_action,
            },
            "conversation": {
                "topic": self.conversation.topic,
                "user_intent": self.conversation.user_intent,
            },
        }
