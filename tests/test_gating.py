from bellu.gating import needs_llm
from bellu.perception.turn_tags import parse_turn
from bellu.protocol import SpeechCommand
from bellu.types import ASRState, GlobalState, STAState, TurnState


def test_protocol_json_roundtrip():
    raw = """
    here is the command
    {"action": "SAY", "text": "I understand why you're frustrated.",
     "style": {"emotion": "empathetic", "intensity": 0.65, "pace": 0.95},
     "timing": {"pause_before_ms": 120}}
    """
    cmd = SpeechCommand.from_llm_text(raw)
    assert cmd.action.value == "SAY"
    assert "frustrated" in cmd.text
    assert cmd.style.emotion == "empathetic"
    assert cmd.timing.pause_before_ms == 120


def test_wait_on_incomplete():
    state = GlobalState()
    state.time = 10.0
    state.asr = ASRState(text="I already called them three times", is_final=False)
    state.sta = STAState(
        user_speaking=True,
        turn_completion=0.31,
        backchannel_opportunity=0.81,
        turn_state=TurnState.INCOMPLETE,
    )
    decision = needs_llm(state, {"backchannel_threshold": 0.75, "min_decision_interval_ms": 0}, last_decision_s=0.0)
    assert decision.needed
    assert decision.reason == "backchannel_opportunity"


def test_interrupt_while_assistant_speaks():
    state = GlobalState()
    state.time = 10.0
    state.assistant.speaking = True
    state.asr = ASRState(text="No that's not what I mean")
    state.sta = STAState(user_speaking=True, interruption_probability=0.7, overlap=True)
    decision = needs_llm(state, {"interruption_threshold": 0.55, "min_decision_interval_ms": 0}, last_decision_s=0.0)
    assert decision.reason == "user_interrupt"


def test_easy_turn_tag_parse():
    state, tag, text = parse_turn("我已经打过三次了<COMPLETE>")
    assert state.value == "COMPLETE"
    assert tag == "<COMPLETE>"
    assert "COMPLETE" not in text


def test_loop_does_not_call_llm_every_tick():
    state = GlobalState()
    state.time = 0.16
    state.sta = STAState(user_speaking=True, turn_completion=0.2, backchannel_opportunity=0.2)
    decision = needs_llm(state, {"min_decision_interval_ms": 320}, last_decision_s=0.0)
    assert not decision.needed
