from bellu.gating import needs_llm
from bellu.language import collapse_repeats, clean_user_text
from bellu.memory import TemporalMemory
from bellu.perception.turn_tags import parse_turn
from bellu.protocol import SpeechCommand
from bellu.types import ASRState, GlobalState, STAState, TurnState


def test_global_context_has_three_layers():
    mem = TemporalMemory()
    state = GlobalState()
    state.rms = 0.04
    state.asr = ASRState(text="ఓకే", language="te")
    state.sta = STAState(user_speaking=True, turn_state=TurnState.INCOMPLETE)
    mem.add("user_partial", "ఓకే")
    mem.add("assistant_say", "అవును")
    block = mem.prompt_block(state)
    assert "CURRENT" in block
    assert "RECENT" in block
    assert "HISTORY" in block
    assert "ఓకే" in block
    assert "ACTIVE STATE" not in block


def test_collapse_ho_ho_loop():
    assert collapse_repeats("हो हो हो हो हो हो") == "हो हो"
    assert clean_user_text("हो हो हो हो हो हो हो") == ""
    assert "నమస్కారం" in clean_user_text("నమస్కారం ఎలా ఉన్నారు")


def test_protocol_invalid_json_does_not_raise():
    cmd = SpeechCommand.from_llm_text("{action: SAY, text: hello}")
    assert cmd.action.value in {"SAY", "WAIT"}
    cmd = SpeechCommand.from_llm_text("{")
    assert cmd.action.value == "WAIT"
    cmd = SpeechCommand.from_llm_text("{'action': 'SAY', 'text': 'Namaste'}")
    assert cmd.action.value == "SAY"
    assert "Namaste" in cmd.text


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
