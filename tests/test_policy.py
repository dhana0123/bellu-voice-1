import numpy as np

from bellu_voice.audio import DUALTURN_HOP
from bellu_voice.dualturn import DualTurnEngine, DualTurnSignals, heuristic
from bellu_voice.lang import asr_backend_for, normalize_lang, system_prompt
from bellu_voice.llm import FALLBACK_LLM, resolve_llm_model
from bellu_voice.protocol import audio_msg, handshake, parse_kind, text_msg


def test_protocol():
    assert handshake() == b"\x00"
    assert parse_kind(audio_msg(b"abc")) == (1, b"abc")
    assert parse_kind(text_msg("hi"))[0] == 2
    assert parse_kind(text_msg("hi"))[1] == b"hi"


def test_lang_default_english():
    assert normalize_lang(None) == "en"
    assert normalize_lang("TE") == "te"
    assert asr_backend_for("en", "auto") == "whisper"
    assert asr_backend_for("te", "auto") == "conformer"
    assert "English" in system_prompt("en")
    assert "Telugu" in system_prompt("te")


def test_heuristic_st_and_sl():
    st = DualTurnSignals(
        vad_user=0.2,
        vad_agent=0.0,
        eot_user=0.8,
        bot_user=0.1,
        bot_agent=0.6,
        hold_user=0.1,
        bc_agent=0.0,
    )
    assert heuristic(st, speaking=False) == "ST"
    sl = DualTurnSignals(
        vad_user=0.7,
        vad_agent=0.8,
        eot_user=0.1,
        bot_user=0.5,
        bot_agent=0.2,
        hold_user=0.0,
        bc_agent=0.0,
    )
    assert heuristic(sl, speaking=True) == "SL"


def test_resolve_llm_keeps_non_30b():
    assert resolve_llm_model("sarvamai/sarvam-m") == "sarvamai/sarvam-m"
    assert FALLBACK_LLM == "sarvamai/sarvam-m"


def test_mock_engine_eot_start_talking():
    eng = DualTurnEngine("cpu", mock=True)
    speech = np.full(DUALTURN_HOP * 3, 0.1, dtype=np.float32)
    silence = np.zeros(DUALTURN_HOP, dtype=np.float32)
    eng.push_user(speech)
    eng.push_user(silence)
    assert eng.ready()
    assert eng.infer(speaking=False) == "ST"
