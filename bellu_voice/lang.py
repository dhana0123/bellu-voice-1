"""Language routing: ASR backend, TTS captions, prompts, backchannels."""

from __future__ import annotations

LANG_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "pt": "Portuguese",
    "ar": "Arabic",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
}

INDIC_CONFORMER = {
    "hi": "ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large",
    "te": "ai4bharat/indicconformer_stt_te_hybrid_ctc_rnnt_large",
    "ta": "ai4bharat/indicconformer_stt_ta_hybrid_ctc_rnnt_large",
    "kn": "ai4bharat/indicconformer_stt_kn_hybrid_ctc_rnnt_large",
}

TTS_DESC: dict[str, str] = {
    "en": (
        "A woman speaks English clearly with a warm, natural conversational tone, "
        "slightly expressive, moderate pace, close microphone, very high quality audio."
    ),
    "te": (
        "Leela speaks Telugu with a natural Indian phone accent, warm and cheerful, "
        "fast pace, high-pitched, casual call-centre energy. The recording is very high quality."
    ),
    "hi": (
        "A woman speaks Hindi clearly with a natural Indian accent, friendly conversational tone, "
        "moderate pace, close microphone, very high quality audio."
    ),
    "ta": (
        "A woman speaks Tamil clearly with a natural Indian accent, friendly conversational tone, "
        "moderate pace, close microphone, very high quality audio."
    ),
    "kn": (
        "A woman speaks Kannada clearly with a natural Indian accent, friendly conversational tone, "
        "moderate pace, close microphone, very high quality audio."
    ),
}

BACKCHANNELS: dict[str, tuple[str, ...]] = {
    "en": ("yeah", "uh-huh", "okay"),
    "hi": ("हाँ", "अच्छा", "ठीक है"),
    "te": ("అవును", "ఉమ్", "సరే"),
    "ta": ("ஆமா", "சரி"),
    "kn": ("ಹೌದು", "ಸರಿ"),
}


def normalize_lang(code: str | None) -> str:
    raw = (code or "en").strip().lower().replace("_", "-")
    raw = raw.split("-")[0]
    aliases = {"tel": "te", "hin": "hi", "tam": "ta", "kan": "kn", "eng": "en"}
    return aliases.get(raw, raw or "en")


def lang_name(code: str) -> str:
    c = normalize_lang(code)
    return LANG_NAMES.get(c, c)


def tts_description(code: str) -> str:
    c = normalize_lang(code)
    if c in TTS_DESC:
        return TTS_DESC[c]
    name = lang_name(c)
    return (
        f"A woman speaks {name} clearly with a natural conversational tone, "
        "moderate pace, close microphone, very high quality audio."
    )


def backchannel_texts(code: str) -> tuple[str, ...]:
    c = normalize_lang(code)
    return BACKCHANNELS.get(c, BACKCHANNELS["en"])


def system_prompt(code: str) -> str:
    name = lang_name(code)
    c = normalize_lang(code)
    if c == "en":
        return (
            "You are Bellu, a helpful voice assistant. Reply in concise spoken English "
            "(one to three short sentences). No markdown, no bullet lists, no stage directions."
        )
    return (
        f"You are Bellu, a helpful voice assistant. Always reply in {name} using the native script. "
        "Match the user if they mix English. Keep answers short for speech "
        "(one to three sentences). No markdown, no bullet lists."
    )


def asr_backend_for(code: str, preference: str) -> str:
    pref = (preference or "auto").lower()
    c = normalize_lang(code)
    if pref in {"whisper", "faster-whisper"}:
        return "whisper"
    if pref in {"conformer", "indic-conformer"}:
        return "conformer"
    if c in INDIC_CONFORMER:
        return "conformer"
    return "whisper"
