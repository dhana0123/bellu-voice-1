from __future__ import annotations

import re

CODE = "te"
NAME = "Telugu"
SCRIPT = "Telugu"

_TE = re.compile(r"[\u0C00-\u0C7F]")
_OTHER_INDIC = re.compile(
    r"[\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F"
    r"\u0B80-\u0BFF\u0D00-\u0D7F\u1C50-\u1C7F]"
)
_JUNK_KEYS = (
    "GLOBAL",
    "CONTEXT",
    "TRIGGER",
    "TURNING",
    "ACTIVE STATE",
    "CURRENT SNAPSHOT",
    "ASR transcript",
    "RECENT TRACE",
    "గ్లోబల్",
    "కాంటెక్స్ట్",
    "స్పోకెన్",
    "డైలాగ్",
    "డైలా",
)


def collapse_repeats(text: str, keep: int = 2) -> str:
    parts = (text or "").split()
    if not parts:
        return ""
    out: list[str] = []
    prev = None
    n = 0
    for tok in parts:
        if tok == prev:
            n += 1
            if n <= keep:
                out.append(tok)
        else:
            prev = tok
            n = 1
            out.append(tok)
    return " ".join(out).strip()


def looks_locked(text: str) -> bool:
    """True if text is empty, or Telugu (optional Latin). Drops other Indic scripts."""

    raw = (text or "").strip()
    if not raw:
        return True
    if any(key in raw for key in _JUNK_KEYS):
        return False
    folded = raw.upper()
    if any(key.upper() in folded for key in _JUNK_KEYS if key.isascii()):
        return False
    te = len(_TE.findall(raw))
    other = len(_OTHER_INDIC.findall(raw))
    if other and te == 0:
        return False
    return True


def clean_user_text(text: str) -> str:
    text = collapse_repeats((text or "").strip())
    if len(text) > 220:
        text = text[:220].rstrip()
    if not looks_locked(text):
        return ""
    return text


def clean_spoken(text: str) -> str:
    text = collapse_repeats((text or "").strip())
    text = text.split("\n", 1)[0].strip()
    if "వినేవాడు" in text:
        text = text.split("వినేవాడు", 1)[0].strip()
    if not looks_locked(text):
        return ""
    return text[:80]
