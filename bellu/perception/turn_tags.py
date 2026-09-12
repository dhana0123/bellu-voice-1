from __future__ import annotations

import re

from bellu.types import TurnState

TAGS = {
    "<COMPLETE>": TurnState.COMPLETE,
    "<INCOMPLETE>": TurnState.INCOMPLETE,
    "<BACKCHANNEL>": TurnState.BACKCHANNEL,
    "<WAIT>": TurnState.WAIT,
}


def parse_turn(text) -> tuple[TurnState, str, str]:
    if isinstance(text, (list, tuple)):
        text = text[0]
    raw = str(text or "").strip()
    state = TurnState.UNKNOWN
    tag = ""
    upper = raw.upper()
    for token, value in TAGS.items():
        name = token.strip("<>")
        if token in upper or re.search(rf"\b{name}\b", upper):
            state = value
            tag = token
            cleaned = re.sub(re.escape(token), "", raw, flags=re.I)
            cleaned = re.sub(rf"<?{name}>?", "", cleaned, flags=re.I)
            raw = cleaned.strip(" -:|,")
            break
    return state, tag, raw
