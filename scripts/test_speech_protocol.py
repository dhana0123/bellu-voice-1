"""First protocol experiment: can ParlerTTS follow the LLM↔TTS contract?"""

from __future__ import annotations

import argparse

from bellu.expression.tts import ParlerExpression, description_from_protocol
from bellu.protocol import SpeechCommand


CASES = [
    SpeechCommand.from_dict(
        {"action": "SAY", "text": "I already told you this.", "style": {"emotion": "angry", "intensity": 0.8, "pace": 1.15}}
    ),
    SpeechCommand.from_dict(
        {"action": "SAY", "text": "That's actually amazing!", "style": {"emotion": "excited", "intensity": 0.7, "smile": 0.6}}
    ),
    SpeechCommand.from_dict(
        {
            "action": "SAY",
            "text": "I understand why you're frustrated.",
            "style": {"emotion": "empathetic", "intensity": 0.6, "pace": 0.9},
        }
    ),
    SpeechCommand.from_dict({"action": "BACKCHANNEL", "nonverbal": "MM_HMM", "style": {"intensity": 0.35}}),
    SpeechCommand.from_dict(
        {"action": "SAY", "text": "That's really funny.", "style": {"emotion": "amused", "intensity": 0.7, "laugh": 0.45}}
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    tts = ParlerExpression(
        {"model_id": "parler-tts/parler-tts-mini-v1", "device": "cuda", "speaker": "Laura", "play_steps_s": 0.5}
    )
    for i, cmd in enumerate(CASES, 1):
        desc = description_from_protocol(cmd)
        print(f"[{i}] {cmd.action.value} {cmd.text or cmd.nonverbal}")
        print(f"    {desc}")
        if args.print_only:
            continue
        tts.speak(cmd, lambda audio, sr: print(f"    chunk {audio.shape} @ {sr} Hz"))


if __name__ == "__main__":
    main()
