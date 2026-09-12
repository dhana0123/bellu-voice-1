# Bellu Voice

Continuous full-duplex spoken dialogue as specialized modules with stable interfaces.

```text
MIC → ASR (what) + STA (what is happening)
        → Global state + temporal memory
        → Event loop (~80 ms)
        → LLM only on meaningful events (what should happen)
        → Speech protocol
        → Streaming TTS (how it sounds)
```

## Models in v1

| Job | Module | Model |
| --- | --- | --- |
| ASR | What did they say? | [Indic-Transcribe-core](https://huggingface.co/bodhan-ai/indic-transcribe-core) (25 Indian languages, gated HF repo) |
| STA | Turn / overlap / backchannel | [Easy-Turn](https://huggingface.co/ASLP-lab/Easy-Turn) (`COMPLETE`, `INCOMPLETE`, `BACKCHANNEL`, `WAIT`) |
| LLM | What should happen? | [Sarvam-30B](https://huggingface.co/sarvamai/sarvam-30b) |
| TTS | How should it sound? | [ParlerTTS Mini v1](https://huggingface.co/parler-tts/parler-tts-mini-v1) (streaming) |

The LLM never sets F0, MFCCs, or spectral tilt. It emits a speech-protocol command. TTS turns that into acoustics.

The 80 ms loop always refreshes state. Sarvam is **not** called every tick.

## Setup

Python 3.10+, CUDA GPU strongly recommended. Sarvam-30B + Easy-Turn + ASR + TTS will not fit a small GPU together; use multiple devices via `config/default.yaml` if needed.

```bash
cd Bellu-Voice-1
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
huggingface-cli login
```

Indic-Transcribe-core is gated: accept the terms on the model card, then log in.

Prepare Easy-Turn (clones the [official repo](https://github.com/ASLP-lab/Easy-Turn), downloads `checkpoint.pt` and `Qwen/Qwen2.5-0.5B-Instruct`):

```bash
python -m bellu.cli setup-sta
```

## Run

Architecture dry-run (no large weights):

```bash
python -m bellu.cli mock
```

Live duplex:

```bash
python -m bellu.cli run
```

Speech-protocol experiment (does ParlerTTS follow emotion / pace / backchannel?):

```bash
python scripts/test_speech_protocol.py --print-only
python scripts/test_speech_protocol.py
```

## Layout

```text
bellu/
  perception/   ASR + Easy-Turn STA + mic ring buffer
  brain/        Sarvam controller → speech protocol JSON
  expression/   ParlerTTS streaming + cancel
  orchestrator  80 ms loop, gating, interruption
  protocol.py   SAY / BACKCHANNEL / WAIT / STOP / INTERRUPT / CONTINUE
```

## Tests

```bash
python -m pytest tests -q
```
