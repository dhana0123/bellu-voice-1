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
| LLM | What should happen? | [OpenHathi-7B](https://huggingface.co/sarvamai/OpenHathi-7B-Hi-v0.1-Base) (Sarvam 7B) |
| TTS | How should it sound? | [ParlerTTS Mini v1](https://huggingface.co/parler-tts/parler-tts-mini-v1) (streaming) |

The LLM never sets F0, MFCCs, or spectral tilt. It emits a speech-protocol command. TTS turns that into acoustics.

The 80 ms loop always refreshes state. Sarvam is **not** called every tick.

## Setup

Python 3.10+, CUDA GPU strongly recommended. Change `llm.model_id` in `config/default.yaml` if you want another Sarvam checkpoint.

```bash
cd Bellu-Voice-1
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,train,tts]"
huggingface-cli login
```

Extras:

- `pip install -e .` — runtime (ASR / STA / Sarvam / protocol)
- `pip install -e ".[dev]"` — pytest, ruff
- `pip install -e ".[train]"` — Easy-Turn / dataset / tensorboard / wandb
- `pip install -e ".[tts]"` — ParlerTTS
- `pip install -e ".[dev,train,tts]"` — everything

Indic-Transcribe-core is gated: accept the terms on the model card, then log in.

Prepare Easy-Turn (clones the [official repo](https://github.com/ASLP-lab/Easy-Turn), downloads `checkpoint.pt` and `Qwen/Qwen2.5-0.5B-Instruct`):

```bash
python -m bellu.cli setup-sta
```

## Chat UI

```bash
python -m bellu.cli serve --mock          # http://127.0.0.1:8998
python -m bellu.cli serve                 # live OpenHathi-7B + Indic ASR
```

Type in the box, or hold **mic** to talk. Replies come from the backend model.

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
