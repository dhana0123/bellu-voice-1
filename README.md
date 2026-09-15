# Bellu-Voice-1

Modular duplex voice agent with a **Moshi web UI** on port **8998**.

Turn-taking is [DualTurn](https://arxiv.org/abs/2603.08216) (`anyreach-ai/dualturn-qwen2.5-mimi-0.5B`). Speech content is ASR → **Sarvam-30B** → **Indic Parler TTS**. Default language is **English**; pass `--lang` or `?lang=` for others.

## H200 + SSH

On the GPU box:

```bash
cd Bellu-Voice-1
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install "git+https://github.com/huggingface/parler-tts.git"
# IndicConformer (optional, hi/te/ta/kn):
# pip install "nemo_toolkit[asr]"

python -m bellu_voice --host 127.0.0.1 --port 8998 --lang en --device cuda
```

From your laptop (mic only works on localhost HTTP):

```bash
ssh -L 8998:127.0.0.1:8998 user@h200-host
```

Open [http://localhost:8998](http://localhost:8998).

### Other languages

```bash
python -m bellu_voice --lang te          # Telugu (IndicConformer if NeMo is installed)
python -m bellu_voice --lang hi
# or keep English server default and override per session:
# ws://host:8998/api/chat?lang=te
```

`--asr auto` uses IndicConformer for `hi`/`te`/`ta`/`kn` and Whisper `large-v3` otherwise. Force Whisper with `--asr whisper`.

If Sarvam-30B VRAM is tight: `--llm-load-in-4bit`.

`--mock` runs energy VAD + dummy ASR/LLM/TTS (no weight downloads).

Sarvam-30B needs `transformers>=4.57`. If Parler fails after that upgrade, install TTS in a second process/venv (same conflict as the duplex data README).

## Stack

| Piece | Default |
| --- | --- |
| UI | Moshi static bundle (`kyutai/moshi-artifacts`), Opus WS `/api/chat` |
| Turn-taking | DualTurn 0.5B, 240 ms hop, actions ST/CL/SL/CT/BC |
| ASR | faster-whisper large-v3 (English + all Whisper langs) |
| LLM | `sarvamai/sarvam-30b` |
| TTS | `ai4bharat/indic-parler-tts` |

DualTurn was trained on English conversational audio. Other languages still use it as an acoustic policy.
