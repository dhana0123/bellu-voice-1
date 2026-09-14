from __future__ import annotations

from typing import Any
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from bellu.brain.prompts import CONTROLLER_SYSTEM
from bellu.language import clean_spoken
from bellu.log import clip, clog
from bellu.protocol import SpeechCommand
from bellu.types import Action, GlobalState

SYSTEM_CHAT = (
    "You are Bellu. Speak only Telugu (తెలుగు). One or two short spoken sentences. "
    "Never Hindi. Never English. Never repeat a word more than twice."
)


def _needs_moe_compat(model_id: str) -> bool:
    mid = model_id.lower()
    return "sarvam-30b" in mid or "sarvam-105b" in mid


def _ensure_sarvam_moe_transformers() -> None:
    """Sarvam MoE remote code needs ALL_ATTENTION_FUNCTIONS (transformers 4.51–4.57)."""

    import transformers.modeling_utils as modeling_utils

    if hasattr(modeling_utils, "ALL_ATTENTION_FUNCTIONS"):
        return
    for path in (
        "transformers.masking_utils",
        "transformers.modeling_flash_attention_utils",
        "transformers.integrations.sdpa_attention",
        "transformers",
    ):
        try:
            module = __import__(path, fromlist=["ALL_ATTENTION_FUNCTIONS", "AttentionInterface"])
        except Exception:
            continue
        obj = getattr(module, "ALL_ATTENTION_FUNCTIONS", None) or getattr(module, "AttentionInterface", None)
        if obj is not None:
            modeling_utils.ALL_ATTENTION_FUNCTIONS = obj
            return
    raise ImportError(
        "This Sarvam MoE model needs transformers 4.51–4.57. "
        "Run: pip install 'transformers>=4.51.3,<5'"
    )


def _format_prompt(tokenizer, messages: list[dict], enable_thinking: bool = False) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            pass
    except Exception:
        pass

    parts: list[str] = []
    for turn in messages:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "system":
            parts.append(f"### System:\n{content}\n")
        elif role == "assistant":
            parts.append(f"### Assistant:\n{content}\n")
        else:
            parts.append(f"### User:\n{content}\n")
    parts.append("### Assistant:\n")
    return "\n".join(parts)


class SarvamBrain:
    """Sarvam LLM brain (OpenHathi-7B by default). Emits speech-protocol or chat text."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        name = self.cfg["model_id"]
        if _needs_moe_compat(name):
            _ensure_sarvam_moe_transformers()
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }
        device = self.cfg.get("device", "auto")
        if device == "auto":
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        if device not in {"auto", None} and device != "auto":
            self.model.to(device)

    def _generate(self, prompt: str, temperature: float) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        gen_kwargs = dict(
            max_new_tokens=int(self.cfg.get("max_new_tokens", 48)),
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=float(self.cfg.get("repetition_penalty", 1.25)),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen = GenerationConfig(**gen_kwargs)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                generation_config=gen,
            )
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()
        clog("llm", clip(text, 220) or "(empty)")
        return text

    def stream_reply(self, asr: str, on_chunk) -> None:
        """DuplexCascade: stream LLM tokens; flush ~10-token micro-turns into TTS."""

        from transformers import TextIteratorStreamer

        if self.model is None:
            self.load()
        asr = (asr or "").strip()
        if not asr:
            return
        prompt = f"{CONTROLLER_SYSTEM}వినేవాడు: {asr}\nస్నేహితుడు:"
        clog("llm", f"stream asr={clip(asr, 80)!r}")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=int(self.cfg.get("max_new_tokens", 24)),
            temperature=float(self.cfg.get("temperature", 0.3)),
            top_p=0.9,
            do_sample=True,
            repetition_penalty=float(self.cfg.get("repetition_penalty", 1.25)),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()
        buf = ""
        for tok in streamer:
            if not tok:
                continue
            buf += tok
            if any(ch in tok for ch in "।.?!\n") and len(clean_spoken(buf)) >= 4:
                piece = clean_spoken(buf)
                buf = ""
                if piece:
                    clog("llm", f"phrase→tts {clip(piece)}")
                    on_chunk(piece)
                break
        piece = clean_spoken(buf)
        if piece and len(piece) >= 4:
            clog("llm", f"phrase→tts {clip(piece)}")
            on_chunk(piece)
        thread.join(timeout=1.0)

    def decide(self, state: GlobalState, memory_block: str, trigger: str) -> SpeechCommand:
        if self.model is None:
            self.load()
        asr = (state.asr.text or "").strip()
        if not asr:
            return SpeechCommand.wait("no user text")
        prompt = f"{CONTROLLER_SYSTEM}వినేవాడు: {asr}\nస్నేహితుడు:"
        clog("llm", f"reply asr={clip(asr, 80)!r}")
        raw = self._generate(prompt, float(self.cfg.get("temperature", 0.3)))
        spoken = clean_spoken(raw)
        if not spoken or spoken == asr:
            return SpeechCommand.wait("empty_or_echo")
        cmd = SpeechCommand(action=Action.SAY, text=spoken, reason="telugu_reply")
        clog("llm", f"parsed SAY text={clip(cmd.text, 80)!r}")
        return cmd

    def chat(self, user_text: str, history: list[dict] | None = None) -> str:
        if self.model is None:
            self.load()
        messages = [{"role": "system", "content": SYSTEM_CHAT}]
        for turn in (history or [])[-16:]:
            messages.append(turn)
        messages.append({"role": "user", "content": user_text})
        prompt = _format_prompt(self.tokenizer, messages, enable_thinking=False)
        return self._generate(prompt, float(self.cfg.get("temperature", 0.6)))
