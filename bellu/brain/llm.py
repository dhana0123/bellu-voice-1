from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from bellu.brain.prompts import CONTROLLER_SYSTEM
from bellu.protocol import SpeechCommand
from bellu.types import GlobalState


def _ensure_sarvam_transformers() -> None:
    """Sarvam remote code imports ALL_ATTENTION_FUNCTIONS from modeling_utils (transformers 4.51–4.57)."""

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
        "Sarvam-30B needs transformers 4.51–4.57 (not 4.44 and not 5.x). "
        "Run: pip install 'transformers>=4.51.3,<5'"
    )


class SarvamBrain:
    """Sarvam-30B as the duplex controller. Outputs Speech Protocol JSON, not acoustics."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        _ensure_sarvam_transformers()
        name = self.cfg["model_id"]
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
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

    def decide(self, state: GlobalState, memory_block: str, trigger: str) -> SpeechCommand:
        if self.model is None:
            self.load()
        user = (
            f"TRIGGER: {trigger}\n\n"
            f"{memory_block}\n\n"
            f"CURRENT SNAPSHOT:\n{state.snapshot()}\n\n"
            "ASR transcript:\n"
            f"{state.asr.text}\n\n"
            "STA:\n"
            f"turn={state.sta.turn_state.value} complete={state.sta.turn_completion} "
            f"backchannel={state.sta.backchannel_opportunity} irq={state.sta.interruption_probability}"
        )
        messages = [
            {"role": "system", "content": CONTROLLER_SYSTEM},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(self.cfg.get("enable_thinking", False)),
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        gen = GenerationConfig(
            max_new_tokens=int(self.cfg.get("max_new_tokens", 256)),
            temperature=float(self.cfg.get("temperature", 0.4)),
            top_p=0.9,
            do_sample=True,
        )
        with torch.no_grad():
            out = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                generation_config=gen,
            )
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        return SpeechCommand.from_llm_text(text)

    def chat(self, user_text: str, history: list[dict] | None = None) -> str:
        if self.model is None:
            self.load()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Bellu, a natural conversational assistant for India. "
                    "Reply in the user's language. Be concise and spoken-friendly."
                ),
            }
        ]
        for turn in (history or [])[-16:]:
            messages.append(turn)
        messages.append({"role": "user", "content": user_text})
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        gen = GenerationConfig(
            max_new_tokens=int(self.cfg.get("max_new_tokens", 256)),
            temperature=float(self.cfg.get("temperature", 0.6)),
            top_p=0.9,
            do_sample=True,
        )
        with torch.no_grad():
            out = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                generation_config=gen,
            )
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()
