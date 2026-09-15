"""Sarvam-30B chat LLM."""

from __future__ import annotations

import logging
from typing import Any

from .lang import system_prompt

logger = logging.getLogger("bellu_voice.llm")

DEFAULT_MODEL = "sarvamai/sarvam-30b"


class LlmEngine:
    def __init__(
        self,
        device: str,
        model_id: str = DEFAULT_MODEL,
        mock: bool = False,
        load_in_4bit: bool = False,
    ):
        self.device = device
        self.model_id = model_id
        self.mock = mock
        self.load_in_4bit = load_in_4bit
        self.tok = None
        self.model = None
        if not mock:
            self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading LLM %s …", self.model_id)
        self.tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self.model.eval()
        logger.info("LLM ready")

    def reply(self, history: list[dict[str, str]], lang: str) -> str:
        if self.mock:
            last = history[-1]["content"] if history else ""
            return f"You said: {last}" if last else "Hello, I am Bellu."
        messages = [{"role": "system", "content": system_prompt(lang)}] + history
        tok = self.tok
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            text = tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            text = tok.apply_chat_template(messages, **kwargs)
        inputs = tok([text], return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        import torch

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
            )
        gen = out[0, inputs["input_ids"].shape[-1] :]
        return tok.decode(gen, skip_special_tokens=True).strip()
