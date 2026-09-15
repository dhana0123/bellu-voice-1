"""Sarvam chat LLM (default: sarvam-30b)."""

from __future__ import annotations

import logging
from typing import Any

from .lang import system_prompt

logger = logging.getLogger("bellu_voice.llm")

DEFAULT_MODEL = "sarvamai/sarvam-30b"
FALLBACK_LLM = "sarvamai/sarvam-m"

TRANSFORMERS_MIN_HINT = (
    "sarvamai/sarvam-30b needs transformers>=4.57 "
    "(ALL_ATTENTION_FUNCTIONS is missing).\n\n"
    "Parler-TTS often downgrades transformers. In this venv run:\n"
    '  pip install -U "transformers>=4.57.0" accelerate\n'
    "Then restart: python -m bellu_voice --host 127.0.0.1 --port 8998\n\n"
    "If Parler breaks after the upgrade, either keep two venvs or pass:\n"
    f"  --llm {FALLBACK_LLM}\n"
)


def _has_all_attention_functions() -> bool:
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: F401

        return True
    except ImportError:
        pass
    try:
        from transformers.modeling_flash_attention_utils import (  # noqa: F401
            ALL_ATTENTION_FUNCTIONS,
        )

        return True
    except ImportError:
        return False


def resolve_llm_model(model_id: str, *, strict_llm: bool = False) -> str:
    """Map sarvam-30b → sarvam-m when transformers is too old."""
    if "sarvam-30b" not in model_id.lower():
        return model_id
    import transformers

    if _has_all_attention_functions():
        return model_id
    ver = getattr(transformers, "__version__", "?")
    if strict_llm:
        raise SystemExit(
            f"{TRANSFORMERS_MIN_HINT}\nInstalled transformers=={ver}"
        )
    logger.warning(
        "sarvam-30b needs transformers>=4.57 (have %s). Falling back to %s. "
        "Fix: pip install -U \"transformers>=4.57.0\" accelerate",
        ver,
        FALLBACK_LLM,
    )
    return FALLBACK_LLM


class LlmEngine:
    def __init__(
        self,
        device: str,
        model_id: str = DEFAULT_MODEL,
        mock: bool = False,
        load_in_4bit: bool = False,
        strict_llm: bool = False,
    ):
        self.device = device
        self.model_id = resolve_llm_model(model_id, strict_llm=strict_llm)
        self.mock = mock
        self.load_in_4bit = load_in_4bit
        self.tok = None
        self.model = None
        if not mock:
            self._load()

    def _load(self) -> None:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(
            "Loading LLM %s (transformers %s) …",
            self.model_id,
            getattr(transformers, "__version__", "?"),
        )
        self.tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        except ImportError as exc:
            if "ALL_ATTENTION_FUNCTIONS" in str(exc) and "sarvam-30b" in self.model_id.lower():
                raise SystemExit(
                    f"{TRANSFORMERS_MIN_HINT}\nUnderlying error: {exc}"
                ) from exc
            raise
        self.model.eval()
        logger.info("LLM ready (%s)", self.model_id)

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
