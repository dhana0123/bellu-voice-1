"""Sarvam chat LLM. Default: 30B GGUF Q4 (~20 GB) to fit typical disks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from .hub import require_free_gb, snapshot
from .lang import system_prompt

logger = logging.getLogger("bellu_voice.llm")

DEFAULT_MODEL = "sarvamai/sarvam-30b"
GGUF_REPO = "sarvamai/sarvam-30b-gguf"
FALLBACK_LLM = "sarvamai/sarvam-m"
Weights = Literal["gguf", "bf16"]

TRANSFORMERS_MIN_HINT = (
    "sarvamai/sarvam-30b (bf16) needs transformers>=4.57 "
    "(ALL_ATTENTION_FUNCTIONS is missing).\n\n"
    "Parler-TTS often downgrades transformers. In this venv run:\n"
    '  pip install -U "transformers>=4.57.0" accelerate\n'
    "Or skip bf16 and use the 20 GB GGUF build:\n"
    "  python -m bellu_voice --llm-weights gguf\n"
)


def _has_all_attention_functions() -> bool:
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_llm_model(model_id: str, *, strict_llm: bool = False, weights: Weights = "gguf") -> str:
    if weights == "gguf":
        return model_id
    if "sarvam-30b" not in model_id.lower() or model_id.endswith("-gguf"):
        return model_id
    import transformers

    if _has_all_attention_functions():
        return model_id
    ver = getattr(transformers, "__version__", "?")
    if strict_llm:
        raise SystemExit(f"{TRANSFORMERS_MIN_HINT}\nInstalled transformers=={ver}")
    logger.warning(
        "sarvam-30b bf16 needs transformers>=4.57 (have %s). Falling back to %s.",
        ver,
        FALLBACK_LLM,
    )
    return FALLBACK_LLM


def _gguf_shard(local_dir: str) -> str:
    root = Path(local_dir)
    shards = sorted(root.glob("*.gguf-00001-of-*.gguf")) + sorted(root.glob("*00001-of-*.gguf"))
    if not shards:
        shards = sorted(root.glob("*.gguf"))
    if not shards:
        raise SystemExit(f"No GGUF files in {local_dir}")
    return str(shards[0])


class LlmEngine:
    def __init__(
        self,
        device: str,
        model_id: str = DEFAULT_MODEL,
        mock: bool = False,
        load_in_4bit: bool = False,
        strict_llm: bool = False,
        weights: Weights = "gguf",
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
    ):
        self.device = device
        self.weights = weights
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers if device.startswith("cuda") else 0
        self.model_id = resolve_llm_model(model_id, strict_llm=strict_llm, weights=weights)
        self.mock = mock
        self.load_in_4bit = load_in_4bit
        self.tok = None
        self.model = None
        self._gguf = None
        if not mock:
            if weights == "gguf" or str(self.model_id).endswith("-gguf"):
                self._load_gguf()
            else:
                self._load_transformers()

    def _load_gguf(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise SystemExit(
                "Sarvam-30B GGUF needs llama-cpp-python (~20 GB download, not 129 GB).\n\n"
                "CPU:\n  pip install llama-cpp-python\n"
                "CUDA:\n  CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python "
                "--force-reinstall --no-cache-dir\n"
            ) from exc

        repo = GGUF_REPO if "sarvam-30b" in self.model_id.lower() else self.model_id
        require_free_gb(25)
        local = snapshot(repo, allow_patterns=["*.gguf", "*.md", "*.json"])
        path = _gguf_shard(local)
        logger.info("Loading GGUF %s (n_gpu_layers=%s) …", path, self.n_gpu_layers)
        self._gguf = Llama(
            model_path=path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )
        self.model_id = repo
        logger.info("LLM ready (GGUF %s)", repo)

    def _load_transformers(self) -> None:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        need = 160.0 if "sarvam-30b" in self.model_id.lower() else 50.0
        require_free_gb(need)
        logger.info(
            "Loading LLM %s (transformers %s) …",
            self.model_id,
            getattr(transformers, "__version__", "?"),
        )
        local = snapshot(self.model_id)
        self.tok = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
        kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(local, **kwargs)
        except ImportError as exc:
            if "ALL_ATTENTION_FUNCTIONS" in str(exc):
                raise SystemExit(f"{TRANSFORMERS_MIN_HINT}\nUnderlying error: {exc}") from exc
            raise
        except OSError as exc:
            if getattr(exc, "errno", None) == 28 or "No space left" in str(exc):
                require_free_gb(need + 20)
            raise
        self.model.eval()
        logger.info("LLM ready (%s)", self.model_id)

    def reply(self, history: list[dict[str, str]], lang: str) -> str:
        if self.mock:
            last = history[-1]["content"] if history else ""
            return f"You said: {last}" if last else "Hello, I am Bellu."
        messages = [{"role": "system", "content": system_prompt(lang)}] + history
        if self._gguf is not None:
            out = self._gguf.create_chat_completion(
                messages=messages,
                max_tokens=128,
                temperature=0.4,
                top_p=0.9,
            )
            return str(out["choices"][0]["message"]["content"]).strip()
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
