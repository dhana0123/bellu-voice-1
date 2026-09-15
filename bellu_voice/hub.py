"""Hugging Face cache: sequential downloads, disk checks, temp cleanup."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger("bellu_voice.hub")


def configure_hub(hf_home: str | None = None, *, disable_xet: bool = True) -> Path:
    """Point caches at hf_home and avoid Xet (uses extra disk while reconstructing)."""
    if hf_home:
        home = Path(hf_home).expanduser().resolve()
        home.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(home)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(home / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(home / "transformers")
    if disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return cache


def cache_root() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def free_bytes(path: Path | None = None) -> int:
    return shutil.disk_usage(str(path or cache_root())).free


def require_free_gb(need_gb: float, *, path: Path | None = None) -> None:
    root = path or cache_root()
    free = free_bytes(root)
    have = free / (1024**3)
    if have + 0.5 < need_gb:
        raise SystemExit(
            f"Not enough disk on {root} (need ~{need_gb:.0f} GB free, have {have:.1f} GB).\n\n"
            "Sarvam-30B bf16 is ~129 GB plus temp files. Use the GGUF build (~20 GB) instead:\n"
            "  python -m bellu_voice --llm-weights gguf --host 127.0.0.1 --port 8998\n\n"
            "Or point the cache at a bigger disk:\n"
            "  python -m bellu_voice --hf-home /data/hf --llm-weights gguf\n\n"
            "Reclaim the failed bf16 download:\n"
            "  rm -rf ~/.cache/huggingface/hub/models--sarvamai--sarvam-30b\n"
            "  find ~/.cache/huggingface -name '*.incomplete' -delete\n"
        )


def cleanup_incomplete(root: Path | None = None) -> int:
    root = root or cache_root()
    n = 0
    if not root.exists():
        return 0
    for pattern in ("*.incomplete", "*.lock"):
        for p in root.rglob(pattern):
            try:
                p.unlink()
                n += 1
            except OSError:
                pass
    if n:
        logger.info("removed %s incomplete/lock files under %s", n, root)
    return n


def snapshot(
    repo_id: str,
    *,
    allow_patterns: list[str] | None = None,
    max_workers: int = 1,
) -> str:
    from huggingface_hub import snapshot_download

    logger.info("Downloading %s (sequential, resume-safe) …", repo_id)
    kwargs = {
        "repo_id": repo_id,
        "resume_download": True,
        "max_workers": max_workers,
    }
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    try:
        return snapshot_download(**kwargs)
    except TypeError:
        kwargs.pop("max_workers", None)
        return snapshot_download(**kwargs)
