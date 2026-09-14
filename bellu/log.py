from __future__ import annotations

from time import strftime


def clog(tag: str, msg: str) -> None:
    """One-line stdout log for local development."""

    print(f"[{strftime('%H:%M:%S')}] {tag:<8} {msg}", flush=True)


def clip(text: object, n: int = 160) -> str:
    s = " ".join(str(text or "").split())
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"
