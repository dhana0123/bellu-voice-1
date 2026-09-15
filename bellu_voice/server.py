"""Moshi-compatible voice server on port 8998."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import secrets
import sys
import tarfile
from pathlib import Path

from aiohttp import web
from huggingface_hub import hf_hub_download

from .asr import AsrEngine
from .backchannels import Backchannels
from .dualturn import DualTurnEngine
from .lang import normalize_lang
from .llm import DEFAULT_MODEL, LlmEngine
from .pipeline import Session
from .protocol import handshake
from .tts import TtsEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("bellu_voice.server")


class ServerState:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lang = normalize_lang(args.lang)
        self.lock = asyncio.Lock()
        self.gpu_lock = asyncio.Lock()
        mock = args.mock
        device = args.device
        self.dualturn = DualTurnEngine(device=device, mock=mock)
        self.asr = AsrEngine(device=device, preference=args.asr, mock=mock)
        self.llm = LlmEngine(
            device=device,
            model_id=args.llm,
            mock=mock,
            load_in_4bit=args.llm_load_in_4bit,
            strict_llm=args.strict_llm,
            weights=args.llm_weights,
        )
        self.tts = TtsEngine(device=device, mock=mock)
        self.backchannels = Backchannels(self.tts, self.lang, mock=mock)
        log.info("stack ready lang=%s mock=%s", self.lang, mock)

    async def handle_chat(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        lang = normalize_lang(request.query.get("lang", self.lang))
        log.info("accepted connection lang=%s", lang)
        await ws.send_bytes(handshake())
        async with self.lock:
            session = Session(
                ws,
                dualturn=self.dualturn,
                asr=self.asr,
                llm=self.llm,
                tts=self.tts,
                backchannels=self.backchannels,
                lang=lang,
                gpu_lock=self.gpu_lock,
            )
            await session.run()
        log.info("connection closed")
        return ws


def _static_path(arg: str | None) -> str | None:
    if arg == "none":
        return None
    if arg:
        return arg
    log.info("retrieving Moshi static UI")
    dist_tgz = Path(hf_hub_download("kyutai/moshi-artifacts", "dist.tgz"))
    dist = dist_tgz.parent / "dist"
    if not dist.exists():
        with tarfile.open(dist_tgz, "r:gz") as tar:
            tar.extractall(path=dist_tgz.parent)
    return str(dist)


def build_app(state: ServerState, static_path: str | None) -> web.Application:
    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)
    if static_path is not None:
        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        app.router.add_get("/", handle_root)
        app.router.add_static("/", path=static_path, follow_symlinks=True, name="static")
    return app


def main() -> None:
    p = argparse.ArgumentParser("bellu-voice")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default=8998, type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--lang", default="en", help="Default session language (en, te, hi, …)")
    p.add_argument("--asr", default="auto", choices=["auto", "whisper", "conformer"])
    p.add_argument("--llm", default=DEFAULT_MODEL)
    p.add_argument(
        "--llm-weights",
        default="bf16",
        choices=["bf16", "gguf"],
        help="bf16 = transformers Sarvam-30B (already in your HF cache). gguf needs llama-cpp-python.",
    )
    p.add_argument("--llm-load-in-4bit", action="store_true")
    p.add_argument(
        "--strict-llm",
        action="store_true",
        help="Fail if sarvam-30b cannot load (do not fall back to sarvam-m).",
    )
    p.add_argument("--static", default=None)
    p.add_argument("--mock", action="store_true", help="Energy VAD + dummy ASR/LLM/TTS (no GPU weights)")
    p.add_argument("--gradio-tunnel", action="store_true")
    p.add_argument("--gradio-tunnel-token", default=None)
    args = p.parse_args()

    if args.device.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available() and not args.mock:
                log.warning("CUDA not available; using cpu")
                args.device = "cpu"
        except ImportError:
            if not args.mock:
                log.error("PyTorch is required")
                sys.exit(1)

    state = ServerState(args)
    static_path = _static_path(args.static)
    app = build_app(state, static_path)

    setup_tunnel = None
    tunnel_token = ""
    if args.gradio_tunnel:
        from gradio import networking

        setup_tunnel = networking.setup_tunnel
        tunnel_token = args.gradio_tunnel_token or secrets.token_urlsafe(32)

    log.info("Web UI at http://%s:%s", args.host, args.port)
    if setup_tunnel is not None:
        tunnel = setup_tunnel("localhost", args.port, tunnel_token, None)
        log.info("Tunnel: %s", tunnel)

    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
