from __future__ import annotations

import asyncio
import json
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bellu.perception.audio import resample_mono
from bellu.protocol import spoken_text

STATIC = Path(__file__).resolve().parent / "static"


class ChatIn(BaseModel):
    text: str


def create_app(runtime, mode: str, model_id: str | None = None) -> FastAPI:
    """Moshi-style continuous duplex: browser streams mic PCM, server streams state + TTS."""

    app = FastAPI(title="Bellu")
    live_model = model_id or "sarvamai/OpenHathi-7B-Hi-v0.1-Base"
    audio_out: deque[bytes] = deque(maxlen=64)
    audio_lock = Lock()
    loop_holder: dict[str, Any] = {"loop": None, "ws": None}

    def on_tts(audio: np.ndarray, sr: int) -> None:
        pcm = resample_mono(audio, sr, 16000)
        raw = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        with audio_lock:
            audio_out.append(raw)
        loop = loop_holder.get("loop")
        ws = loop_holder.get("ws")
        if loop and ws:
            asyncio.run_coroutine_threadsafe(_flush_audio(ws, audio_out, audio_lock), loop)

    runtime.on_tts = on_tts
    runtime.mode = mode

    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health():
        view = runtime.view()
        return {
            "ok": True,
            "mode": mode,
            "model": live_model if mode == "live" else "mock",
            "running": view["running"],
        }

    @app.post("/api/text")
    def chat(payload: ChatIn):
        runtime.ingest_text(payload.text)
        cmd = runtime.last_command
        return {"reply": spoken_text(cmd) if cmd else ""}

    @app.websocket("/api/chat")
    @app.websocket("/ws")
    async def duplex_ws(socket: WebSocket):
        await socket.accept()
        loop_holder["loop"] = asyncio.get_running_loop()
        loop_holder["ws"] = socket
        sr_in = 48000
        runtime.start(use_microphone=False)
        last_log = 0
        try:
            await socket.send_json({"type": "hello", "mode": mode, "model": live_model, "sr": 16000})
            while True:
                message = await socket.receive()
                if message.get("text") is not None:
                    data = json.loads(message["text"])
                    kind = data.get("type")
                    if kind == "hello":
                        sr_in = int(data.get("sr") or 48000)
                    elif kind == "chat":
                        await asyncio.to_thread(runtime.ingest_text, data.get("text") or "")
                        cmd = runtime.last_command
                        await socket.send_json(
                            {
                                "type": "reply",
                                "text": spoken_text(cmd) if cmd else "",
                                "trigger": runtime.last_trigger,
                            }
                        )
                    elif kind == "interrupt":
                        runtime.interrupt()
                        await socket.send_json({"type": "state", **_thin_state(runtime)})
                    elif kind == "ping":
                        await socket.send_json({"type": "pong", **_thin_state(runtime)})
                elif message.get("bytes") is not None:
                    pcm = np.frombuffer(message["bytes"], dtype=np.int16).astype(np.float32) / 32768.0
                    wav = resample_mono(pcm, sr_in, runtime.cfg["sample_rate"])
                    runtime.push_audio(wav)
                    view = runtime.view()
                    log = view.get("log") or []
                    if len(log) > last_log:
                        for item in log[last_log:]:
                            await socket.send_json({"type": "log", **item})
                        last_log = len(log)
                    snap = view.get("snapshot") or {}
                    user = snap.get("user") or {}
                    await socket.send_json(
                        {
                            "type": "state",
                            "rms": view.get("rms"),
                            "transcript": user.get("transcript") or "",
                            "turn": user.get("turn_state"),
                            "assistant": (snap.get("assistant") or {}).get("current_action"),
                            "speaking_user": user.get("speaking"),
                            "speaking_bot": (snap.get("assistant") or {}).get("speaking"),
                            "last_text": runtime.state.assistant.last_text or "",
                            "trigger": runtime.last_trigger,
                        }
                    )
                    await _flush_audio(socket, audio_out, audio_lock)
        except WebSocketDisconnect:
            pass
        finally:
            if loop_holder.get("ws") is socket:
                loop_holder["ws"] = None

    return app


def _thin_state(runtime) -> dict[str, Any]:
    view = runtime.view()
    snap = view.get("snapshot") or {}
    user = snap.get("user") or {}
    return {
        "rms": view.get("rms"),
        "transcript": user.get("transcript") or "",
        "running": view.get("running"),
        "last_text": runtime.state.assistant.last_text or "",
    }


async def _flush_audio(socket: WebSocket, queue: deque, lock: Lock) -> None:
    while True:
        with lock:
            if not queue:
                return
            chunk = queue.popleft()
        await socket.send_bytes(chunk)
