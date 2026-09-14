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

from bellu.log import clip, clog
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
        pcm = resample_mono(np.asarray(audio, dtype=np.float32).reshape(-1), sr, 16000)
        peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
        raw = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        clog("tts", f"audio {len(raw)} bytes peak={peak:.3f} sr={sr}")
        if pcm.size == 0 or peak < 0.02:
            clog("tts", f"skip quiet chunk peak={peak:.3f}")
            return
        with audio_lock:
            audio_out.append(raw)
        loop = loop_holder.get("loop")
        ws = loop_holder.get("ws")
        if loop and ws:
            asyncio.run_coroutine_threadsafe(_flush_audio(ws, audio_out, audio_lock), loop)
        else:
            clog("tts", "no websocket to send audio")

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
        clog("http", f"text {clip(payload.text)} -> {clip(spoken_text(cmd) if cmd else '')}")
        return {"reply": spoken_text(cmd) if cmd else ""}

    @app.websocket("/api/chat")
    @app.websocket("/ws")
    async def duplex_ws(socket: WebSocket):
        await socket.accept()
        loop_holder["loop"] = asyncio.get_running_loop()
        loop_holder["ws"] = socket
        sr_in = 48000
        audio_packets = 0
        clog("ws", "client connected")
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
                        clog("ws", f"hello sr={sr_in}")
                    elif kind == "chat":
                        clog("ws", f"chat {clip(data.get('text'))}")
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
                        clog("ws", "interrupt")
                        runtime.interrupt()
                        await socket.send_json({"type": "state", **_thin_state(runtime)})
                    elif kind == "ping":
                        await socket.send_json({"type": "pong", **_thin_state(runtime)})
                elif message.get("bytes") is not None:
                    pcm = np.frombuffer(message["bytes"], dtype=np.int16).astype(np.float32) / 32768.0
                    wav = resample_mono(pcm, sr_in, runtime.cfg["sample_rate"])
                    runtime.push_audio(wav)
                    audio_packets += 1
                    if audio_packets == 1 or audio_packets % 50 == 0:
                        clog("ws", f"mic packets={audio_packets} rms={runtime.ring.rms:.4f}")
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
            clog("ws", "client disconnected")
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
