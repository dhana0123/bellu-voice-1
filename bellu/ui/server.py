from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bellu.chat import ChatSession
from bellu.perception.audio import resample_mono

STATIC = Path(__file__).resolve().parent / "static"


class ChatIn(BaseModel):
    text: str


def create_app(session: ChatSession, mode: str) -> FastAPI:
    app = FastAPI(title="Bellu")
    clients: set[WebSocket] = set()
    loop_holder: dict[str, Any] = {}

    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health():
        return {"ok": True, "mode": mode, "model": "sarvamai/sarvam-30b" if mode == "live" else "mock"}

    @app.post("/api/chat")
    def chat(payload: ChatIn):
        reply = session.reply(payload.text)
        return {"reply": reply}

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        clients.add(socket)
        loop_holder["loop"] = asyncio.get_running_loop()
        sr_in = 16000
        try:
            await socket.send_json({"type": "hello", "mode": mode})
            while True:
                message = await socket.receive()
                if message.get("text"):
                    import json

                    data = json.loads(message["text"])
                    kind = data.get("type")
                    if kind == "hello":
                        sr_in = int(data.get("sr") or 16000)
                    elif kind == "chat":
                        reply = await asyncio.to_thread(session.reply, data.get("text") or "")
                        await socket.send_json({"type": "reply", "text": reply})
                elif message.get("bytes"):
                    pcm = np.frombuffer(message["bytes"], dtype=np.int16).astype(np.float32) / 32768.0
                    wav = resample_mono(pcm, sr_in, 16000)
                    text = await asyncio.to_thread(_transcribe, session, wav)
                    if text:
                        await socket.send_json({"type": "transcript", "text": text})
                        reply = await asyncio.to_thread(session.reply, text)
                        await socket.send_json({"type": "reply", "text": reply})
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(socket)

    return app


def _transcribe(session: ChatSession, wav: np.ndarray) -> str:
    if session.asr is None:
        return ""
    from time import time

    state = session.asr.transcribe(wav, 16000, time())
    return (state.text or "").strip()
