"""One websocket session: DualTurn actions → ASR → LLM → TTS."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import numpy as np
import sphn
from aiohttp import web

from .audio import FRAME, SAMPLE_RATE, split_sentences
from .backchannels import Backchannels
from .dualturn import DualTurnEngine
from .protocol import audio_msg, text_msg

logger = logging.getLogger("bellu_voice.pipeline")

MIN_UTTERANCE = int(0.4 * SAMPLE_RATE)
BC_COOLDOWN = 2.5
ST_COOLDOWN = 0.8


class Session:
    def __init__(
        self,
        ws: web.WebSocketResponse,
        *,
        dualturn: DualTurnEngine,
        asr: Any,
        llm: Any,
        tts: Any,
        backchannels: Backchannels,
        lang: str,
        gpu_lock: asyncio.Lock,
    ):
        self.ws = ws
        self.dualturn = dualturn
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.backchannels = backchannels
        self.lang = lang
        self.gpu_lock = gpu_lock
        self.opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)
        self.opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)
        self.history: list[dict[str, str]] = []
        self.speaking = False
        self.busy = False
        self.cancel = asyncio.Event()
        self.talk_task: asyncio.Task | None = None
        self.user_utt = np.zeros(0, dtype=np.float32)
        self.last_bc = 0.0
        self.last_st = 0.0
        self.pcm_acc: np.ndarray | None = None

    def reset(self) -> None:
        self.dualturn.reset()
        self.history.clear()
        self.speaking = False
        self.busy = False
        self.cancel.clear()
        self.user_utt = np.zeros(0, dtype=np.float32)
        self.pcm_acc = None

    async def send_pcm(self, pcm: np.ndarray) -> None:
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if pcm.size == 0:
            return
        self.dualturn.push_agent(pcm)
        # Stream in 80 ms frames so Opus packetizes like Moshi.
        offset = 0
        while offset < pcm.size:
            if self.cancel.is_set():
                return
            chunk = pcm[offset : offset + FRAME]
            if chunk.size < FRAME:
                chunk = np.pad(chunk, (0, FRAME - chunk.size))
            opus = self.opus_writer.append_pcm(chunk)
            if len(opus) > 0:
                await self.ws.send_bytes(audio_msg(opus))
            offset += FRAME
            await asyncio.sleep(0)

    async def send_text(self, text: str) -> None:
        if not text:
            return
        await self.ws.send_bytes(text_msg(text))

    async def on_user_pcm(self, pcm: np.ndarray) -> None:
        self.dualturn.push_user(pcm)
        self.user_utt = np.concatenate([self.user_utt, pcm])
        if not self.dualturn.ready():
            return
        async with self.gpu_lock:
            action = self.dualturn.infer(self.speaking or self.busy)
        now = time.monotonic()
        if action == "SL" and (self.speaking or self.busy):
            logger.info("DualTurn SL (barge-in)")
            self.cancel.set()
            self.speaking = False
            self.busy = False
            return
        if action == "BC" and not self.speaking and not self.busy and (now - self.last_bc) > BC_COOLDOWN:
            self.last_bc = now
            logger.info("DualTurn BC")
            await self.send_pcm(self.backchannels.next())
            return
        if action == "ST" and not self.speaking and not self.busy and (now - self.last_st) > ST_COOLDOWN:
            if self.user_utt.size < MIN_UTTERANCE:
                return
            self.last_st = now
            utt = self.user_utt.copy()
            self.user_utt = np.zeros(0, dtype=np.float32)
            logger.info("DualTurn ST (%d samples)", utt.size)
            self.busy = True
            if self.talk_task and not self.talk_task.done():
                self.cancel.set()
                try:
                    await self.talk_task
                except Exception:
                    pass
            self.cancel.clear()
            self.talk_task = asyncio.create_task(self._respond(utt))

    async def _respond(self, utt: np.ndarray) -> None:
        loop = asyncio.get_running_loop()
        try:
            async with self.gpu_lock:
                user_text = await loop.run_in_executor(
                    None, self.asr.transcribe, utt, SAMPLE_RATE, self.lang
                )
            if self.cancel.is_set() or not user_text:
                return
            await self.send_text(user_text + " ")
            self.history.append({"role": "user", "content": user_text})
            async with self.gpu_lock:
                reply = await loop.run_in_executor(
                    None, self.llm.reply, list(self.history), self.lang
                )
            if self.cancel.is_set() or not reply:
                return
            self.history.append({"role": "assistant", "content": reply})
            self.speaking = True
            for sent in split_sentences(reply) or [reply]:
                if self.cancel.is_set():
                    break
                await self.send_text(sent + " ")
                async with self.gpu_lock:
                    wav = await loop.run_in_executor(
                        None, self.tts.synthesize, sent, self.lang
                    )
                await self.send_pcm(wav)
        except Exception:
            logger.exception("respond failed")
        finally:
            self.speaking = False
            self.busy = False
            self.cancel.clear()

    async def run(self) -> None:
        self.reset()
        try:
            async for message in self.ws:
                if message.type == web.WSMsgType.ERROR:
                    logger.error("%s", self.ws.exception())
                    break
                if message.type == web.WSMsgType.CLOSED:
                    break
                if message.type != web.WSMsgType.BINARY:
                    continue
                data = message.data
                if not isinstance(data, bytes) or not data:
                    continue
                kind, payload = data[0], data[1:]
                if kind != 1:
                    continue
                pcm = self.opus_reader.append_bytes(payload)
                if pcm is None:
                    continue
                pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
                if pcm.size == 0:
                    continue
                if self.pcm_acc is None:
                    self.pcm_acc = pcm
                else:
                    self.pcm_acc = np.concatenate([self.pcm_acc, pcm])
                while self.pcm_acc.size >= FRAME:
                    chunk = self.pcm_acc[:FRAME]
                    self.pcm_acc = self.pcm_acc[FRAME:]
                    await self.on_user_pcm(chunk)
        finally:
            self.cancel.set()
            if self.talk_task and not self.talk_task.done():
                self.talk_task.cancel()
