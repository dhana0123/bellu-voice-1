"""Moshi binary websocket message helpers."""

from __future__ import annotations


def handshake() -> bytes:
    return b"\x00"


def audio_msg(opus: bytes) -> bytes:
    return b"\x01" + opus


def text_msg(text: str) -> bytes:
    return b"\x02" + text.encode("utf-8")


def error_msg(text: str) -> bytes:
    return b"\x05" + text.encode("utf-8")


def parse_kind(message: bytes) -> tuple[int, bytes]:
    if not message:
        return -1, b""
    return message[0], message[1:]
