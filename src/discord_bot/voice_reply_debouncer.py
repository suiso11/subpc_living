"""Debounce and merge voice-transcript fragments before triggering one LLM reply.

Discord voice STT ends an utterance after a short silence
(`DISCORD_VOICE_STT_SILENCE_MS`, default 700ms). When someone speaks while
thinking, one sentence gets split into several transcript fragments, and the
bot would otherwise fire a separate LLM reply for each fragment.

This debouncer sits between "a transcript fragment arrived" and "reply with the
LLM". It buffers fragments per speaker, waits `debounce_ms` for the next
fragment, merges them, and flushes exactly once — but never waits longer than
`max_ms` after the first fragment (so a speaker who keeps talking still gets an
answer).

Timing logic is intentionally kept free of asyncio internals: it only needs a
`loop` exposing `time()` and `call_later(delay, cb, *args) -> handle` (the real
asyncio loop, or a fake in tests). The actual reply dispatch is delegated to the
`on_flush(message, merged_text)` callback so this class stays synchronous and
deterministic to test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable


@dataclass
class _SpeakerBuffer:
    texts: list[str] = field(default_factory=list)
    first_at: float = 0.0
    message: Any = None
    timer: Any = None


class VoiceReplyDebouncer:
    """Buffer per-speaker transcript fragments and flush a merged reply once."""

    def __init__(
        self,
        *,
        debounce_ms: int,
        max_ms: int,
        on_flush: Callable[[Any, str], Any],
        loop: Any = None,
        joiner: str = "",
    ) -> None:
        self.debounce_ms = max(0, int(debounce_ms))
        self.max_ms = max(0, int(max_ms))
        self._on_flush = on_flush
        self._loop = loop
        self._joiner = joiner
        self._buffers: dict[Hashable, _SpeakerBuffer] = {}

    @property
    def enabled(self) -> bool:
        return self.debounce_ms > 0

    def _get_loop(self) -> Any:
        if self._loop is None:
            import asyncio

            self._loop = asyncio.get_event_loop()
        return self._loop

    def submit(self, speaker_key: Hashable, message: Any, text: str) -> None:
        """Accept a transcript fragment for a speaker.

        When disabled (debounce_ms == 0) this flushes immediately, preserving
        the original fragment-by-fragment reply behaviour.
        """
        text = text or ""
        if not self.enabled:
            self._on_flush(message, text)
            return

        loop = self._get_loop()
        now = loop.time()
        buf = self._buffers.get(speaker_key)
        if buf is None:
            buf = _SpeakerBuffer(first_at=now)
            self._buffers[speaker_key] = buf
        buf.texts.append(text)
        buf.message = message

        if buf.timer is not None:
            buf.timer.cancel()
            buf.timer = None

        delay = self.debounce_ms / 1000.0
        if self.max_ms > 0:
            remaining = self.max_ms / 1000.0 - (now - buf.first_at)
            if remaining < delay:
                delay = remaining

        if delay <= 0:
            # First fragment already older than max_ms (or max_ms tiny): flush now.
            self._fire(speaker_key)
            return
        buf.timer = loop.call_later(delay, self._fire, speaker_key)

    def _fire(self, speaker_key: Hashable) -> None:
        buf = self._buffers.pop(speaker_key, None)
        if buf is None:
            return
        if buf.timer is not None:
            buf.timer.cancel()
            buf.timer = None
        merged = self._joiner.join(t for t in buf.texts if t)
        if not merged:
            return
        self._on_flush(buf.message, merged)

    def flush_all(self) -> None:
        """Flush every pending speaker immediately (e.g. on STT stop)."""
        for speaker_key in list(self._buffers.keys()):
            self._fire(speaker_key)

    def cancel_all(self) -> None:
        """Drop all pending buffers without flushing."""
        for buf in self._buffers.values():
            if buf.timer is not None:
                buf.timer.cancel()
        self._buffers.clear()
