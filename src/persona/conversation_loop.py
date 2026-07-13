"""継続的な自発会話の状態管理。

Discord へ送った会話のきっかけをローカル JSON に保存し、
再起動後の返信文脈、通知上限、無視時のバックオフ、スヌーズを扱う。
ユーザーの返答本文はここには保存しない。
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class ConversationLoopStore:
    """チャンネル単位の自発会話状態をスレッドセーフに保存する。"""

    VERSION = 1

    def __init__(
        self,
        path: str | Path | None,
        *,
        base_interval_sec: float,
        reply_timeout_sec: float,
        daily_limit: int = 1,
        max_backoff_sec: float = 72 * 3600,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.base_interval_sec = max(60.0, float(base_interval_sec))
        self.reply_timeout_sec = max(60.0, float(reply_timeout_sec))
        self.daily_limit = max(1, int(daily_limit))
        self.max_backoff_sec = max(self.base_interval_sec, float(max_backoff_sec))
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {"version": self.VERSION, "channels": {}}
        self.last_error = ""
        self._load()

    @staticmethod
    def _default_channel() -> dict[str, Any]:
        return {
            "pending": None,
            "last_prompt_at": 0.0,
            "last_reply_at": 0.0,
            "ignored_streak": 0,
            "snoozed_until": 0.0,
            "paused": False,
            "daily_date": "",
            "daily_prompt_count": 0,
            "prompt_count": 0,
            "reply_count": 0,
            "controls_shown": False,
        }

    def _channel_locked(self, channel_id: int) -> dict[str, Any]:
        channels = self._data.setdefault("channels", {})
        key = str(int(channel_id))
        raw = channels.get(key)
        if not isinstance(raw, dict):
            raw = {}
            channels[key] = raw
        defaults = self._default_channel()
        for name, value in defaults.items():
            raw.setdefault(name, value)
        return raw

    @staticmethod
    def _local_date(now: float) -> str:
        return datetime.fromtimestamp(now).astimezone().date().isoformat()

    def _roll_daily_locked(self, state: dict[str, Any], now: float) -> bool:
        today = self._local_date(now)
        if state.get("daily_date") == today:
            return False
        state["daily_date"] = today
        state["daily_prompt_count"] = 0
        return True

    def _expire_pending_locked(self, state: dict[str, Any], now: float) -> bool:
        pending = state.get("pending")
        if not isinstance(pending, dict):
            return False
        created_at = float(pending.get("created_at") or 0.0)
        if created_at > 0 and (now - created_at) <= self.reply_timeout_sec:
            return False
        state["pending"] = None
        state["ignored_streak"] = min(8, int(state.get("ignored_streak") or 0) + 1)
        return True

    def can_prompt(self, channel_id: int, *, now: float | None = None) -> bool:
        """現在、会話のきっかけを1件送ってよいかを返す。"""
        now = time.time() if now is None else float(now)
        with self._lock:
            state = self._channel_locked(channel_id)
            changed = self._roll_daily_locked(state, now)
            changed = self._expire_pending_locked(state, now) or changed
            if changed:
                self._save_locked()
            if bool(state.get("paused")):
                return False
            if now < float(state.get("snoozed_until") or 0.0):
                return False
            if isinstance(state.get("pending"), dict):
                return False
            if int(state.get("daily_prompt_count") or 0) >= self.daily_limit:
                return False
            ignored = min(8, max(0, int(state.get("ignored_streak") or 0)))
            interval = min(self.max_backoff_sec, self.base_interval_sec * (2 ** ignored))
            last_prompt = float(state.get("last_prompt_at") or 0.0)
            return last_prompt <= 0 or (now - last_prompt) >= interval

    def record_prompt(
        self,
        channel_id: int,
        prompt: str,
        *,
        now: float | None = None,
    ) -> None:
        now = time.time() if now is None else float(now)
        with self._lock:
            state = self._channel_locked(channel_id)
            self._roll_daily_locked(state, now)
            state["pending"] = {"created_at": now, "prompt": str(prompt)[:1200]}
            state["last_prompt_at"] = now
            state["daily_prompt_count"] = int(state.get("daily_prompt_count") or 0) + 1
            state["prompt_count"] = int(state.get("prompt_count") or 0) + 1
            state["controls_shown"] = True
            self._save_locked()

    def consume_reply(self, channel_id: int, *, now: float | None = None) -> str | None:
        """有効な保留中質問を1度だけ消費し、質問文を返す。"""
        now = time.time() if now is None else float(now)
        with self._lock:
            state = self._channel_locked(channel_id)
            if self._expire_pending_locked(state, now):
                self._save_locked()
                return None
            pending = state.get("pending")
            if not isinstance(pending, dict):
                return None
            prompt = str(pending.get("prompt") or "").strip()
            state["pending"] = None
            state["last_reply_at"] = now
            state["ignored_streak"] = 0
            state["reply_count"] = int(state.get("reply_count") or 0) + 1
            self._save_locked()
            return prompt or None

    def has_pending(self, channel_id: int, *, now: float | None = None) -> bool:
        now = time.time() if now is None else float(now)
        with self._lock:
            state = self._channel_locked(channel_id)
            if self._expire_pending_locked(state, now):
                self._save_locked()
            return isinstance(state.get("pending"), dict)

    def snooze(
        self,
        channel_id: int,
        *,
        until: float,
        clear_pending: bool = True,
    ) -> None:
        with self._lock:
            state = self._channel_locked(channel_id)
            state["snoozed_until"] = max(float(until), time.time())
            if clear_pending:
                state["pending"] = None
            self._save_locked()

    def pause(self, channel_id: int) -> None:
        with self._lock:
            state = self._channel_locked(channel_id)
            state["paused"] = True
            state["pending"] = None
            self._save_locked()

    def resume(self, channel_id: int) -> None:
        with self._lock:
            state = self._channel_locked(channel_id)
            state["paused"] = False
            state["snoozed_until"] = 0.0
            self._save_locked()

    def controls_shown(self, channel_id: int) -> bool:
        with self._lock:
            return bool(self._channel_locked(channel_id).get("controls_shown"))

    def status(self, channel_id: int) -> dict[str, Any]:
        with self._lock:
            state = dict(self._channel_locked(channel_id))
            pending = state.get("pending")
            state["pending"] = isinstance(pending, dict)
            state["next_interval_hours"] = round(
                min(
                    self.max_backoff_sec,
                    self.base_interval_sec
                    * (2 ** min(8, max(0, int(state.get("ignored_streak") or 0)))),
                )
                / 3600,
                1,
            )
            return state

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or not isinstance(raw.get("channels", {}), dict):
                raise ValueError("invalid conversation loop state")
            self._data = {"version": self.VERSION, "channels": raw.get("channels", {})}
        except Exception as exc:
            self.last_error = str(exc)
            self._data = {"version": self.VERSION, "channels": {}}

    def _save_locked(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp, self.path)
            self.last_error = ""
        except Exception as exc:
            # 会話の配送自体は止めず、次回更新で再試行する。
            self.last_error = str(exc)
