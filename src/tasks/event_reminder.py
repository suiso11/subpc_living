"""予定 (Google Calendar イベント) の開始前リマインドエンジン。

TaskReminderEngine (期限ベースのエスカレーション) と対になる、開始時刻ベースの
単発リマインド。CalendarPullWorker が定期更新する upcoming.json を読み、
開始 lead_min 分前〜開始時刻の間に一度だけ
callback(trigger_type="event_remind", message=..., event_id=..., title=...) を呼ぶ。

- 終日イベント (start に時刻が無い) は対象外。
- 重複防止: tasks.db の event_reminder_log へ INSERT OR IGNORE で claim し、
  行を挿入できた呼び出しだけが通知する。プロセス再起動や万一の多重起動でも
  二重送信しない (TaskStore と同じ WAL 運用)。
- quiet hours (既定 1-8 時) の間は発火しない。quiet 明けの tick でまだ開始前
  なら通知し、開始を過ぎていたら黙って流す (過ぎた予定を今更知らせない)。
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from src.tasks.reminder import in_quiet_hours

UTC = timezone.utc

_PRUNE_AFTER_DAYS = 14


def format_event_reminder(title: str, start_local: datetime, now_local: datetime, location: str = "") -> str:
    """「まもなく予定〜」の定型文。口調の言い換えは配送側が行う。"""
    minutes = max(1, int((start_local - now_local).total_seconds() // 60))
    msg = f"まもなく予定「{title}」です。{start_local.strftime('%H:%M')} 開始 (あと{minutes}分)。"
    if location:
        msg += f"場所: {location}"
    return msg


class EventReminderEngine:
    """upcoming.json の予定を定期チェックし、開始前に一度だけ通知するエンジン。"""

    def __init__(
        self,
        *,
        callback: Callable[..., None],
        upcoming_path: str | Path = "data/calendar/upcoming.json",
        db_path: str | Path = "data/tasks/tasks.db",
        lead_min: float = 15.0,
        timezone_name: str = "Asia/Tokyo",
        quiet_hours: tuple[int, int] = (1, 8),
        check_interval: float = 60.0,
        busy_timeout_ms: int = 5000,
        now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.callback = callback
        self.upcoming_path = Path(upcoming_path)
        self.db_path = Path(db_path)
        self.lead_min = max(1.0, lead_min)
        self.timezone_name = timezone_name
        self.quiet_hours = quiet_hours
        self.check_interval = check_interval
        self.busy_timeout_ms = busy_timeout_ms
        self._now_fn = now_fn

        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone_name)

    @property
    def is_running(self) -> bool:
        return self._running

    # --- ライフサイクル ---

    def initialize(self) -> "EventReminderEngine":
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,
            timeout=self.busy_timeout_ms / 1000.0,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS event_reminder_log (
                key TEXT PRIMARY KEY,
                event_id TEXT NOT NULL DEFAULT '',
                start_at TEXT NOT NULL DEFAULT '',
                fired_at TEXT NOT NULL
            )
            """
        )
        return self

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def start(self) -> None:
        if self._running:
            return
        if self._conn is None:
            self.initialize()
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="event-reminder")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self.close()

    def _loop(self) -> None:
        while self._running:
            try:
                self.run_once()
            except Exception as e:  # エンジンは落とさない
                print(f"[EventReminder] tick error: {e}")
            if self._stop_event.wait(self.check_interval):
                break

    # --- 本体 (テストから直接呼べる) ---

    def run_once(self, now: Optional[datetime] = None) -> int:
        """1回分の評価。発火した通知数を返す。"""
        now = now or self._now_fn()
        tz = self.tz
        local_now = now.astimezone(tz)
        if in_quiet_hours(local_now, self.quiet_hours):
            return 0

        events = self._load_events()
        fired = 0
        lead = timedelta(minutes=self.lead_min)
        for ev in events:
            start_raw = str(ev.get("start") or "")
            if "T" not in start_raw:
                continue  # 終日イベントは対象外
            try:
                start_dt = datetime.fromisoformat(start_raw)
            except ValueError:
                continue
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=tz)
            if not (start_dt - lead <= now < start_dt):
                continue

            title = str(ev.get("title") or "(無題)")
            event_id = str(ev.get("event_id") or "")
            # дедуп キーは event_id を優先 (タイトル変更で再通知しない、
            # 開始時刻の変更では再通知する)。id が無いのは楽観 append 直後の
            # キャッシュだけで、その場合のみ title でフォールバックする
            # (同名・同時刻の別イベントは1通にまとまるが、文面は同一なので許容)。
            key = f"{event_id or title}:{start_raw}"
            if not self._claim(key, event_id, start_raw, now):
                continue

            message = format_event_reminder(
                title, start_dt.astimezone(tz), local_now, str(ev.get("location") or "")
            )
            try:
                self.callback(
                    trigger_type="event_remind",
                    message=message,
                    event_id=event_id,
                    title=title,
                )
                fired += 1
            except Exception as e:
                print(f"[EventReminder] callback error: {e}")

        self._prune(now)
        return fired

    def _load_events(self) -> list[dict]:
        try:
            with open(self.upcoming_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return []
        events = data.get("events", []) if isinstance(data, dict) else []
        return [e for e in events if isinstance(e, dict)]

    def _claim(self, key: str, event_id: str, start_at: str, now: datetime) -> bool:
        """INSERT OR IGNORE で claim。挿入できたときだけ True (=このプロセスが通知する)。"""
        if self._conn is None:
            return False
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO event_reminder_log (key, event_id, start_at, fired_at) "
                "VALUES (?, ?, ?, ?)",
                (key, event_id, start_at, now.astimezone(UTC).isoformat()),
            )
            return cur.rowcount == 1

    def _prune(self, now: datetime) -> None:
        if self._conn is None:
            return
        cutoff = (now.astimezone(UTC) - timedelta(days=_PRUNE_AFTER_DAYS)).isoformat()
        try:
            with self._lock:
                self._conn.execute("DELETE FROM event_reminder_log WHERE fired_at < ?", (cutoff,))
        except sqlite3.Error:
            pass
