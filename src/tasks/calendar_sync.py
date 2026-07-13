"""
タスク ⇔ Google Calendar 双方向同期。

3 つの部品からなる:

- TaskCalendarSync : タスク → カレンダー (書き込み)。
  TaskStore.on_change から (task_id, event) を enqueue され、バックグラウンド
  スレッドで npx MCP 呼び出し (数秒) を実行する。Discord 応答をブロックしない。
- CalendarPullWorker : カレンダー → bot (読み取り)。定期的に向こう N 日の予定を
  取得し、data/calendar/upcoming.json に書き出し、UserProfile.schedule の
  gcal: 印付きエントリを洗い替えする。
- CalendarContext : upcoming.json を読んで LLM プロンプト用のブロックを生成
  (ファイル読取のみ。音声パイプライン側でも安全に使える)。

設計上の重要ポイント:
- カレンダー障害でタスク操作を絶対に失敗させない (enqueue は非ブロッキング、
  ワーカー内の例外は握り潰してリトライ)。
- 自作イベントには description に `subpc-task:{task_id}` マーカーを入れ、
  pull 時に除外して再輸入ループを防ぐ。
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time as _time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

TASK_MARKER_PREFIX = "subpc-task:"


def _marker_for(task_id: int) -> str:
    return f"{TASK_MARKER_PREFIX}{task_id}"


def _log(message: str) -> None:
    print(f"[CalendarSync] {message}")


# =====================================================================
# タスク → カレンダー
# =====================================================================


class TaskCalendarSync:
    """タスク変更をバックグラウンドで Google Calendar に反映するワーカー。"""

    def __init__(
        self,
        *,
        store: Any,
        calendar_client: Any,
        calendar_id: str = "primary",
        enabled: bool = False,
        timezone: str = "Asia/Tokyo",
        max_retries: int = 3,
        retry_base_delay: float = 5.0,
        max_queue: int = 1000,
    ):
        self.store = store
        self.client = calendar_client
        self.calendar_id = calendar_id or "primary"
        self.enabled = enabled
        self.timezone = timezone
        self.max_retries = max(1, max_retries)
        self.retry_base_delay = retry_base_delay
        self._queue: "queue.Queue[Optional[tuple[int, str]]]" = queue.Queue(maxsize=max_queue)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # --- ライフサイクル ---

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, name="task-calendar-sync", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)  # sentinel で早期起床
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    # --- enqueue (TaskStore.on_change のコールバック) ---

    def enqueue(self, task_id: int, event: str) -> None:
        """(task_id, event) を投入する。無効時・満杯時は静かに捨てる。"""
        if not self.enabled:
            return
        try:
            self._queue.put_nowait((int(task_id), str(event)))
        except queue.Full:
            _log(f"queue full, dropping task={task_id} event={event}")

    # --- ワーカー ---

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                self._queue.task_done()
                break
            task_id, event = item
            try:
                self._process_with_retry(task_id, event)
            except Exception as exc:  # 最終防衛線
                _log(f"unexpected error task={task_id} event={event}: {exc}")
            finally:
                self._queue.task_done()

    def _process_with_retry(self, task_id: int, event: str) -> bool:
        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                ok, err = self._sync_task(task_id, event)
            except Exception as exc:
                ok, err = False, str(exc)
            if ok:
                return True
            last_error = err
            if attempt < self.max_retries:
                delay = self.retry_base_delay * attempt
                if self._stop.wait(delay):
                    break
        _log(f"failed after {self.max_retries} attempts task={task_id} event={event}: {last_error}")
        return False

    # --- 1 件分の同期。(ok, error) を返す。 ---

    def _sync_task(self, task_id: int, event: str) -> tuple[bool, str]:
        task = self.store.get(task_id)
        if task is None:
            return True, ""  # 既に消えている: 何もしない

        event_id = task.get("calendar_event_id")
        title = task.get("title") or ""
        due = task.get("due_at")
        gran = task.get("due_granularity")
        note = task.get("note") or ""

        if event == "drop":
            return self._delete_if_present(task_id, event_id)

        if event == "done":
            if not event_id:
                return True, ""  # カレンダーに載っていないタスクは何もしない
            res = self.client.update_event(
                event_id,
                calendar_id=self.calendar_id,
                summary=f"✅ {title}",
                timezone=self.timezone,
            )
            if res.ok:
                self.store.set_calendar_event(task_id, event_id)
                return True, ""
            return False, res.error

        # add / update
        if due is None:
            # 期限が無い → カレンダー対象外。既存イベントがあれば削除。
            return self._delete_if_present(task_id, event_id)

        start_str, end_str = self._event_window(due, gran)
        summary = f"📋 {title}"
        description = _marker_for(task_id)
        if note:
            description = f"{description}\n{note}"

        if event_id:
            res = self.client.update_event(
                event_id,
                calendar_id=self.calendar_id,
                summary=summary,
                start=start_str,
                end=end_str,
                description=description,
                timezone=self.timezone,
            )
            if res.ok:
                self.store.set_calendar_event(task_id, event_id)
                return True, ""
            return False, res.error

        res = self.client.create_event(
            calendar_id=self.calendar_id,
            summary=summary,
            start=start_str,
            end=end_str,
            description=description,
            timezone=self.timezone,
        )
        if not res.ok:
            return False, res.error
        if res.event_id:
            self.store.set_calendar_event(task_id, res.event_id)
        else:
            # イベントは作成できたが id を取れなかった (レスポンス形式差)。
            # リトライすると二重作成になるため成功扱いにし、pull 側のマーカー
            # 照合 (CalendarPullWorker) で後から対応付けを回復させる。
            _log(f"created event without parseable id (task={task_id}); will backfill via pull")
        return True, ""

    def _delete_if_present(self, task_id: int, event_id: Optional[str]) -> tuple[bool, str]:
        if not event_id:
            return True, ""
        res = self.client.delete_event(event_id, calendar_id=self.calendar_id)
        if res.ok:
            self.store.clear_calendar_event(task_id)
            return True, ""
        return False, res.error

    def _event_window(self, due: datetime, granularity: Optional[str]) -> tuple[str, str]:
        """(start, end) の ISO 文字列を返す。"""
        tz = ZoneInfo(self.timezone)
        local = due.astimezone(tz)
        if granularity == "date":
            d = local.date()
            return d.isoformat(), (d + timedelta(days=1)).isoformat()
        # datetime: 期限の 30 分前 〜 期限
        start_local = local - timedelta(minutes=30)
        return (
            start_local.replace(tzinfo=None).isoformat(timespec="seconds"),
            local.replace(tzinfo=None).isoformat(timespec="seconds"),
        )


# =====================================================================
# カレンダー → bot
# =====================================================================


class CalendarPullWorker:
    """向こう N 日の予定を定期取得し、upcoming.json と profile.schedule に反映する。"""

    def __init__(
        self,
        *,
        calendar_client: Any,
        calendar_id: str = "primary",
        profile: Any = None,
        store: Any = None,
        timezone: str = "Asia/Tokyo",
        interval_min: float = 20.0,
        days_ahead: int = 7,
        upcoming_path: str | Path = "data/calendar/upcoming.json",
    ):
        self.client = calendar_client
        self.calendar_id = calendar_id or "primary"
        self.profile = profile
        self.store = store
        self.timezone = timezone
        self.interval_sec = max(60.0, interval_min * 60.0)
        self.days_ahead = max(1, days_ahead)
        self.upcoming_path = Path(upcoming_path)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # --- ライフサイクル ---

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="calendar-pull", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _loop(self) -> None:
        # 起動直後に一度実行してから定期化する。
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception as exc:
                _log(f"pull error: {exc}")
            if self._stop.wait(self.interval_sec):
                break

    # --- 本体 (テストから直接呼べる) ---

    def run_once(self, *, now: Optional[datetime] = None) -> bool:
        tz = ZoneInfo(self.timezone)
        now = now or datetime.now(tz)
        today = now.astimezone(tz).date()
        end_date = today + timedelta(days=self.days_ahead - 1)

        result = self.client.list_events_range(
            today, end_date, timezone=self.timezone, calendar_id=self.calendar_id
        )
        if not result.ok:
            _log(f"list_events_range failed: {result.error}")
            return False

        all_events = list(result.events)
        # 自作イベント (subpc-task: マーカー) は取り込まない (再輸入ループ防止)。
        external = [e for e in all_events if TASK_MARKER_PREFIX not in (e.description or "")]

        self._reconcile_markers(all_events)
        self._write_upcoming(external, now)
        if self.profile is not None:
            self._sync_profile_schedule(external, today, tz)
        return True

    def _reconcile_markers(self, events: list[Any]) -> None:
        """subpc-task: マーカー付きイベントの id を、未対応付けのタスクへ補完する。"""
        if self.store is None:
            return
        for ev in events:
            desc = ev.description or ""
            idx = desc.find(TASK_MARKER_PREFIX)
            if idx < 0 or not ev.event_id:
                continue
            tail = desc[idx + len(TASK_MARKER_PREFIX):]
            digits = ""
            for ch in tail:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if not digits:
                continue
            try:
                task = self.store.get(int(digits))
            except Exception:
                task = None
            if task is None:
                continue
            if not task.get("calendar_event_id") and task.get("status") == "open":
                try:
                    self.store.set_calendar_event(int(digits), ev.event_id)
                    _log(f"backfilled calendar_event_id for task={digits}")
                except Exception:
                    pass

    def _write_upcoming(self, events: list[Any], now: datetime) -> None:
        payload = {
            "generated_at": now.isoformat(),
            "timezone": self.timezone,
            "events": [
                {
                    "title": e.title,
                    "start": e.start,
                    "end": e.end,
                    "location": e.location,
                    "event_id": e.event_id,
                }
                for e in sorted(events, key=lambda x: x.sort_key)
            ],
        }
        self.upcoming_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.upcoming_path.with_suffix(self.upcoming_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.upcoming_path)

    def _sync_profile_schedule(self, events: list[Any], today: date, tz: ZoneInfo) -> None:
        """当日+翌日の予定を profile.schedule の gcal: 印エントリとして洗い替えする。"""
        tomorrow = today + timedelta(days=1)
        target = {today.isoformat(), tomorrow.isoformat()}

        try:
            # 他プロセス/スレッドの変更を取り込んでから gcal: 分だけ差し替える。
            self.profile.load()
        except Exception:
            pass

        manual = [
            s for s in self.profile.schedule
            if not str(s.get("note", "")).startswith("gcal:")
        ]
        gcal_entries: list[dict] = []
        added_at = datetime.now().isoformat()
        for e in events:
            date_str, time_str = self._local_date_time(e.start, tz)
            if date_str not in target:
                continue
            gcal_entries.append(
                {
                    "date": date_str,
                    "time": time_str,
                    "title": e.title,
                    "note": f"gcal:{e.event_id}",
                    "added_at": added_at,
                }
            )
        self.profile.data["schedule"] = manual + gcal_entries
        self.profile.save()

    @staticmethod
    def _local_date_time(start: str, tz: ZoneInfo) -> tuple[str, str]:
        """イベント start (ISO) からローカルの (YYYY-MM-DD, HH:MM) を得る。終日は time=''。"""
        if not start:
            return "", ""
        if "T" not in start:
            return start[:10], ""  # 終日イベント (date のみ)
        try:
            dt = datetime.fromisoformat(start)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=tz)
            local = dt.astimezone(tz)
            return local.date().isoformat(), local.strftime("%H:%M")
        except Exception:
            return start[:10], ""


# =====================================================================
# LLM コンテキスト (ファイル読取のみ)
# =====================================================================


class CalendarContext:
    """upcoming.json を読み、今日〜明日の予定ブロックを生成する。"""

    def __init__(
        self,
        upcoming_path: str | Path = "data/calendar/upcoming.json",
        *,
        timezone: str = "Asia/Tokyo",
        max_items: int = 6,
        stale_after_hours: float = 24.0,
    ):
        self.upcoming_path = Path(upcoming_path)
        self.timezone = timezone
        self.max_items = max_items
        self.stale_after_hours = stale_after_hours

    def get_context_text(self, *, now: Optional[datetime] = None) -> str:
        tz = ZoneInfo(self.timezone)
        now = now or datetime.now(tz)

        try:
            with open(self.upcoming_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return ""

        # 鮮度: 24 時間を超えて古いファイルは無視する (予定情報が完全に陳腐)。
        generated_at = data.get("generated_at", "")
        try:
            gen_dt = datetime.fromisoformat(generated_at)
            if gen_dt.tzinfo is None:
                gen_dt = gen_dt.replace(tzinfo=tz)
            age_hours = (now - gen_dt).total_seconds() / 3600.0
            if age_hours > self.stale_after_hours:
                return ""
        except (ValueError, TypeError):
            return ""

        today = now.astimezone(tz).date()
        tomorrow = today + timedelta(days=1)
        target = {today.isoformat(), tomorrow.isoformat()}

        rows: list[tuple[str, str]] = []  # (sort_key, line)
        for e in data.get("events", []):
            if not isinstance(e, dict):
                continue
            start = str(e.get("start") or "")
            date_str, time_str = CalendarPullWorker._local_date_time(start, tz)
            if date_str not in target:
                continue
            title = str(e.get("title") or "(無題)")
            day_label = "今日" if date_str == today.isoformat() else "明日"
            when = f"{day_label} {time_str}" if time_str else f"{day_label} 終日"
            rows.append((start or date_str, f"- {when} {title}"))

        if not rows:
            return ""
        rows.sort(key=lambda r: r[0])
        lines = ["\n--- 予定 (Google Calendar) ---"]
        lines.extend(line for _, line in rows[: self.max_items])
        return "\n".join(lines)
