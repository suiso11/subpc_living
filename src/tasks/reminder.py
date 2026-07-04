"""
タスクのリマインドエンジン

ProactiveEngine とは独立した threading ベースの定期チェック。
60秒間隔で TaskStore から通知対象を claim し、エスカレーション段階を
計算して callback(trigger_type="task_remind", message, task_id, stage) を呼ぶ。

エスカレーション:
    期限24h前 (1回) → 3h前 (1回) → 1h前 (30分毎) → 超過 (2時間毎)

ステートは task_notifications に永続化されるため、エンジンを作り直しても
(=プロセス再起動でも) 二重送信しない。

quiet hours (env TASKS_QUIET_HOURS, default "1-8", ローカル時) の間は
「超過」以外の通知を送らず、quiet 明けに繰り越す。

メッセージは「タスク名 + 残り時間 + action_hint(あれば)」の定型文。
ペルソナ口調への言い換えは配送側 (bot / proactive_bridge の rewrite) が行う。
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from src.tasks.store import TaskStore, format_local_due

UTC = timezone.utc

HOUR = 3600
# 各段階の再通知間隔 (秒)。単発段階は次段階の境界まで飛ばす。
STAGE_1H_INTERVAL = 30 * 60      # 1h前は30分毎
STAGE_OVERDUE_INTERVAL = 2 * HOUR  # 超過は2時間毎

# 段階の進行順 (単発段階のスキップ判定に使う)
_STAGE_ORDER = {"pending": 0, "24h": 1, "3h": 2, "1h": 3, "overdue": 4}


def parse_quiet_hours(value: Optional[str], default: tuple[int, int] = (1, 8)) -> tuple[int, int]:
    """"1-8" → (1, 8)。パース失敗時は default。start==end は quiet 無効扱い(=(0,0))。"""
    if not value or not value.strip():
        return default
    try:
        start_s, end_s = value.split("-", 1)
        start = int(start_s.strip()) % 24
        end = int(end_s.strip()) % 24
        return (start, end)
    except (ValueError, AttributeError):
        return default


def in_quiet_hours(local_dt: datetime, quiet: tuple[int, int]) -> bool:
    start, end = quiet
    if start == end:
        return False
    hour = local_dt.hour
    if start < end:
        return start <= hour < end
    # 折り返し (例: 22-6)
    return hour >= start or hour < end


def quiet_end_after(local_dt: datetime, quiet: tuple[int, int]) -> datetime:
    """local_dt が quiet 中のとき、直近の quiet 終了時刻 (ローカル) を返す。"""
    start, end = quiet
    end_today = local_dt.replace(hour=end % 24, minute=0, second=0, microsecond=0)
    if end == 0:
        end_today = end_today + timedelta(days=1)
    if end_today <= local_dt:
        end_today = end_today + timedelta(days=1)
    return end_today


def compute_action(
    due_at: datetime,
    now: datetime,
    last_stage: Optional[str],
    repeat_count: int,
) -> tuple[bool, str, datetime, int]:
    """次にとるべき通知アクションを計算する。

    Returns:
        (should_fire, stage, next_notify_at, new_repeat_count)
    """
    last_rank = _STAGE_ORDER.get(last_stage or "pending", 0)
    remaining = (due_at - now).total_seconds()

    if remaining > 24 * HOUR:
        # まだどの窓にも入っていない。24h前まで待つ。
        return (False, last_stage or "pending", due_at - timedelta(hours=24), repeat_count)

    if remaining > 3 * HOUR:
        stage = "24h"
        if last_rank < _STAGE_ORDER["24h"]:
            # 24h前を一度だけ通知。次は 3h前の境界。
            return (True, stage, due_at - timedelta(hours=3), 1)
        return (False, last_stage or stage, due_at - timedelta(hours=3), repeat_count)

    if remaining > 1 * HOUR:
        stage = "3h"
        if last_rank < _STAGE_ORDER["3h"]:
            return (True, stage, due_at - timedelta(hours=1), 1)
        return (False, last_stage or stage, due_at - timedelta(hours=1), repeat_count)

    if remaining > 0:
        stage = "1h"
        # 30分毎。この段階に入るたび / next_notify_at 到達で発火。
        new_count = repeat_count + 1 if last_stage == "1h" else 1
        return (True, stage, now + timedelta(seconds=STAGE_1H_INTERVAL), new_count)

    # 超過。2時間毎。
    stage = "overdue"
    new_count = repeat_count + 1 if last_stage == "overdue" else 1
    return (True, stage, now + timedelta(seconds=STAGE_OVERDUE_INTERVAL), new_count)


def _humanize_remaining(due_at: datetime, now: datetime) -> str:
    delta = due_at - now
    secs = abs(delta.total_seconds())
    days = int(secs // 86400)
    hours = int((secs % 86400) // 3600)
    minutes = int((secs % 3600) // 60)
    if days >= 1:
        base = f"{days}日" + (f"{hours}時間" if hours else "")
    elif hours >= 1:
        base = f"{hours}時間" + (f"{minutes}分" if minutes else "")
    else:
        base = f"{max(minutes, 1)}分"
    if delta.total_seconds() < 0:
        return f"期限を{base}過ぎています"
    return f"あと{base}"


def format_reminder_message(task: dict, now: datetime, stage: str, tz: ZoneInfo) -> str:
    """「タスク名 + 残り時間 + action_hint」の定型文。"""
    title = task["title"]
    due_at = task["due_at"]
    remaining = _humanize_remaining(due_at, now)
    due_str = format_local_due(due_at, task.get("due_granularity"), tz, now)
    msg = f"タスク「{title}」の期限は {due_str} です。{remaining}。"
    hint = task.get("action_hint")
    if hint:
        msg += f"次の一手: {hint}"
    return msg


class TaskReminderEngine:
    """タスクのリマインドを定期チェックするエンジン。"""

    def __init__(
        self,
        store: TaskStore,
        callback: Callable[..., None],
        *,
        owner: str = "reminder",
        timezone_name: str = "Asia/Tokyo",
        quiet_hours: tuple[int, int] = (1, 8),
        check_interval: float = 60.0,
        now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.store = store
        self.callback = callback
        self.owner = owner
        self.timezone_name = timezone_name
        self.quiet_hours = quiet_hours
        self.check_interval = check_interval
        self._now_fn = now_fn

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone_name)

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _loop(self) -> None:
        # 起動直後に一度実行してから間隔待ち。
        while self._running:
            try:
                self.run_once()
            except Exception as e:  # エンジンは落とさない
                print(f"[TaskReminder] tick error: {e}")
            if self._stop_event.wait(self.check_interval):
                break

    def run_once(self, now: Optional[datetime] = None) -> int:
        """1回分の評価。発火した通知数を返す (テスト用に公開)。"""
        now = now or self._now_fn()
        tz = self.tz
        local_now = now.astimezone(tz)
        is_quiet = in_quiet_hours(local_now, self.quiet_hours)

        claimed = self.store.claim_due_notifications(self.owner, now)
        fired = 0
        for task in claimed:
            notif = task["notification"]
            due_at = task["due_at"]
            if due_at is None:
                self.store.release_lease(task["id"], self.owner)
                continue

            should_fire, stage, next_at, new_count = compute_action(
                due_at, now, notif["last_stage"], notif["repeat_count"]
            )

            if not should_fire:
                self.store.record_notification(
                    task["id"], self.owner,
                    stage=stage, next_notify_at=next_at,
                    repeat_count=new_count, fired=False, now=now,
                )
                continue

            # quiet hours: 超過以外は繰り越す (quiet 明けに再評価)。
            if is_quiet and stage != "overdue":
                carry_local = quiet_end_after(local_now, self.quiet_hours)
                carry_utc = carry_local.astimezone(UTC)
                self.store.record_notification(
                    task["id"], self.owner,
                    stage=notif["last_stage"],  # 段階は進めない
                    next_notify_at=carry_utc,
                    repeat_count=notif["repeat_count"],
                    fired=False, now=now,
                )
                continue

            message = format_reminder_message(task, now, stage, tz)
            try:
                self.callback(
                    trigger_type="task_remind",
                    message=message,
                    task_id=task["id"],
                    stage=stage,
                )
                fired += 1
            except Exception as e:
                print(f"[TaskReminder] callback error: {e}")

            self.store.record_notification(
                task["id"], self.owner,
                stage=stage, next_notify_at=next_at,
                repeat_count=new_count, fired=True, now=now,
            )
        return fired
