"""透明な優先順位付けと、現在のフォーカスの永続化。

タスクの優先順位を LLM に丸投げせず、期限・明示優先度・滞留期間・
次の一手の有無を説明可能なスコアへ変換する。Discord の ``/focus`` と、
全チャット経路へ注入するコンテキストが同じ状態ファイルを読む。

状態にはタスク本文を保存しない。タスクID、選択/開始時刻、見送り回数、
日別完了数だけを atomic replace で永続化する。
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from zoneinfo import ZoneInfo

from src.tasks.formatting import format_short_due

if TYPE_CHECKING:
    from src.tasks.store import TaskStore

UTC = timezone.utc
DEFAULT_STATE_PATH = "data/tasks/priority_state.json"
DEFAULT_UPCOMING_PATH = "data/calendar/upcoming.json"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def _from_iso(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _default_state() -> dict[str, Any]:
    return {
        "version": 1,
        "active_task_id": None,
        "selected_at": None,
        "started_at": None,
        "deferred_until": {},
        "feedback": {},
        "completion_days": {},
        "counted_completion_ids": [],
        "decision_count": 0,
    }


@dataclass(frozen=True)
class RankedTask:
    task: dict[str, Any]
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FocusDecision:
    ranked: RankedTask
    current: bool
    started: bool
    focus_minutes: int
    next_event_title: Optional[str]
    next_event_at: Optional[datetime]
    completed_today: int
    streak_days: int

    @property
    def task(self) -> dict[str, Any]:
        return self.ranked.task


def rank_tasks(
    tasks: list[dict[str, Any]],
    *,
    now: Optional[datetime] = None,
    feedback: Optional[dict[str, Any]] = None,
    deferred_until: Optional[dict[str, Any]] = None,
) -> list[RankedTask]:
    """open タスクを説明可能な決定規則で順位付けする。"""
    now = now or datetime.now(UTC)
    feedback = feedback or {}
    deferred_until = deferred_until or {}
    ranked: list[RankedTask] = []

    for task in tasks:
        if task.get("status") != "open":
            continue
        task_id = str(task.get("id"))
        deferred = _from_iso(deferred_until.get(task_id))
        if deferred is not None and deferred > now:
            continue

        score = 0.0
        reasons: list[str] = []
        priority = task.get("priority", "normal")
        if priority == "high":
            score += 35
            reasons.append("高優先度")
        elif priority == "normal":
            score += 15
        else:
            reasons.append("低優先度指定")

        due = task.get("due_at")
        if isinstance(due, datetime):
            hours = (due - now).total_seconds() / 3600
            if hours < 0:
                score += 120 + min(40, abs(hours) / 6)
                reasons.append(f"期限を{_short_span(abs(hours))}超過")
            elif hours <= 2:
                score += 100
                reasons.append(f"期限まで{_short_span(hours)}")
            elif hours <= 8:
                score += 80
                reasons.append(f"期限まで{_short_span(hours)}")
            elif hours <= 24:
                score += 60
                reasons.append("今日が期限")
            elif hours <= 72:
                score += 45
                reasons.append("3日以内が期限")
            elif hours <= 24 * 7:
                score += 30
                reasons.append("1週間以内が期限")
            else:
                score += 10

        created = task.get("created_at")
        if isinstance(created, datetime):
            age_days = max(0, int((now - created).total_seconds() // 86400))
            if age_days:
                score += min(30, age_days * 2)
                if age_days >= 3:
                    reasons.append(f"{age_days}日滞留")

        if task.get("action_hint"):
            score += 8
            reasons.append("次の一手が明確")

        skips = 0
        item_feedback = feedback.get(task_id)
        if isinstance(item_feedback, dict):
            try:
                skips = max(0, int(item_feedback.get("skip_count", 0)))
            except (TypeError, ValueError):
                skips = 0
        # 見送りは好みとして反映する。ただし超過タスクは埋もれさせない。
        if skips and not (isinstance(due, datetime) and due < now):
            score -= min(30, skips * 8)
            reasons.append(f"直近{skips}回見送り")

        if not reasons:
            reasons.append("未処理タスクの中で最上位")
        ranked.append(RankedTask(task=task, score=round(score, 1), reasons=tuple(reasons)))

    ranked.sort(
        key=lambda item: (
            -item.score,
            item.task.get("due_at") or datetime.max.replace(tzinfo=UTC),
            item.task.get("id", 0),
        )
    )
    return ranked


def _short_span(hours: float) -> str:
    if hours < 1:
        return f"{max(1, int(hours * 60))}分"
    if hours < 24:
        return f"{max(1, int(hours))}時間"
    return f"{max(1, int(hours // 24))}日"


class PriorityController:
    """共有状態を使って「今やる1件」を固定し、完了/見送りを学習する。"""

    def __init__(
        self,
        store: "TaskStore",
        *,
        state_path: str | Path = DEFAULT_STATE_PATH,
        upcoming_path: str | Path = DEFAULT_UPCOMING_PATH,
        timezone_name: str = "Asia/Tokyo",
        skip_hours: float = 2.0,
        calendar_buffer_min: int = 10,
    ) -> None:
        self.store = store
        self.state_path = Path(state_path)
        self.upcoming_path = Path(upcoming_path)
        self.tz = ZoneInfo(timezone_name)
        self.skip_hours = max(0.25, float(skip_hours))
        self.calendar_buffer_min = max(0, int(calendar_buffer_min))
        self._lock = threading.RLock()
        self.last_error: Optional[str] = None
        self._state = self._read_state()

    def _read_state(self) -> dict[str, Any]:
        state = _default_state()
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for key in state:
                    if key in raw:
                        state[key] = raw[key]
        except FileNotFoundError:
            pass
        except (OSError, json.JSONDecodeError) as exc:
            self.last_error = str(exc)
        return state

    def _refresh(self) -> None:
        # Web/音声/Discord の別プロセスが同じファイルを見るため毎回再読込する。
        self._state = self._read_state()

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_name(f".{self.state_path.name}.tmp")
        tmp.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, self.state_path)

    def _open_tasks(self) -> list[dict[str, Any]]:
        return self.store.list("open", 1000)

    def _active_task(self, tasks: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        active_id = self._state.get("active_task_id")
        return next((t for t in tasks if t.get("id") == active_id), None)

    def _record_completion(self, task_id: int, now: datetime) -> None:
        counted = [int(x) for x in self._state.get("counted_completion_ids", []) if str(x).isdigit()]
        if task_id in counted:
            return
        day = now.astimezone(self.tz).date().isoformat()
        days = self._state.setdefault("completion_days", {})
        days[day] = int(days.get(day, 0)) + 1
        counted.append(task_id)
        self._state["counted_completion_ids"] = counted[-500:]

    def _reconcile(self, now: datetime, tasks: list[dict[str, Any]]) -> None:
        active_id = self._state.get("active_task_id")
        if not isinstance(active_id, int) or self._active_task(tasks) is not None:
            return
        previous = self.store.get(active_id)
        if previous and previous.get("status") == "done" and self._state.get("started_at"):
            self._record_completion(active_id, now)
        self._clear_active()
        self._save()

    def _clear_active(self) -> None:
        self._state["active_task_id"] = None
        self._state["selected_at"] = None
        self._state["started_at"] = None

    def _metrics(self, now: datetime) -> tuple[int, int]:
        days = self._state.get("completion_days", {})
        if not isinstance(days, dict):
            return (0, 0)
        today = now.astimezone(self.tz).date()
        completed_today = int(days.get(today.isoformat(), 0))
        cursor = today if completed_today else today - timedelta(days=1)
        streak = 0
        while int(days.get(cursor.isoformat(), 0)) > 0:
            streak += 1
            cursor -= timedelta(days=1)
        return completed_today, streak

    def _next_event(self, now: datetime) -> tuple[Optional[str], Optional[datetime]]:
        try:
            raw = json.loads(self.upcoming_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return (None, None)
        events = raw.get("events", []) if isinstance(raw, dict) else []
        candidates: list[tuple[datetime, str]] = []
        for event in events:
            if not isinstance(event, dict) or "subpc-task:" in str(event.get("description", "")):
                continue
            start_raw = str(event.get("start") or "")
            if "T" not in start_raw:
                continue
            start = _from_iso(start_raw)
            if start is not None and start > now:
                candidates.append((start, str(event.get("title") or "(無題)")))
        if not candidates:
            return (None, None)
        start, title = min(candidates, key=lambda item: item[0])
        return (title, start)

    def _decision(self, ranked: RankedTask, *, current: bool, now: datetime) -> FocusDecision:
        title, event_at = self._next_event(now)
        focus_minutes = 25
        if event_at is not None:
            available = int((event_at - now).total_seconds() // 60) - self.calendar_buffer_min
            focus_minutes = max(5, min(25, available))
        completed_today, streak = self._metrics(now)
        return FocusDecision(
            ranked=ranked,
            current=current,
            started=bool(self._state.get("started_at")),
            focus_minutes=focus_minutes,
            next_event_title=title,
            next_event_at=event_at,
            completed_today=completed_today,
            streak_days=streak,
        )

    def recommend(self, *, now: Optional[datetime] = None, select: bool = True) -> Optional[FocusDecision]:
        now = now or datetime.now(UTC)
        with self._lock:
            self._refresh()
            tasks = self._open_tasks()
            # 読み取り専用のチャットコンテキストは共有状態を書き換えない。
            # Discord の選定操作だけが stale active の掃除と保存を担当する。
            if select:
                self._reconcile(now, tasks)
            active = self._active_task(tasks)
            ranked = rank_tasks(
                tasks,
                now=now,
                feedback=self._state.get("feedback"),
                deferred_until=self._state.get("deferred_until"),
            )
            if active is not None:
                active_ranked = next((r for r in ranked if r.task["id"] == active["id"]), None)
                if active_ranked is None:
                    # active は明示的に選ばれたものなので defer 状態より優先する。
                    active_ranked = rank_tasks([active], now=now)[0]
                return self._decision(active_ranked, current=True, now=now)
            if not ranked:
                return None
            chosen = ranked[0]
            if select:
                self._state["active_task_id"] = int(chosen.task["id"])
                self._state["selected_at"] = _iso(now)
                self._state["started_at"] = None
                self._state["decision_count"] = int(self._state.get("decision_count", 0)) + 1
                self._save()
            return self._decision(chosen, current=False, now=now)

    def start(self, *, now: Optional[datetime] = None) -> Optional[FocusDecision]:
        now = now or datetime.now(UTC)
        with self._lock:
            decision = self.recommend(now=now, select=True)
            if decision is None:
                return None
            if not self._state.get("started_at"):
                self._state["started_at"] = _iso(now)
                self._save()
            return self._decision(decision.ranked, current=True, now=now)

    def pick(self, task_id: int, *, now: Optional[datetime] = None) -> Optional[FocusDecision]:
        now = now or datetime.now(UTC)
        with self._lock:
            self._refresh()
            task = self.store.get(task_id)
            if task is None or task.get("status") != "open":
                return None
            self._state["active_task_id"] = task_id
            self._state["selected_at"] = _iso(now)
            self._state["started_at"] = None
            self._state.setdefault("deferred_until", {}).pop(str(task_id), None)
            self._state["decision_count"] = int(self._state.get("decision_count", 0)) + 1
            self._save()
            ranked = rank_tasks([task], now=now)[0]
            return self._decision(ranked, current=False, now=now)

    def next(self, *, now: Optional[datetime] = None) -> Optional[FocusDecision]:
        now = now or datetime.now(UTC)
        with self._lock:
            self._refresh()
            active_id = self._state.get("active_task_id")
            if isinstance(active_id, int):
                self._state.setdefault("deferred_until", {})[str(active_id)] = _iso(
                    now + timedelta(hours=self.skip_hours)
                )
                feedback = self._state.setdefault("feedback", {}).setdefault(str(active_id), {})
                feedback["skip_count"] = int(feedback.get("skip_count", 0)) + 1
                feedback["last_skipped_at"] = _iso(now)
            self._clear_active()
            self._save()
            return self.recommend(now=now, select=True)

    def complete(self, *, now: Optional[datetime] = None) -> tuple[bool, Optional[FocusDecision]]:
        now = now or datetime.now(UTC)
        with self._lock:
            self._refresh()
            active_id = self._state.get("active_task_id")
            if not isinstance(active_id, int):
                return (False, self.recommend(now=now, select=True))
            if not self.store.done(active_id, now=now):
                self._clear_active()
                self._save()
                return (False, self.recommend(now=now, select=True))
            self._record_completion(active_id, now)
            self._clear_active()
            self._save()
            return (True, self.recommend(now=now, select=True))

    def status(self, *, now: Optional[datetime] = None) -> dict[str, Any]:
        now = now or datetime.now(UTC)
        with self._lock:
            decision = self.recommend(now=now, select=False)
            completed_today, streak = self._metrics(now)
            return {
                "active_task_id": decision.task["id"] if decision and decision.current else None,
                "started": bool(decision and decision.started),
                "completed_today": completed_today,
                "streak_days": streak,
                "decision_count": int(self._state.get("decision_count", 0)),
                "last_error": self.last_error,
            }


def format_focus_decision(decision: Optional[FocusDecision], tz: ZoneInfo) -> str:
    if decision is None:
        return "🎯 未完了タスクはありません。新しい依頼は『タスク: …』でここへ集められます。"
    task = decision.task
    state = "実行中" if decision.started else ("固定中" if decision.current else "選定")
    lines = [
        f"🎯 今やる: `#{task['id']}` {task['title']}（{state}）",
        f"理由: {' / '.join(decision.ranked.reasons[:3])}",
    ]
    hint = str(task.get("action_hint") or "").strip()
    lines.append(f"最初の{decision.focus_minutes}分: {hint or '完了条件を小さく決めて着手'}")
    if decision.next_event_at is not None:
        event_time = format_short_due(decision.next_event_at.astimezone(tz), with_time=True)
        lines.append(f"次の予定: {event_time} {decision.next_event_title}")
    momentum = f"今日{decision.completed_today}件完了"
    if decision.streak_days:
        momentum += f"・{decision.streak_days}日連続"
    lines.append(f"勢い: {momentum}")
    lines.append("操作: `/focus start` `/focus done` `/focus next` `/focus pick`")
    return "\n".join(lines)


def build_priority_context(
    store: "TaskStore",
    *,
    now: Optional[datetime] = None,
    state_path: Optional[str | Path] = None,
    upcoming_path: Optional[str | Path] = None,
) -> str:
    """全チャット経路向けの読み取り専用コンテキスト。選定状態は変更しない。"""
    resolved_state = Path(
        state_path or os.environ.get("PRIORITY_STATE_PATH", DEFAULT_STATE_PATH)
    )
    resolved_upcoming = Path(
        upcoming_path or os.environ.get("PRIORITY_UPCOMING_PATH", DEFAULT_UPCOMING_PATH)
    )
    if not resolved_state.is_absolute():
        resolved_state = PROJECT_ROOT / resolved_state
    if not resolved_upcoming.is_absolute():
        resolved_upcoming = PROJECT_ROOT / resolved_upcoming
    controller = PriorityController(
        store,
        state_path=resolved_state,
        upcoming_path=resolved_upcoming,
        timezone_name=getattr(store, "timezone_name", "Asia/Tokyo"),
    )
    decision = controller.recommend(now=now, select=False)
    if decision is None:
        return ""
    task = decision.task
    mode = "現在固定中" if decision.current else "次の推奨"
    return (
        "\n--- 優先順位オーケストレーター ---\n"
        f"{mode}: #{task['id']} {task['title']}\n"
        f"根拠: {' / '.join(decision.ranked.reasons[:3])}\n"
        "ユーザーが優先順位を尋ねたら、この1件を最初に提案し、理由を短く説明する。"
        "拒否・見送り・手動指定は尊重する。"
    )
