"""決定的 Proactive Policy エンジン。

LLM に判断させず、CompanionState + 現在時刻 + 予定接近情報を入力とし、
介入可否・内容・種別を決定的に返す。提案と実行を分け、変更操作は承認必須とする
(現状このエンジンは変更操作を行わないため requires_approval は常に False)。

message_hint は発話内容のヒントであり、予定タイトル等の個人情報・生データ・
エラー本文を含まない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

from src.companion.contracts import CompanionState

ActionKind = Literal["schedule_remind", "break_suggest", "away_return", "silent"]
ProactiveAction = Literal["schedule_remind", "break_suggest", "away_return"]

_MESSAGE_HINTS: dict[ProactiveAction, str] = {
    "schedule_remind": "予定が近づいています",
    "break_suggest": "長時間の作業、休憩を提案",
    "away_return": "離席からの復帰の声かけ",
}

DEFAULT_COOLDOWN_SECONDS: dict[str, float] = {
    "schedule_remind": 900.0,
    "break_suggest": 1800.0,
    "away_return": 300.0,
}


@dataclass(frozen=True)
class PolicyDecision:
    """1回の判定結果。

    should_act が False のとき action_kind は "silent"。
    """

    should_act: bool
    action_kind: ActionKind | None
    message_hint: str
    requires_approval: bool
    cooldown_key: str
    reason: str


@dataclass(frozen=True)
class PolicyContext:
    """decide の入力。時刻は注入値 (now) と state.updated_at のみを使う。"""

    state: CompanionState
    now: float
    next_event_at: float | None = None
    next_event_title: str | None = None
    last_fired: dict[str, float] = field(default_factory=dict)


class DeterministicProactivePolicy:
    """CompanionState と時刻から介入可否を決定的に決める Policy。

    - focused 中は黙る (予定が迫っていて interruptible なら schedule_remind 可)。
    - away 復帰直後は away_return。
    - 長時間作業 (focused_since から long_work_seconds 以上) かつ interruptible
      なら break_suggest (提案のみ)。
    - 予定接近 (0 <= next_event_at - now <= schedule_lead_seconds) かつ
      interruptible なら schedule_remind。
    - 同一 cooldown_key の再通知は cooldown_seconds 内で抑止 (reason="cooldown")。
    - 変更操作は行わないため requires_approval は常に False。
    """

    def __init__(
        self,
        long_work_seconds: float = 2 * 60 * 60,
        schedule_lead_seconds: float = 10 * 60,
        away_return_seconds: float = 60,
        cooldown_seconds: Mapping[str, float] | None = None,
    ) -> None:
        self.long_work_seconds = long_work_seconds
        self.schedule_lead_seconds = schedule_lead_seconds
        self.away_return_seconds = away_return_seconds
        self.cooldown_seconds = dict(DEFAULT_COOLDOWN_SECONDS)
        if cooldown_seconds:
            self.cooldown_seconds.update(cooldown_seconds)

    def decide(self, ctx: PolicyContext) -> PolicyDecision:
        state = ctx.state

        if state.activity_mode == "focused":
            if self._schedule_approaching(ctx) and state.interruptible:
                return self._action(ctx, "schedule_remind", "schedule_approaching")
            return self._silent(reason="focused", cooldown_key="focused")

        if self._is_away_return(ctx):
            return self._action(ctx, "away_return", "away_return")

        if self._is_long_work(ctx) and state.interruptible:
            return self._action(ctx, "break_suggest", "long_work")

        if self._schedule_approaching(ctx) and state.interruptible:
            return self._action(ctx, "schedule_remind", "schedule_approaching")

        return self._silent(reason="silent", cooldown_key="silent")

    def _action(
        self, ctx: PolicyContext, kind: ProactiveAction, reason: str
    ) -> PolicyDecision:
        if self._on_cooldown(ctx, kind):
            return self._silent(reason="cooldown", cooldown_key=kind)
        return PolicyDecision(
            should_act=True,
            action_kind=kind,
            message_hint=_MESSAGE_HINTS[kind],
            requires_approval=False,
            cooldown_key=kind,
            reason=reason,
        )

    def _silent(self, reason: str, cooldown_key: str) -> PolicyDecision:
        return PolicyDecision(
            should_act=False,
            action_kind="silent",
            message_hint="",
            requires_approval=False,
            cooldown_key=cooldown_key,
            reason=reason,
        )

    def _on_cooldown(self, ctx: PolicyContext, key: str) -> bool:
        last = ctx.last_fired.get(key)
        if last is None:
            return False
        cooldown = self.cooldown_seconds.get(key, 0.0)
        return ctx.now - last < cooldown

    def _schedule_approaching(self, ctx: PolicyContext) -> bool:
        if ctx.next_event_at is None:
            return False
        delta = ctx.next_event_at - ctx.now
        return 0.0 <= delta <= self.schedule_lead_seconds

    def _is_long_work(self, ctx: PolicyContext) -> bool:
        focused_since = ctx.state.focused_since
        return (
            focused_since is not None
            and ctx.now - focused_since >= self.long_work_seconds
        )

    def _is_away_return(self, ctx: PolicyContext) -> bool:
        state = ctx.state
        if not state.present or state.activity_mode == "away":
            return False
        return 0.0 <= ctx.now - state.updated_at <= self.away_return_seconds
