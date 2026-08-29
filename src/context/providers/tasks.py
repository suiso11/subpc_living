"""Tasks Context Provider。

src.tasks.store.build_task_context() を呼び出し、その結果を
source=TASKS_SOURCE / sensitivity=personal / local_only=True の str ContextBlock として返す。
tasks 側の store / prioritizer / DB / editor 実装は変更・呼び出しせず、TaskStore も変更しない。
build_task_context は 0 件でも非空の権威ブロックを返すため、0 件 store でも必ず Block 化する。
空・空白のみの結果、非 str の結果、または例外時は None を返す (例外は会話を止めない)。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Protocol, runtime_checkable

from src.context.contracts import ContextBlock, TASKS_SOURCE

logger = logging.getLogger(__name__)


@runtime_checkable
class TasksSource(Protocol):
    """TaskStore 互換の最小契約。

    build_task_context が読む get_context_tasks / tz / timezone_name を提供する任意の
    source へ構造的適合 (duck-typing) する。tasks 側の TaskStore へ runtime import して
    循環を作らないため、concrete 型には依存しない。
    """

    tz: object

    def get_context_tasks(
        self, limit: int = 8, *, now: Optional[datetime] = None
    ) -> list[dict]: ...


class TasksContextProvider:
    """タスク状態の権威ブロックを str ContextBlock へ包む Provider。

    collect は source 自身を変更せず、build_task_context の結果を不変な ContextBlock へ
    包む。0 件 store でも build_task_context が返す非空の権威ブロックは必ず Block 化する。
    空・空白のみの結果、非 str の結果、例外時は None を返す。例外は型だけ
    logging.warning に残し、タスク本文・path・例外本文はログに含めない。
    """

    source = TASKS_SOURCE
    sensitivity = "personal"
    local_only = True

    @classmethod
    def collect(
        cls,
        source: TasksSource,
        limit: int = 8,
        *,
        now: Optional[datetime] = None,
    ) -> ContextBlock | None:
        """build_task_context の結果を str ContextBlock として返す。

        0 件 store でも build_task_context が返す非空の権威ブロックは必ず Block 化する。
        空・空白のみの結果、非 str の結果、または source が例外を上げた場合 (build_task_context
        が空文字列を返す場合) は None を返す。例外は型だけ logging.warning に残し、
        タスク本文・path・例外本文をログに含めない。
        """
        try:
            from src.tasks.store import build_task_context

            text = build_task_context(source, limit=limit, now=now)
        except Exception as exc:
            logger.warning(
                "TasksContextProvider: tasks failed (%s)",
                type(exc).__name__,
            )
            return None
        if not isinstance(text, str) or not text.strip():
            return None
        return ContextBlock(
            source=cls.source,
            content=text,
            sensitivity=cls.sensitivity,
            local_only=cls.local_only,
        )
