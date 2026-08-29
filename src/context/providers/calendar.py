"""Calendar Context Provider。

CalendarContext.get_context_text() を呼び出し、その結果を
source=calendar / sensitivity=personal / local_only=True の str ContextBlock として返す。
calendar 側の file / sync 実装 (calendar_sync / sync lifecycle) は変更・呼び出しせず、
CalendarContext 自身も変更しない。空・空白のみの結果、非 str の結果、
または calendar が例外を上げた場合は None を返す (例外は会話を止めない)。
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from src.context.contracts import ContextBlock

logger = logging.getLogger(__name__)


@runtime_checkable
class CalendarSource(Protocol):
    """get_context_text() -> str を提供する契約。

    calendar 側の CalendarContext へ runtime import して循環を作らないため、
    concrete 型には依存せず構造的適合 (duck-typing) の型契約だけを定義する。
    collect はこの契約に適合する任意の calendar を受け取れる。
    """

    def get_context_text(self) -> str: ...


class CalendarContextProvider:
    """予定文脈を str ContextBlock へ包む Provider。

    collect は calendar 自身を変更せず、get_context_text() の結果を
    不変な ContextBlock へ包む。空・空白のみの結果、非 str の結果、例外時は
    None を返す。例外は型だけ logging.warning に残し、本文・例外本文は
    ログに含めない。
    """

    source = "calendar"
    sensitivity = "personal"
    local_only = True

    @classmethod
    def collect(cls, calendar: CalendarSource) -> ContextBlock | None:
        """get_context_text() の結果を str ContextBlock として返す。

        空・空白のみの結果、非 str の結果、または calendar が例外を上げた場合
        は None を返す。例外は型だけ logging.warning に残し、本文・
        例外本文をログに含めない。
        """
        try:
            text = calendar.get_context_text()
        except Exception as exc:
            logger.warning(
                "CalendarContextProvider: calendar failed (%s)",
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
