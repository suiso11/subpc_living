"""Monitor Context Provider。

MonitorContext.get_context_text() を呼び出し、その結果を
source=monitor / sensitivity=personal / local_only=True の str ContextBlock として返す。
monitor 側の収集・ストレージ (collector / storage / lifecycle) は変更・呼び出しせず、
MonitorContext 自身も変更しない。空・空白のみの結果、非 str の結果、
または monitor が例外を上げた場合は None を返す (例外は会話を止めない)。
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from src.context.contracts import ContextBlock

logger = logging.getLogger(__name__)


@runtime_checkable
class MonitorSource(Protocol):
    """get_context_text() -> str を提供する契約。

    monitor 側の MonitorContext へ runtime import して循環を作らないため、
    concrete 型には依存せず構造的適合 (duck-typing) の型契約だけを定義する。
    collect はこの契約に適合する任意の monitor を受け取れる。
    """

    def get_context_text(self) -> str: ...


class MonitorContextProvider:
    """PC モニター文脈を str ContextBlock へ包む Provider。

    collect は monitor 自身を変更せず、get_context_text() の結果を
    不変な ContextBlock へ包む。空・空白のみの結果、非 str の結果、例外時は
    None を返す。例外は型だけ logging.warning に残し、本文・例外本文は
    ログに含めない。
    """

    source = "monitor"
    sensitivity = "personal"
    local_only = True

    @classmethod
    def collect(cls, monitor: MonitorSource) -> ContextBlock | None:
        """get_context_text() の結果を str ContextBlock として返す。

        空・空白のみの結果、非 str の結果、または monitor が例外を上げた場合
        は None を返す。例外は型だけ logging.warning に残し、本文・
        例外本文をログに含めない。
        """
        try:
            text = monitor.get_context_text()
        except Exception as exc:
            logger.warning(
                "MonitorContextProvider: monitor failed (%s)",
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
