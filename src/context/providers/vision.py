"""Vision Context Provider。

VisionContext.get_context_text() を呼び出し、その結果を
source=vision / sensitivity=secret / local_only=True の str ContextBlock として返す。
カメラ映像はカメラ由来の機密情報のため、vision 側の収集・解析 (camera / analysis /
start / stop) は変更・呼び出しせず、VisionContext 自身も変更しない。
空・空白のみの結果、非 str の結果、または vision が例外を上げた場合は None を返す
(例外は会話を止めない)。
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from src.context.contracts import ContextBlock

logger = logging.getLogger(__name__)


@runtime_checkable
class VisionSource(Protocol):
    """get_context_text() -> str を提供する契約。

    vision 側の VisionContext へ runtime import して循環を作らないため、
    concrete 型には依存せず構造的適合 (duck-typing) の型契約だけを定義する。
    collect はこの契約に適合する任意の vision を受け取れる。
    """

    def get_context_text(self) -> str: ...


class VisionContextProvider:
    """カメラ映像文脈を str ContextBlock へ包む Provider。

    collect は vision 自身を変更せず、get_context_text() の結果を
    不変な ContextBlock へ包む。空・空白のみの結果、非 str の結果、例外時は
    None を返す。例外は型だけ logging.warning に残し、本文・例外本文は
    ログに含めない。
    """

    source = "vision"
    sensitivity = "secret"
    local_only = True

    @classmethod
    def collect(cls, vision: VisionSource) -> ContextBlock | None:
        """get_context_text() の結果を str ContextBlock として返す。

        空・空白のみの結果、非 str の結果、または vision が例外を上げた場合
        は None を返す。例外は型だけ logging.warning に残し、本文・
        例外本文をログに含めない。
        """
        try:
            text = vision.get_context_text()
        except Exception as exc:
            logger.warning(
                "VisionContextProvider: vision failed (%s)",
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
