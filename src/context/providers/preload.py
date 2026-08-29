"""Preload Context Provider。

SessionPreloader.build_preload_context() を呼び出し、その結果を
source=preload / sensitivity=personal / local_only=True の str ContextBlock として返す。
build_preload_context() は profile・schedule・summary・時刻を一つの str へまとめる
Preload 移行であり、独立した Profile Provider ではない。
preloader の例外は会話を止めず、標準 logging で型だけ warning する (本文はログしない)。
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from src.context.contracts import ContextBlock

logger = logging.getLogger(__name__)


@runtime_checkable
class PreloadSource(Protocol):
    """build_preload_context() -> str を提供する契約。

    persona 側の SessionPreloader へ runtime import して循環を作らないため、
    concrete 型には依存せず構造的適合 (duck-typing) の型契約だけを定義する。
    collect はこの契約に適合する任意の preloader を受け取れる。
    """

    def build_preload_context(self) -> str: ...


class PreloadContextProvider:
    """プリロード文脈を str ContextBlock へ包む Provider。

    collect は preloader 自身を変更せず、build_preload_context() の結果を
    不変な ContextBlock へ包む。空・空白のみの結果や例外時は None を返す。
    この結果は SessionPreloader が profile・schedule・summary・時刻を
    一括でまとめた Preload であり、独立 Profile Provider の成果ではない。
    """

    source = "preload"
    sensitivity = "personal"
    local_only = True

    @classmethod
    def collect(cls, preloader: PreloadSource) -> ContextBlock | None:
        """build_preload_context() の結果を str ContextBlock として返す。

        空・空白のみの結果、または preloader が例外を上げた場合は None を返す。
        例外は型だけ logging.warning に残し、本文はログに含めない。
        """
        try:
            text = preloader.build_preload_context()
        except Exception as exc:
            logger.warning(
                "PreloadContextProvider: preloader failed (%s)", type(exc).__name__
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
