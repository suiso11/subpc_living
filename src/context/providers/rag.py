"""RAG Context Provider。

RAGRetriever.build_context_prompt(query) を呼び出し、その結果を
source=rag / sensitivity=personal / local_only=True の str ContextBlock として返す。
retriever 側の store_turn / store_knowledge / retrieve は変更・呼び出しせず、
RAGRetriever 自身も変更しない。空・空白のみの結果、非 str の結果、
または retriever が例外を上げた場合は None を返す (例外は会話を止めない)。
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from src.context.contracts import ContextBlock

logger = logging.getLogger(__name__)


@runtime_checkable
class RAGSource(Protocol):
    """build_context_prompt(query) -> str を提供する契約。

    memory 側の RAGRetriever へ runtime import して循環を作らないため、
    concrete 型には依存せず構造的適合 (duck-typing) の型契約だけを定義する。
    collect はこの契約に適合する任意の retriever を受け取れる。
    """

    def build_context_prompt(self, query: str) -> str: ...


class RAGContextProvider:
    """RAG 検索文脈を str ContextBlock へ包む Provider。

    collect は retriever 自身を変更せず、build_context_prompt(query) の結果を
    不変な ContextBlock へ包む。空・空白のみの結果、非 str の結果、例外時は
    None を返す。例外は型だけ logging.warning に残し、query・本文・例外本文は
    ログに含めない。
    """

    source = "rag"
    sensitivity = "personal"
    local_only = True

    @classmethod
    def collect(cls, retriever: RAGSource, query: str) -> ContextBlock | None:
        """build_context_prompt(query) の結果を str ContextBlock として返す。

        空・空白のみの結果、非 str の結果、または retriever が例外を上げた場合
        は None を返す。例外は型だけ logging.warning に残し、query・本文・
        例外本文をログに含めない。
        """
        try:
            text = retriever.build_context_prompt(query)
        except Exception as exc:
            logger.warning(
                "RAGContextProvider: retriever failed (%s)", type(exc).__name__
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
