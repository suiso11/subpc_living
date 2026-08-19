"""History Context Provider。

現在の会話履歴（role/content dict列）を ContextMessage 列へコピーし、
source=history / sensitivity=personal / local_only=True の ContextBlock として返す。
会話本文を JSON 化・復号したり、Policy 外の side-channel で本文を運んだりしない。
"""

from __future__ import annotations

from collections.abc import Sequence

from src.context.contracts import ContextBlock, ContextMessage


class HistoryContextProvider:
    """現在の履歴を構造化 ContextBlock へコピーする Provider。

    collect は入力の履歴を変更せず、各 dict を不変な ContextMessage へ変換する。
    空履歴・ContextMessage 化できるメッセージが無い場合は None を返す。
    """

    source = "history"
    sensitivity = "personal"
    local_only = True

    @classmethod
    def collect(cls, messages: Sequence[dict]) -> ContextBlock | None:
        """現在の履歴を ContextMessage tuple にコピーした ContextBlock を返す。

        未知 role や非 str content は ContextMessage の構築時に明示的に拒否される。
        """
        if not messages:
            return None
        content = tuple(
            ContextMessage(role=message["role"], content=message["content"])
            for message in messages
        )
        if not content:
            return None
        return ContextBlock(
            source=cls.source,
            content=content,
            sensitivity=cls.sensitivity,
            local_only=cls.local_only,
        )
