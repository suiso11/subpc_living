"""ContextBlock契約。

ContextPolicy の入力となる送信候補ブロックの契約を定義する。
content は文字列のまま組み立てられる従来の文脈、または移行済みの
不変な構造化メッセージ列（ContextMessage）のどちらかで表現できる。
History は構造化 ContextBlock（ContextMessage列）として移行済みである。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from src.llm.contracts import ChatRole

Sensitivity = Literal["public", "personal", "secret"]

VALID_SENSITIVITIES: frozenset[str] = frozenset({"public", "personal", "secret"})

VALID_CHAT_ROLES: frozenset[str] = frozenset(get_args(ChatRole))

VALID_PRIVACY_MODES: frozenset[str] = frozenset({"local_only", "local_preferred", "cloud_allowed"})

TASKS_SOURCE = "tasks"


@dataclass(frozen=True)
class ContextMessage:
    """構造化文脈の1メッセージ。

    role は ChatRole、content は str に限定し、未知roleと非str contentを構築時に拒否する。
    既存ChatSession互換のため、content は空・空白 str も保持できる。
    変更できないため、Policy外のside-channelとして本文を改変できない。
    """

    role: ChatRole
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or self.role not in VALID_CHAT_ROLES:
            raise ValueError(f"unknown role: {self.role!r}")
        if not isinstance(self.content, str):
            raise TypeError(f"content must be str, got {type(self.content).__name__}")


@dataclass(frozen=True)
class ContextBlock:
    """送信候補の文脈1単位。

    source / content の空値と未知の sensitivity は構築時に明示的に拒否する。
    さらに source は str、content は str または非空 tuple[ContextMessage, ...]、
    local_only は bool、priority は int であることを構築時に検証する。
    文字列 content は空白のみを、tuple は空または ContextMessage 以外の要素を拒否する。
    mutable な list / dict は content として許可しない。Policy は content の本文を
    検査せず、この metadata だけで判断する。
    History は構造化 content（ContextMessage列）として移行済みである。
    """

    source: str
    content: str | tuple[ContextMessage, ...]
    sensitivity: Sensitivity = "public"
    local_only: bool = False
    priority: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.source, str):
            raise TypeError(f"source must be str, got {type(self.source).__name__}")
        if isinstance(self.content, str):
            if not self.content.strip():
                raise ValueError("content must not be empty")
        elif isinstance(self.content, tuple):
            if not self.content:
                raise ValueError("content tuple must not be empty")
            for message in self.content:
                if not isinstance(message, ContextMessage):
                    raise TypeError(
                        "content tuple items must be ContextMessage, "
                        f"got {type(message).__name__}"
                    )
        else:
            raise TypeError(
                "content must be str or tuple[ContextMessage, ...], "
                f"got {type(self.content).__name__}"
            )
        if not isinstance(self.local_only, bool):
            raise TypeError(f"local_only must be bool, got {type(self.local_only).__name__}")
        if not isinstance(self.priority, int) or isinstance(self.priority, bool):
            raise TypeError(f"priority must be int, got {type(self.priority).__name__}")
        if not self.source.strip():
            raise ValueError("source must not be empty")
        if self.sensitivity not in VALID_SENSITIVITIES:
            raise ValueError(f"unknown sensitivity: {self.sensitivity!r}")