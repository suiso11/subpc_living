"""ContextBlock契約。

ContextPolicy の入力となる送信候補ブロックの契約を定義する。
会話履歴（History）は文字列のまま組み立てられる前段階にあり、
今回の ContextBlock / ContextPolicy の移行対象外である。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.llm.routing.contracts import PrivacyMode

Sensitivity = Literal["public", "personal", "secret"]

VALID_SENSITIVITIES: frozenset[str] = frozenset({"public", "personal", "secret"})

VALID_PRIVACY_MODES: frozenset[str] = frozenset({"local_only", "local_preferred", "cloud_allowed"})

TASKS_SOURCE = "tasks"


@dataclass(frozen=True)
class ContextBlock:
    """送信候補の文脈1単位。

    source / content の空値と未知の sensitivity は構築時に明示的に拒否する。
    さらに source / content は str、local_only は bool、priority は int で
    あることを構築時に検証する。Policy は content の本文を検査せず、
    この metadata だけで判断する。
    History は ContextBlock 化の前段階にあり、今回の対象外である。
    """

    source: str
    content: str
    sensitivity: Sensitivity = "public"
    local_only: bool = False
    priority: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.source, str):
            raise TypeError(f"source must be str, got {type(self.source).__name__}")
        if not isinstance(self.content, str):
            raise TypeError(f"content must be str, got {type(self.content).__name__}")
        if not isinstance(self.local_only, bool):
            raise TypeError(f"local_only must be bool, got {type(self.local_only).__name__}")
        if not isinstance(self.priority, int) or isinstance(self.priority, bool):
            raise TypeError(f"priority must be int, got {type(self.priority).__name__}")
        if not self.source.strip():
            raise ValueError("source must not be empty")
        if not self.content.strip():
            raise ValueError("content must not be empty")
        if self.sensitivity not in VALID_SENSITIVITIES:
            raise ValueError(f"unknown sensitivity: {self.sensitivity!r}")