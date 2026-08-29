"""クラウド送信の事前承認と匿名化（publicのみの送信Payload構築）。

Phase Kの必要条件:
- 送信Payloadを事前表示 (``ApprovalGate.preview``)
- 1リクエスト単位の明示承認 (``ApprovalGate.approve`` / ``require``)
- ローカル匿名化 (``CloudPayloadBuilder`` は ContextPolicy で public のみを選択)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.context.builder import ContextBuilder
from src.context.contracts import ContextBlock
from src.llm.routing.contracts import PrivacyMode


class ApprovalRequiredError(RuntimeError):
    """承認なしでクラウド送信しようとした。"""


class ApprovalDeniedError(RuntimeError):
    """明示的に却下されたリクエストでクラウド送信しようとした。"""


@dataclass(frozen=True)
class CloudPreview:
    """クラウドへ送信しようとする正確なPayloadの事前表示。"""

    request_id: str
    messages: list[dict[str, Any]]


class ApprovalGate:
    """1リクエスト単位の明示承認を管理する。"""

    def __init__(self, max_entries: int = 1024) -> None:
        self._max_entries = max_entries
        self._approved: dict[str, None] = {}
        self._denied: dict[str, None] = {}

    def preview(
        self, request_id: str, messages: Sequence[dict[str, Any]]
    ) -> CloudPreview:
        """送信予定Payloadをそのまま返す（事前表示用）。"""
        return CloudPreview(
            request_id=request_id,
            messages=[dict(message) for message in messages],
        )

    def approve(self, request_id: str) -> None:
        self._denied.pop(request_id, None)
        if request_id not in self._approved:
            if len(self._approved) >= self._max_entries:
                self._approved.pop(next(iter(self._approved)))
        self._approved[request_id] = None

    def deny(self, request_id: str) -> None:
        self._approved.pop(request_id, None)
        if request_id not in self._denied:
            if len(self._denied) >= self._max_entries:
                self._denied.pop(next(iter(self._denied)))
        self._denied[request_id] = None

    def revoke(self, request_id: str) -> None:
        self._approved.pop(request_id, None)
        self._denied.pop(request_id, None)

    def is_approved(self, request_id: str) -> bool:
        return request_id in self._approved

    def is_denied(self, request_id: str) -> bool:
        return request_id in self._denied

    def require(self, request_id: str) -> None:
        if request_id in self._denied:
            raise ApprovalDeniedError(f"cloud send denied: {request_id}")
        if request_id not in self._approved:
            raise ApprovalRequiredError(f"cloud send not approved: {request_id}")


class CloudPayloadBuilder:
    """ContextBlock から public のみを選択してクラウド送信Payloadを構築する。

    ``ContextPolicy.select(..., target_local=False)`` により personal / secret /
    local_only の block を除外する＝ローカル匿名化。history は ContextBlock
    として blocks に含めて渡し、Policy で選別する（target_local=False で除外）。
    """

    def __init__(self, base_system: str = "") -> None:
        self._builder = ContextBuilder(base_system)

    def build(
        self,
        blocks: Sequence[ContextBlock],
        *,
        privacy: PrivacyMode = "cloud_allowed",
    ) -> list[dict[str, Any]]:
        messages = self._builder.build_messages(
            blocks, privacy=privacy, target_local=False
        )
        return messages
