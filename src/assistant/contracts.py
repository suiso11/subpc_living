"""Assistant層で共有する要求、応答、例外の契約。"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from src.llm.routing.contracts import PrivacyMode, RouteDecision


AssistantChannel = Literal["cli", "web", "discord", "voice", "internal"]
AssistantProfile = Literal[
    "chat_auto",
    "voice_fast",
    "task_local",
    "code_auto",
    "deep_reasoning",
    "private_local",
]


@dataclass(frozen=True)
class AssistantRequest:
    """Assistantへ渡す1回の生成要求。

    ``requested_model`` はPhase Cでは無視され、応答経路のmodelには実際に利用した
    Providerの固定modelが設定される。
    """

    text: str
    conversation_id: str
    channel: AssistantChannel
    profile: AssistantProfile = "chat_auto"
    privacy: PrivacyMode = "local_preferred"
    requested_provider: str | None = None
    requested_model: str | None = None
    allow_cloud: bool = False


@dataclass(frozen=True)
class AssistantResponse:
    """生成本文と実際に利用した経路、計測値をまとめた応答。"""

    text: str
    route: RouteDecision
    latency_ms: int
    stats: Mapping[str, Any]


class AssistantError(RuntimeError):
    """Assistant処理に失敗したときの共通基底例外。"""


class AssistantGenerationError(AssistantError):
    """候補Providerのすべてで生成に失敗した。"""

    def __init__(self, attempts: tuple[tuple[str, str], ...]) -> None:
        self.attempts = attempts
        details = ", ".join(
            f"{provider_id}={reason}" for provider_id, reason in attempts
        )
        super().__init__(f"all provider candidates failed: {details}")
