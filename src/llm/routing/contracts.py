"""モデル経路選択に必要な契約。"""

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable


PrivacyMode = Literal["local_only", "local_preferred", "cloud_allowed"]


@runtime_checkable
class RoutingRequest(Protocol):
    """将来のAssistantRequestが構造的に満たすRouting用の最小入力。"""

    profile: str
    privacy: PrivacyMode
    requested_provider: str | None
    allow_cloud: bool


@dataclass(frozen=True)
class RouteDecision:
    provider_id: str
    model: str
    local: bool
    reason: str
    fallback_provider_ids: tuple[str, ...] = ()


class NoRouteError(RuntimeError):
    """Privacyと可用性を満たすProviderが存在しない。"""


class ModelRouter(Protocol):
    def route(self, request: RoutingRequest) -> RouteDecision:
        """要求に利用するProvider経路を決定する。"""
        ...
