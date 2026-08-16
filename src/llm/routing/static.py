"""明示ルールだけで経路を選ぶ決定的Router。"""

from collections.abc import Mapping, Sequence

from src.llm.registry import ProviderRegistry
from src.llm.routing.contracts import NoRouteError, RouteDecision, RoutingRequest


class StaticRouter:
    """明示指定、profile、default、fallbackの順でProviderを選択する。"""

    def __init__(
        self,
        registry: ProviderRegistry,
        *,
        default_provider_id: str,
        profile_routes: Mapping[str, str] | None = None,
        fallback_provider_ids: Sequence[str] = (),
    ) -> None:
        self.registry = registry
        self.default_provider_id = default_provider_id
        self.profile_routes = dict(profile_routes or {})
        self.fallback_provider_ids = tuple(fallback_provider_ids)

        configured_ids = [
            default_provider_id,
            *self.profile_routes.values(),
            *self.fallback_provider_ids,
        ]
        for provider_id in configured_ids:
            registry.get(provider_id)
        if len(self.fallback_provider_ids) != len(set(self.fallback_provider_ids)):
            raise ValueError("fallback_provider_ids must not contain duplicates")

    @staticmethod
    def _remote_allowed(request: RoutingRequest) -> bool:
        return request.privacy == "cloud_allowed" and request.allow_cloud

    def route(self, request: RoutingRequest) -> RouteDecision:
        if request.requested_provider is not None:
            primary_id = request.requested_provider
            self.registry.get(primary_id)
            primary_reason = "explicit provider request"
        elif request.profile in self.profile_routes:
            primary_id = self.profile_routes[request.profile]
            primary_reason = f"profile route: {request.profile}"
        else:
            primary_id = self.default_provider_id
            primary_reason = "default route"

        candidates: list[str] = []
        for provider_id in (primary_id, *self.fallback_provider_ids):
            if provider_id not in candidates:
                candidates.append(provider_id)

        rejected: list[str] = []
        for provider_id in candidates:
            entry = self.registry.get(provider_id)
            if not entry.local and not self._remote_allowed(request):
                rejected.append(f"{provider_id}=cloud-disallowed")
                continue
            if not entry.provider.is_available():
                rejected.append(f"{provider_id}=unavailable")
                continue

            reason = primary_reason
            if provider_id != primary_id:
                details = ", ".join(rejected) or f"{primary_id}=rejected"
                reason = f"fallback from {primary_id}: {details}"
            remaining = tuple(
                candidate for candidate in candidates if candidate != provider_id
            )
            return RouteDecision(
                provider_id=provider_id,
                model=entry.provider.model,
                local=entry.local,
                reason=reason,
                fallback_provider_ids=remaining,
            )

        details = ", ".join(rejected) or "no configured candidates"
        raise NoRouteError(f"no allowed and available provider: {details}")
