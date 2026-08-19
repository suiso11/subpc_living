"""ノードInventoryから複数Provider構成を組み立てる。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

from src.assistant.factory import _resolve_run_logger, _UNSET
from src.assistant.service import AssistantService
from src.llm.contracts import GenerationOptions
from src.llm.provider import LLMProvider
from src.llm.providers.ollama import OllamaProvider
from src.llm.registry import ProviderRegistry
from src.llm.routing.static import StaticRouter


class InvalidNodeInventoryError(ValueError):
    """ノードInventoryの設定が不正である。"""


@dataclass(frozen=True)
class ProviderSpec:
    provider_id: str
    base_url: str
    model: str
    local: bool = True
    profiles: tuple[str, ...] = ()
    node_id: str = ""


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    providers: tuple[ProviderSpec, ...]
    role: str = ""
    hostname: str = ""
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True)
class NodeInventory:
    nodes: tuple[NodeSpec, ...]
    default_provider_id: str
    fallback_provider_ids: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "NodeInventory":
        """Mappingを検証し、不変なノードInventoryへ変換する。"""
        raw_nodes = data.get("nodes")
        if not isinstance(raw_nodes, (list, tuple)):
            raise InvalidNodeInventoryError("nodes: must be a non-empty sequence")
        if not raw_nodes:
            raise InvalidNodeInventoryError("nodes: must not be empty")

        nodes: list[NodeSpec] = []
        known_node_ids: set[str] = set()
        known_provider_ids: set[str] = set()
        for node_index, raw_node in enumerate(raw_nodes):
            if not isinstance(raw_node, Mapping):
                raise InvalidNodeInventoryError(
                    f"node[{node_index}]: must be a mapping"
                )

            raw_node_id = raw_node.get("node_id", "")
            if not isinstance(raw_node_id, str):
                raise InvalidNodeInventoryError(
                    f"node[{node_index}].node_id: must be a string"
                )
            node_id = raw_node_id.strip()
            if not node_id:
                raise InvalidNodeInventoryError(
                    f"node[{node_index}].node_id: must not be empty"
                )
            if node_id in known_node_ids:
                raise InvalidNodeInventoryError(f"duplicate node_id: {node_id}")

            raw_providers = raw_node.get("providers")
            if not isinstance(raw_providers, (list, tuple)):
                raise InvalidNodeInventoryError(
                    f"node {node_id}: providers must be a non-empty sequence"
                )
            if not raw_providers:
                raise InvalidNodeInventoryError(
                    f"node {node_id}: providers must not be empty"
                )

            providers: list[ProviderSpec] = []
            for provider_index, raw_provider in enumerate(raw_providers):
                if not isinstance(raw_provider, Mapping):
                    raise InvalidNodeInventoryError(
                        f"node {node_id} provider[{provider_index}]: "
                        "must be a mapping"
                    )

                raw_provider_id = raw_provider.get("provider_id", "")
                if not isinstance(raw_provider_id, str):
                    raise InvalidNodeInventoryError(
                        f"node {node_id} provider[{provider_index}].provider_id: "
                        "must be a string"
                    )
                provider_id = raw_provider_id.strip()
                if not provider_id:
                    raise InvalidNodeInventoryError(
                        f"node {node_id} provider[{provider_index}].provider_id: "
                        "must not be empty"
                    )
                if provider_id in known_provider_ids:
                    raise InvalidNodeInventoryError(
                        f"duplicate provider_id: {provider_id}"
                    )

                base_url = raw_provider.get("base_url", "")
                if not isinstance(base_url, str):
                    raise InvalidNodeInventoryError(
                        f"provider {provider_id}: base_url must be a string"
                    )
                if not base_url.strip():
                    raise InvalidNodeInventoryError(
                        f"provider {provider_id}: base_url must not be empty"
                    )

                model = raw_provider.get("model", "")
                if not isinstance(model, str):
                    raise InvalidNodeInventoryError(
                        f"provider {provider_id}: model must be a string"
                    )
                if not model.strip():
                    raise InvalidNodeInventoryError(
                        f"provider {provider_id}: model must not be empty"
                    )

                local = raw_provider.get("local", True)
                if not isinstance(local, bool):
                    raise InvalidNodeInventoryError(
                        f"provider {provider_id}: local must be a bool"
                    )

                raw_profiles = raw_provider.get("profiles", ())
                if not isinstance(raw_profiles, (list, tuple)):
                    raise InvalidNodeInventoryError(
                        f"provider {provider_id}: profiles must be a list or tuple"
                    )
                if not all(isinstance(profile, str) for profile in raw_profiles):
                    raise InvalidNodeInventoryError(
                        f"provider {provider_id}: profiles must contain only strings"
                    )

                known_provider_ids.add(provider_id)
                providers.append(
                    ProviderSpec(
                        provider_id=provider_id,
                        base_url=base_url,
                        model=model,
                        local=local,
                        profiles=tuple(raw_profiles),
                        node_id=node_id,
                    )
                )

            raw_metadata = raw_node.get("metadata", {})
            if not isinstance(raw_metadata, Mapping):
                raise InvalidNodeInventoryError(
                    f"node {node_id}: metadata must be a mapping"
                )

            known_node_ids.add(node_id)
            nodes.append(
                NodeSpec(
                    node_id=node_id,
                    providers=tuple(providers),
                    role=raw_node.get("role", ""),
                    hostname=raw_node.get("hostname", ""),
                    metadata=MappingProxyType(dict(raw_metadata)),
                )
            )

        raw_default_provider_id = data.get("default_provider_id", "")
        if not isinstance(raw_default_provider_id, str):
            raise InvalidNodeInventoryError(
                "default_provider_id: must be a string"
            )
        default_provider_id = raw_default_provider_id.strip()
        if default_provider_id not in known_provider_ids:
            raise InvalidNodeInventoryError(
                f"unknown default_provider_id: {default_provider_id!r}"
            )

        raw_fallbacks = data.get("fallback_provider_ids", ())
        if not isinstance(raw_fallbacks, (list, tuple)):
            raise InvalidNodeInventoryError(
                "fallback_provider_ids: must be a list or tuple"
            )

        fallback_provider_ids: list[str] = []
        seen_fallbacks: set[str] = set()
        for raw_provider_id in raw_fallbacks:
            if not isinstance(raw_provider_id, str):
                raise InvalidNodeInventoryError(
                    "fallback_provider_ids: entries must be strings"
                )
            provider_id = raw_provider_id.strip()
            if provider_id not in known_provider_ids:
                raise InvalidNodeInventoryError(
                    f"unknown fallback_provider_id: {provider_id!r}"
                )
            if provider_id in seen_fallbacks:
                raise InvalidNodeInventoryError(
                    f"duplicate fallback_provider_id: {provider_id}"
                )
            if provider_id == default_provider_id:
                raise InvalidNodeInventoryError(
                    "default_provider_id "
                    f"{provider_id!r} must not appear in fallback_provider_ids"
                )
            seen_fallbacks.add(provider_id)
            fallback_provider_ids.append(provider_id)

        return cls(
            nodes=tuple(nodes),
            default_provider_id=default_provider_id,
            fallback_provider_ids=tuple(fallback_provider_ids),
        )

    @classmethod
    def load(cls, path: str | Path) -> "NodeInventory":
        """JSONファイルを読み、検証済みInventoryを返す。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise InvalidNodeInventoryError("inventory root: must be a mapping")
        return cls.from_mapping(data)

    def providers(self) -> tuple[ProviderSpec, ...]:
        """全物理ノードのProviderを定義順に平坦化して返す。"""
        return tuple(
            provider
            for node in self.nodes
            for provider in node.providers
        )


def _build_ollama_provider(spec: ProviderSpec) -> LLMProvider:
    return OllamaProvider(
        base_url=spec.base_url,
        model=spec.model,
        provider_id=spec.provider_id,
    )


def build_node_service(
    inventory: NodeInventory,
    *,
    options: GenerationOptions | None = None,
    provider_factory: Callable[[ProviderSpec], LLMProvider] | None = None,
    run_logger: Any | None = _UNSET,
) -> tuple[AssistantService, ProviderRegistry]:
    """Providerを定義順に登録し、profile routeを持つServiceを返す。

    同じprofileは定義順で最初のProviderが優先される。
    設定の並び順が経路に影響するため注意すること。
    """
    factory = provider_factory or _build_ollama_provider
    registry = ProviderRegistry()
    profile_routes: dict[str, str] = {}

    for spec in inventory.providers():
        registry.register(
            spec.provider_id,
            factory(spec),
            local=spec.local,
            profiles=spec.profiles,
        )
        for profile in spec.profiles:
            profile_routes.setdefault(profile, spec.provider_id)

    router = StaticRouter(
        registry,
        default_provider_id=inventory.default_provider_id,
        profile_routes=profile_routes,
        fallback_provider_ids=inventory.fallback_provider_ids,
    )
    return (
        AssistantService(
            registry,
            router,
            options=options,
            run_logger=_resolve_run_logger(run_logger),
        ),
        registry,
    )
