"""ローカルProvider構成のAssistantサービスを組み立てる。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.assistant.service import AssistantService
from src.llm.contracts import GenerationOptions
from src.llm.providers.ollama import OllamaProvider
from src.llm.registry import ProviderRegistry
from src.llm.routing.static import StaticRouter

if TYPE_CHECKING:
    from src.chat.config import ChatConfig


def build_local_service(
    config: ChatConfig, *, provider=None, provider_id: str = "ollama"
) -> tuple[AssistantService, ProviderRegistry]:
    """ChatConfigからローカルOllama単体構成のServiceとRegistryを作る。"""
    if provider is None:
        provider = OllamaProvider(
            base_url=config.ollama_base_url,
            model=config.model,
        )

    registry = ProviderRegistry()
    registry.register(provider_id, provider, local=True)
    router = StaticRouter(registry, default_provider_id=provider_id)
    options = GenerationOptions(
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        repeat_penalty=config.repeat_penalty,
        num_ctx=config.num_ctx,
        num_predict=config.num_predict,
    )
    return AssistantService(registry, router, options=options), registry
