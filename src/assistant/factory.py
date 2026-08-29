"""ローカルProvider構成のAssistantサービスを組み立てる。"""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import TYPE_CHECKING

from src.assistant.cloud_service import CloudRouteBridge
from src.assistant.run_logger import RunLogger, SQLiteRunLogger
from src.assistant.service import AssistantService
from src.chat.config import (
    resolve_local_api_key,
    resolve_local_base_url,
    resolve_local_provider_id,
    validate_local_provider_kind,
)
from src.llm.approval import ApprovalGate
from src.llm.cloud_config import CloudConfig, CloudConfigError
from src.llm.contracts import GenerationOptions
from src.llm.providers.cloud import FakeCloudProvider
from src.llm.providers.cloud_http import OpenAICompatibleProvider
from src.llm.providers.local_openai import LocalOpenAICompatibleProvider
from src.llm.providers.ollama import OllamaProvider
from src.llm.registry import ProviderRegistry
from src.llm.routing.static import StaticRouter

if TYPE_CHECKING:
    from src.chat.config import ChatConfig
    from src.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

_UNSET = object()


def _resolve_run_logger(run_logger: RunLogger | None | object) -> RunLogger | None:
    """引数省略と明示Noneを区別し、既定のSQLiteRunLoggerを解決する。

    省略時は ``ASSISTANT_RUN_LOG_DB`` か既定の ``data/assistant/model_runs.db`` へ
    書き込むloggerを作る。初期化が失敗した場合はwarningを残して ``None`` へ
    fallbackし、Service組み立てを妨げない。明示的な ``None`` はloggerもDBも
    作らず無効のまま渡す。
    """
    if run_logger is _UNSET:
        db_path = os.environ.get("ASSISTANT_RUN_LOG_DB") or "data/assistant/model_runs.db"
        try:
            return SQLiteRunLogger(db_path)
        except (OSError, sqlite3.Error) as exc:
            logger.warning(
                "run logger initialization failed; falling back to disabled "
                "logger (db=%s): %s",
                db_path,
                exc,
            )
            return None
    return run_logger  # type: ignore[return-value]


def _build_options(config: "ChatConfig") -> GenerationOptions:
    return GenerationOptions(
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        repeat_penalty=config.repeat_penalty,
        num_ctx=config.num_ctx,
        num_predict=config.num_predict,
    )


def build_local_provider(
    config: "ChatConfig",
    *,
    provider=None,
    provider_id: str = "ollama",
) -> tuple[str, "LLMProvider"]:
    """config からローカルbackend Providerを一つ解決して返す。

    Provider の組み立てだけを行い、AssistantService / ProviderRegistry / Router /
    run logger は作らない。よって logger や DB の副作用は一切ない。

    ``provider`` が注入されていれば ``provider_id`` (既定 ``"ollama"``) と一緒に
    そのまま返す (後方互換)。無ければ ``local_provider_kind`` に応じて Ollama /
    OpenAI互換ローカルbackendを作り、config から導出した provider_id で返す。
    どちらのProviderも ``local=True`` として登録される想定で、cloud の承認・
    redaction セマンティクスを受けない。

    返り値は ``(resolved_id, provider)``。``resolved_id`` は Registry キー・
    error/log 用の provider_id として使える。
    """
    if provider is not None:
        return provider_id, provider

    kind = validate_local_provider_kind(config)
    resolved_id = resolve_local_provider_id(config)
    if kind == "openai_compatible":
        resolved: "LLMProvider" = LocalOpenAICompatibleProvider(
            model=config.model,
            base_url=resolve_local_base_url(config),
            provider_id=resolved_id,
            api_key=resolve_local_api_key(config),
        )
    else:
        resolved = OllamaProvider(
            base_url=resolve_local_base_url(config),
            model=config.model,
            provider_id=resolved_id,
        )
    return resolved_id, resolved


def build_local_service(
    config: "ChatConfig",
    *,
    provider=None,
    provider_id: str = "ollama",
    run_logger: RunLogger | None = _UNSET,  # type: ignore[assignment]
) -> tuple[AssistantService, ProviderRegistry]:
    """ChatConfigからローカル単体構成のServiceとRegistryを作る。

    クラウドProviderは一切登録しない（既定はローカルのみ）。backend は
    ``local_provider_kind`` で選び、無設定 (``"ollama"``) 時は従来通り
    Ollama + ``ollama_base_url`` + provider_id ``"ollama"`` を維持する。
    """
    resolved_id, resolved_provider = build_local_provider(
        config, provider=provider, provider_id=provider_id
    )
    registry = ProviderRegistry()
    registry.register(resolved_id, resolved_provider, local=True)
    router = StaticRouter(registry, default_provider_id=resolved_id)
    options = _build_options(config)
    return (
        AssistantService(
            registry,
            router,
            options=options,
            run_logger=_resolve_run_logger(run_logger),
        ),
        registry,
    )


def build_assistant_service(
    config: "ChatConfig",
    *,
    provider=None,
    provider_id: str = "ollama",
    cloud_config: CloudConfig | None = None,
    cloud_provider=None,
    approval: ApprovalGate | None = None,
    run_logger: RunLogger | None = _UNSET,  # type: ignore[assignment]
) -> tuple[AssistantService, ProviderRegistry, CloudRouteBridge | None]:
    """ローカル単体、または明示的に有効化されたクラウドを含むServiceを組み立てる。

    クラウドは ``cloud_config`` が渡され、かつ ``cloud_config.enabled`` のときだけ
    登録される。それ以外は ``build_local_service`` と同等にローカルのみ。

    ``cloud_provider`` を渡した場合は cloud_config の設定にかかわらず
    そのオブジェクトをクラウドProviderとして登録する（テスト注入用）。
    """
    resolved_id, resolved_provider = build_local_provider(
        config, provider=provider, provider_id=provider_id
    )
    registry = ProviderRegistry()
    registry.register(resolved_id, resolved_provider, local=True)
    router = StaticRouter(registry, default_provider_id=resolved_id)
    options = _build_options(config)
    service = AssistantService(
        registry,
        router,
        options=options,
        run_logger=_resolve_run_logger(run_logger),
    )

    bridge: CloudRouteBridge | None = None
    if cloud_config is not None and cloud_config.enabled:
        cloud_config.validate()
        if cloud_provider is not None:
            resolved = cloud_provider
        elif cloud_config.provider_kind == "openai_compatible":
            resolved = OpenAICompatibleProvider(
                model=cloud_config.model,
                api_key=cloud_config.resolve_api_key() or "",
                base_url=cloud_config.base_url or "https://api.openai.com/v1",
                provider_id=cloud_config.provider_id,
            )
        else:
            resolved = FakeCloudProvider(model=cloud_config.model or "cloud-model")
        registry.register(cloud_config.provider_id, resolved, local=False)
        bridge = CloudRouteBridge(
            registry,
            cloud_config.provider_id,
            approval=approval or ApprovalGate(),
            local_service=service,
            base_system=config.effective_system_prompt(),
        )
        service.set_cloud_bridge(bridge)
    return service, registry, bridge
