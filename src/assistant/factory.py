"""ローカルProvider構成のAssistantサービスを組み立てる。"""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import TYPE_CHECKING

from src.assistant.cloud_service import CloudRouteBridge
from src.assistant.run_logger import RunLogger, SQLiteRunLogger
from src.assistant.service import AssistantService
from src.llm.approval import ApprovalGate
from src.llm.cloud_config import CloudConfig, CloudConfigError
from src.llm.contracts import GenerationOptions
from src.llm.providers.cloud import FakeCloudProvider
from src.llm.providers.ollama import OllamaProvider
from src.llm.registry import ProviderRegistry
from src.llm.routing.static import StaticRouter

if TYPE_CHECKING:
    from src.chat.config import ChatConfig

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


def build_local_service(
    config: "ChatConfig",
    *,
    provider=None,
    provider_id: str = "ollama",
    run_logger: RunLogger | None = _UNSET,  # type: ignore[assignment]
) -> tuple[AssistantService, ProviderRegistry]:
    """ChatConfigからローカルOllama単体構成のServiceとRegistryを作る。

    クラウドProviderは一切登録しない（既定はローカルのみ）。
    """
    if provider is None:
        provider = OllamaProvider(
            base_url=config.ollama_base_url,
            model=config.model,
        )

    registry = ProviderRegistry()
    registry.register(provider_id, provider, local=True)
    router = StaticRouter(registry, default_provider_id=provider_id)
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
    approval: ApprovalGate | None = None,
    run_logger: RunLogger | None = _UNSET,  # type: ignore[assignment]
) -> tuple[AssistantService, ProviderRegistry, CloudRouteBridge | None]:
    """ローカル単体、または明示的に有効化されたクラウドを含むServiceを組み立てる。

    クラウドは ``cloud_config`` が渡され、かつ ``cloud_config.enabled`` のときだけ
    登録される。それ以外は ``build_local_service`` と同等にローカルのみ。
    """
    if provider is None:
        provider = OllamaProvider(
            base_url=config.ollama_base_url,
            model=config.model,
        )

    registry = ProviderRegistry()
    registry.register(provider_id, provider, local=True)
    router = StaticRouter(registry, default_provider_id=provider_id)
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
        cloud_provider = FakeCloudProvider(model=cloud_config.model or "cloud-model")
        registry.register(cloud_config.provider_id, cloud_provider, local=False)
        bridge = CloudRouteBridge(
            registry,
            cloud_config.provider_id,
            approval=approval or ApprovalGate(),
            local_service=service,
        )
    return service, registry, bridge
