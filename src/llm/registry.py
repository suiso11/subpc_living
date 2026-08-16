"""Providerインスタンスと公開メタデータをIDで管理するRegistry。"""

from dataclasses import dataclass

from .provider import LLMProvider


class ProviderRegistryError(RuntimeError):
    """Provider Registryの設定または参照に失敗した。"""


class DuplicateProviderError(ProviderRegistryError):
    """同じProvider IDが複数回登録された。"""


class UnknownProviderError(ProviderRegistryError):
    """未登録のProvider IDが参照された。"""


@dataclass(frozen=True)
class ProviderEntry:
    """Routerが利用するProviderと安全性メタデータ。"""

    provider_id: str
    provider: LLMProvider
    local: bool
    profiles: tuple[str, ...] = ()


class ProviderRegistry:
    """Providerの登録、参照、終了処理を一元化する。"""

    def __init__(self) -> None:
        self._entries: dict[str, ProviderEntry] = {}

    def register(
        self,
        provider_id: str,
        provider: LLMProvider,
        *,
        local: bool,
        profiles: tuple[str, ...] = (),
    ) -> ProviderEntry:
        normalized_id = provider_id.strip()
        if not normalized_id:
            raise ValueError("provider_id must not be empty")
        if normalized_id in self._entries:
            raise DuplicateProviderError(f"provider already registered: {normalized_id}")

        entry = ProviderEntry(
            provider_id=normalized_id,
            provider=provider,
            local=local,
            profiles=tuple(profiles),
        )
        self._entries[normalized_id] = entry
        return entry

    def get(self, provider_id: str) -> ProviderEntry:
        try:
            return self._entries[provider_id]
        except KeyError as exc:
            raise UnknownProviderError(f"unknown provider: {provider_id}") from exc

    def __contains__(self, provider_id: object) -> bool:
        return provider_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> tuple[ProviderEntry, ...]:
        return tuple(self._entries.values())

    def close(self) -> None:
        """同じProvider実体を複数IDで保持しても一度だけ終了する。"""
        closed: set[int] = set()
        first_error: Exception | None = None
        for entry in self._entries.values():
            identity = id(entry.provider)
            if identity in closed:
                continue
            closed.add(identity)
            try:
                entry.provider.close()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
