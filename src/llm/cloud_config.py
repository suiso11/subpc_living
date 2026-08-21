"""クラウドProviderの有効化を制御する設定ゲート（Phase K）。

既定は無効。実キーは環境変数からのみ読み、有効時かつ ``api_key_env`` が設定された
ときだけ解決する。実設定・実キーをコードに埋め込まない。
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class CloudConfigError(RuntimeError):
    """CloudConfigの検証に失敗した。"""


@dataclass(frozen=True)
class CloudConfig:
    """クラウド経路の有効化設定。

    既定では ``enabled=False`` であり、どの呼び出し経路もクラウドProviderを
    登録しない。有効化は ``CloudConfig(enabled=True, ...)`` を factory へ渡す
    明示的な操作のみとする。
    """

    enabled: bool = False
    provider_id: str = "cloud"
    model: str = ""
    api_key_env: str = ""
    base_url: str | None = None

    def resolve_api_key(self) -> str | None:
        """有効時かつ ``api_key_env`` 指定時だけ環境変数からキーを返す。

        無効、または ``api_key_env`` 未指定、または未設定のときは ``None`` を返す。
        キーをコードに保持せず、実送信は将来の swap 先Providerへ委ねる。
        """
        if not self.enabled or not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env) or None

    def requires_key(self) -> bool:
        """有効かつ ``api_key_env`` 指定時、実送信にキーが必要。"""
        return self.enabled and bool(self.api_key_env)

    def validate(self) -> None:
        """有効化時に必要な条件を検証する。"""
        if not self.enabled:
            return
        if not self.model:
            raise CloudConfigError("cloud enabled but model is empty")
        if self.requires_key() and not self.resolve_api_key():
            raise CloudConfigError(
                f"cloud enabled but API key not found in {self.api_key_env!r}"
            )
