"""
ヘルスチェックモジュール
Ollama 接続・ディスク容量・メモリ使用率・Web サーバー応答を検査する。
systemd ExecStartPre やWatchdog、外部監視ツールとの連携用。
"""
import sys
import shutil
from pathlib import Path
from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None

try:
    import httpx
except ImportError:
    httpx = None


class HealthChecker:
    """システム全体のヘルスチェックを実行するクラス"""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        web_url: str = "http://localhost:8000",
        disk_warn_percent: float = 90.0,
        memory_warn_percent: float = 90.0,
        timeout: float = 5.0,
    ):
        self.ollama_url = ollama_url
        self.web_url = web_url
        self.disk_warn_percent = disk_warn_percent
        self.memory_warn_percent = memory_warn_percent
        self.timeout = timeout

    def check_ollama(self, include_ollama: bool = True) -> dict:
        """Ollama API の接続確認

        Args:
            include_ollama: False のとき ``/api/tags`` プローブを実行しない
                (openai_compatible など Ollama 以外のローカル backend 用)。
                既定は True で従来どおり検査する (後方互換)。
        """
        if not include_ollama:
            return {"status": "skip", "message": "ollama check disabled"}
        try:
            if httpx is None:
                return {"status": "skip", "message": "httpx not installed"}
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(f"{self.ollama_url.rstrip('/')}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m["name"] for m in data.get("models", [])]
                    return {"status": "ok", "models": models}
                return {"status": "error", "message": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"status": "error", "message": f"request failed ({type(e).__name__})"}

    def check_web(self) -> dict:
        """Web UI サーバーの応答確認"""
        try:
            if httpx is None:
                return {"status": "skip", "message": "httpx not installed"}
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(f"{self.web_url.rstrip('/')}/api/health")
                if resp.status_code == 200:
                    return {"status": "ok", "data": resp.json()}
                return {"status": "error", "message": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"status": "error", "message": f"request failed ({type(e).__name__})"}

    def check_disk(self, path: str = "/") -> dict:
        """ディスク空き容量の確認"""
        try:
            usage = shutil.disk_usage(path)
            percent = (usage.used / usage.total) * 100
            free_gb = usage.free / (1024 ** 3)
            status = "ok" if percent < self.disk_warn_percent else "warning"
            return {
                "status": status,
                "used_percent": round(percent, 1),
                "free_gb": round(free_gb, 1),
                "total_gb": round(usage.total / (1024 ** 3), 1),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def check_memory(self) -> dict:
        """メモリ使用率の確認"""
        if psutil is None:
            return {"status": "skip", "message": "psutil not installed"}
        try:
            mem = psutil.virtual_memory()
            status = "ok" if mem.percent < self.memory_warn_percent else "warning"
            return {
                "status": status,
                "used_percent": round(mem.percent, 1),
                "available_gb": round(mem.available / (1024 ** 3), 1),
                "total_gb": round(mem.total / (1024 ** 3), 1),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def check_openai_compatible(self, base_url: str, api_key: Optional[str] = None) -> dict:
        """OpenAI 互換ローカル backend のモデル一覧エンドポイントを確認する。

        ``GET <base_url>/models`` をプローブし、レスポンスを分類する:
        - 200 かつ JSON の ``data`` がリスト: ``ok`` + モデルID列
        - 404 / 405: ``unknown`` (モデル一覧ディスカバリ非対応)
        - タイムアウト / 接続失敗 / 認証エラー / その他の非2xx /
          成功レスポンスの形が不正: ``error``

        Args:
            base_url: プローブ先の base URL。
            api_key: 空でない場合のみ ``Authorization: Bearer`` ヘッダを送る。
                キー自体は結果やエラーメッセージへ一切含めない。

        エラーメッセージには URL や資格情報を一切含めない。
        """
        if httpx is None:
            return {"status": "error", "message": "httpx not installed"}
        headers = {}
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key}"
        client = None
        try:
            client = httpx.Client(timeout=self.timeout)
            url = f"{base_url.rstrip('/')}/models"
            if headers:
                resp = client.get(url, headers=headers)
            else:
                resp = client.get(url)
            if resp.status_code in (404, 405):
                return {"status": "unknown", "message": "model discovery unsupported"}
            if resp.status_code != 200:
                return {"status": "error", "message": f"HTTP {resp.status_code}"}
            try:
                data = resp.json()
            except ValueError:
                return {"status": "error", "message": "malformed model list response"}
            if not isinstance(data, dict) or not isinstance(data.get("data"), list):
                return {"status": "error", "message": "malformed model list response"}
            models = [
                str(entry["id"])
                for entry in data["data"]
                if isinstance(entry, dict) and entry.get("id")
            ]
            return {"status": "ok", "models": models}
        except httpx.TimeoutException:
            return {"status": "error", "message": "request timed out"}
        except httpx.RequestError:
            return {"status": "error", "message": "connection failed"}
        except Exception as e:
            return {"status": "error", "message": f"request failed ({type(e).__name__})"}
        finally:
            if client is not None:
                client.close()

    def _probe_ollama(self, base_url: str) -> dict:
        """``<base_url>/api/tags`` をプローブする (check_ollama と同セマンティクス)。

        check_ollama は ``self.ollama_url`` 固定のため、選択backendの
        base URL へ向ける本ヘルパーを ``check_selected_provider`` から使う。
        全経路で client を閉じる。エラーメッセージに URL を含めない。
        """
        if httpx is None:
            return {"status": "error", "message": "httpx not installed"}
        client = None
        try:
            client = httpx.Client(timeout=self.timeout)
            resp = client.get(f"{base_url.rstrip('/')}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                models = [m["name"] for m in data.get("models", [])]
                return {"status": "ok", "models": models}
            return {"status": "error", "message": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"status": "error", "message": f"request failed ({type(e).__name__})"}
        finally:
            if client is not None:
                client.close()

    def check_selected_provider(
        self,
        kind: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> dict:
        """選択中のローカル backend 種別に応じてヘルスプローブをディスパッチする。

        Args:
            kind: ``"ollama"`` または ``"openai"`` / ``"openai_compatible"``
            base_url: プローブ先の base URL。Ollama では None のとき
                ``self.ollama_url`` を使う (OpenAI 互換では必須)。
            api_key: OpenAI 互換向けの任意キー。空なら Bearer ヘッダを送らない。
                キー自体は結果へ含めない。

        Returns:
            ``check_ollama`` / ``check_openai_compatible`` と同形式の単一チェック辞書。
            未知の kind は例外を投げず ``error`` を返す。
        """
        if kind == "ollama":
            return self._probe_ollama(base_url or self.ollama_url)
        if kind in ("openai", "openai_compatible"):
            if not base_url:
                return {"status": "error", "message": "provider base url required"}
            return self.check_openai_compatible(base_url, api_key=api_key)
        return {"status": "error", "message": "unknown provider kind"}

    def check_all(
        self,
        include_web: bool = False,
        include_ollama: bool = True,
        provider_kind: Optional[str] = None,
        provider_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
    ) -> dict:
        """
        全項目をチェックして結果を返す。

        Args:
            include_web: Web サーバー応答もチェックするか (デフォルト False)
            include_ollama: Ollama ``/api/tags`` プローブを実行するか
                (デフォルト True、後方互換)。``provider_kind`` 指定時はこちらではなく
                選択backendに従う。openai_compatible など Ollama 以外のローカル
                backend では False を渡すとプローブを省略する。
            provider_kind: 選択中ローカル backend ("ollama" / "openai_compatible")。
                指定時は include_ollama より優先し、Ollama は legacy の
                ``checks["ollama"]`` に、OpenAI 互換は ``checks["local_provider"]``
                (kind 付き、URL/キーなし) に出力する。
            provider_url: 選択backendの base URL。None なら Ollama では
                ``self.ollama_url`` を使う (OpenAI 互換ではエラー)。
            provider_api_key: 選択backendの任意 API キー。空なら Bearer ヘッダを
                送らない。キー自体は結果へ一切含めない。

        Returns:
            {
                "status": "ok" | "degraded" | "error",
                "checks": { "ollama": {...} | "local_provider": {...}, "disk": {...}, ... }
            }
        """
        checks: dict = {}
        if provider_kind is not None:
            selected = self.check_selected_provider(
                provider_kind, provider_url, api_key=provider_api_key
            )
            if provider_kind == "ollama":
                checks["ollama"] = selected
            else:
                selected["kind"] = provider_kind
                checks["local_provider"] = selected
        elif include_ollama:
            checks["ollama"] = self.check_ollama()
        checks["disk"] = self.check_disk()
        checks["memory"] = self.check_memory()
        if include_web:
            checks["web"] = self.check_web()

        # 全体ステータスを決定
        statuses = [c["status"] for c in checks.values()]
        if any(s == "error" for s in statuses):
            overall = "error"
        elif any(s in ("warning", "unknown") for s in statuses):
            overall = "degraded"
        else:
            overall = "ok"

        return {"status": overall, "checks": checks}


def main():
    """
    CLI エントリポイント。
    終了コード 0 = OK, 1 = degraded/error (または設定不正)
    systemd ExecStartPre や cron での利用を想定。
    """
    import json

    from src.chat.config import (
        ChatConfig,
        resolve_local_base_url,
        validate_local_provider_kind,
    )

    # プロジェクトルートの設定を読む
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "config" / "chat_config.json"

    try:
        config = ChatConfig.load(config_path)
        provider_kind = validate_local_provider_kind(config)
        provider_url = resolve_local_base_url(config)
        provider_api_key = config.resolve_local_api_key()
    except Exception as e:
        # URL・資格情報・例外詳細は出力しない
        print(json.dumps(
            {
                "status": "error",
                "message": f"invalid local provider config ({type(e).__name__})",
            },
            ensure_ascii=False,
            indent=2,
        ))
        sys.exit(1)

    checker = HealthChecker(ollama_url=provider_url)
    result = checker.check_all(
        include_web=False,
        provider_kind=provider_kind,
        provider_url=provider_url,
        provider_api_key=provider_api_key,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result["status"] == "ok":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
