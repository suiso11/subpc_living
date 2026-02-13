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

    def check_ollama(self) -> dict:
        """Ollama API の接続確認"""
        try:
            if httpx is None:
                return {"status": "skip", "message": "httpx not installed"}
            client = httpx.Client(timeout=self.timeout)
            resp = client.get(f"{self.ollama_url}/api/tags")
            client.close()
            if resp.status_code == 200:
                data = resp.json()
                models = [m["name"] for m in data.get("models", [])]
                return {"status": "ok", "models": models}
            return {"status": "error", "message": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def check_web(self) -> dict:
        """Web UI サーバーの応答確認"""
        try:
            if httpx is None:
                return {"status": "skip", "message": "httpx not installed"}
            client = httpx.Client(timeout=self.timeout)
            resp = client.get(f"{self.web_url}/api/health")
            client.close()
            if resp.status_code == 200:
                return {"status": "ok", "data": resp.json()}
            return {"status": "error", "message": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

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

    def check_all(self, include_web: bool = False) -> dict:
        """
        全項目をチェックして結果を返す。

        Args:
            include_web: Web サーバー応答もチェックするか (デフォルト False)

        Returns:
            {
                "status": "ok" | "degraded" | "error",
                "checks": { "ollama": {...}, "disk": {...}, ... }
            }
        """
        checks = {
            "ollama": self.check_ollama(),
            "disk": self.check_disk(),
            "memory": self.check_memory(),
        }
        if include_web:
            checks["web"] = self.check_web()

        # 全体ステータスを決定
        statuses = [c["status"] for c in checks.values()]
        if any(s == "error" for s in statuses):
            overall = "error"
        elif any(s == "warning" for s in statuses):
            overall = "degraded"
        else:
            overall = "ok"

        return {"status": overall, "checks": checks}


def main():
    """
    CLI エントリポイント。
    終了コード 0 = OK, 1 = degraded/error
    systemd ExecStartPre や cron での利用を想定。
    """
    import json

    # プロジェクトルートの設定を読む
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "config" / "chat_config.json"

    ollama_url = "http://localhost:11434"
    if config_path.exists():
        try:
            import json as _json
            conf = _json.loads(config_path.read_text(encoding="utf-8"))
            ollama_url = conf.get("ollama_base_url", ollama_url)
        except Exception:
            pass

    checker = HealthChecker(ollama_url=ollama_url)
    result = checker.check_all(include_web=False)

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result["status"] == "ok":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
