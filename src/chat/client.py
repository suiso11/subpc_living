"""
Ollama APIクライアント
Phase 2: Ollamaとの通信を担当するモジュール
"""
import httpx
from typing import Generator
import json


class OllamaClient:
    """Ollama APIクライアント"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:7b-instruct-q4_K_M"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(base_url=self.base_url, timeout=120.0)

    def is_available(self) -> bool:
        """Ollamaサーバーが応答可能か確認"""
        try:
            resp = self._client.get("/api/tags")
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def list_models(self) -> list[str]:
        """利用可能なモデル一覧を取得"""
        try:
            resp = self._client.get("/api/tags")
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    def has_model(self, model: str | None = None) -> bool:
        """指定モデルが存在するか確認"""
        model = model or self.model
        models = self.list_models()
        # モデル名の部分一致で確認（タグ省略対応）
        return any(model in m or m in model for m in models)

    def generate(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        num_ctx: int = 4096,
        repeat_penalty: float = 1.1,
    ) -> str:
        """非ストリーミングでチャット応答を生成"""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_ctx": num_ctx,
                "repeat_penalty": repeat_penalty,
            },
        }
        resp = self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def generate_stream(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        num_ctx: int = 4096,
        repeat_penalty: float = 1.1,
    ) -> Generator[str, None, None]:
        """ストリーミングでチャット応答を生成（トークン単位で返す）"""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_ctx": num_ctx,
                "repeat_penalty": repeat_penalty,
            },
        }
        with self._client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    if not data.get("done", False):
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                    else:
                        # 最終チャンク: メタデータ含む
                        self._last_stats = {
                            "total_duration": data.get("total_duration", 0),
                            "eval_count": data.get("eval_count", 0),
                            "eval_duration": data.get("eval_duration", 0),
                        }

    @property
    def last_stats(self) -> dict:
        """最後の生成の統計情報"""
        return getattr(self, "_last_stats", {})

    def close(self) -> None:
        """クライアントをクローズ"""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
