"""
Ollama APIクライアント
Phase 2: Ollamaとの通信を担当するモジュール
"""
import httpx
import re
from collections import Counter
from typing import Generator
import json
import queue
import threading

# モデルが返す <think>...</think> 思考ブロックを除去するパターン
_THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_SHORT_REPEAT_PATTERN = re.compile(r"(.{1,12})\1{12,}", re.DOTALL)
_DEGENERATE_FALLBACK = "出力が乱れたので止めました。もう一度、短く言い直してください。"


class OllamaClient:
    """Ollama APIクライアント"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:7b-instruct-q4_K_M"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(base_url=self.base_url, timeout=300.0)

    def is_available(self) -> bool:
        """Ollamaサーバーが応答可能か確認"""
        try:
            resp = self._client.get("/api/tags", timeout=5.0)
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
        if not models:
            return False

        if model in models:
            return True

        # タグ省略時のみ、同名ベースモデルの :latest を許可する。
        if ":" not in model:
            return f"{model}:latest" in models

        return False

    @staticmethod
    def _strip_think(text: str) -> str:
        """思考トークン <think>...</think> を除去"""
        return _THINK_PATTERN.sub("", text).strip()

    @staticmethod
    def _is_degenerate_response(text: str) -> bool:
        """記号や短文の反復で崩れた生成を検出する。"""
        compact = re.sub(r"\s+", "", text)
        if len(compact) < 80:
            return False

        if re.search(r"(?:……。?){12,}", compact):
            return True

        punct_count = sum(1 for ch in compact if ch in "…。．.、,・!！?？")
        if punct_count / max(len(compact), 1) > 0.75:
            return True

        if _SHORT_REPEAT_PATTERN.search(compact):
            return True

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) >= 16:
            most_common = Counter(lines).most_common(1)[0][1]
            if most_common >= max(12, int(len(lines) * 0.65)):
                return True

        return False

    @classmethod
    def _sanitize_response(cls, text: str) -> str:
        cleaned = cls._strip_think(text)
        if cls._is_degenerate_response(cleaned):
            return _DEGENERATE_FALLBACK
        return cleaned

    @staticmethod
    def _build_options(
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        num_ctx: int,
    ) -> dict:
        """Ollama options を統一的に組み立てる"""
        return {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "num_ctx": num_ctx,
        }

    def generate(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        num_ctx: int = 8192,
    ) -> str:
        """非ストリーミングでチャット応答を生成"""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": -1,
            "think": False,
            "options": self._build_options(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                num_ctx=num_ctx,
            ),
        }
        resp = self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        raw = resp.json()["message"]["content"]
        return self._sanitize_response(raw)

    def generate_stream(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        num_ctx: int = 8192,
    ) -> Generator[str, None, None]:
        """ストリーミングでチャット応答を生成（トークン単位で返す）

        <think>...</think> ブロックはバッファリングして除去する。
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": -1,
            "think": False,
            "options": self._build_options(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                num_ctx=num_ctx,
            ),
        }

        in_think = False
        think_buf = ""

        with self._client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    if not data.get("done", False):
                        token = data.get("message", {}).get("content", "")
                        if not token:
                            continue

                        # <think> ブロック除去ロジック
                        if in_think:
                            think_buf += token
                            if "</think>" in think_buf:
                                # 思考終了: </think> 以降の残りを出力
                                after = think_buf.split("</think>", 1)[1]
                                in_think = False
                                think_buf = ""
                                if after.strip():
                                    yield after
                        elif "<think>" in token:
                            parts = token.split("<think>", 1)
                            before = parts[0]
                            if before:
                                yield before
                            in_think = True
                            think_buf = parts[1] if len(parts) > 1 else ""
                            # 即座に閉じるケース
                            if "</think>" in think_buf:
                                after = think_buf.split("</think>", 1)[1]
                                in_think = False
                                think_buf = ""
                                if after.strip():
                                    yield after
                        else:
                            yield token
                    else:
                        self._last_stats = {
                            "total_duration": data.get("total_duration", 0),
                            "eval_count": data.get("eval_count", 0),
                            "eval_duration": data.get("eval_duration", 0),
                        }

    def generate_stream_queue(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        num_ctx: int = 8192,
    ) -> queue.Queue:
        """スレッドセーフなキューベースのストリーミング。

        WebSocket から async で使うために、バックグラウンドスレッドで
        generate_stream() を走らせ、トークンをキューに投入する。
        キューには文字列トークンが入り、終了時に None が入る。
        エラー時は Exception オブジェクトが入る。
        """
        q: queue.Queue = queue.Queue(maxsize=256)

        def _worker():
            try:
                for token in self.generate_stream(
                    messages,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    num_ctx=num_ctx,
                ):
                    q.put(token)
            except Exception as e:
                q.put(e)
            finally:
                q.put(None)  # sentinel

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return q

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
