"""
スクリーン描写 (VLM)
スクリーンショット JPEG を Ollama の vision 対応モデルに渡し、
「ユーザーが何をしているか」を日本語 1〜2 文で描写させる。
"""
import base64
import json
from pathlib import Path
from typing import Optional

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# VLM への描写指示
DESCRIBE_PROMPT = (
    "このPC画面のスクリーンショットで、ユーザーが何をしているかを"
    "日本語1〜2文で簡潔に述べてください。"
    "アプリ名やサイト名が分かれば含めてください。"
)

DEFAULT_MODEL = "gemma4:26b"


def _default_model() -> str:
    """chat_config.json の model を既定モデルとして読む。無ければ gemma4:26b。"""
    try:
        cfg_path = PROJECT_ROOT / "config" / "chat_config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            model = data.get("model")
            if model:
                return model
    except Exception:
        pass
    return DEFAULT_MODEL


class ScreenDescriber:
    """Ollama /api/chat に画像付きメッセージを投げて日本語描写を得る。"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: Optional[str] = None,
        timeout: float = 240.0,
        num_predict: int = 120,
        temperature: float = 0.2,
    ):
        """
        Args:
            base_url: Ollama サーバー URL
            model: 使用する vision 対応モデル。None なら chat_config.json の model
            timeout: HTTP タイムアウト秒 (VLM 推論は重い)
            num_predict: 最大生成トークン数 (描写は短くて良い)
            temperature: 生成温度 (描写なので低め)
        """
        self.base_url = base_url.rstrip("/")
        self.model = model or _default_model()
        self.timeout = timeout
        self.num_predict = num_predict
        self.temperature = temperature

    def describe(self, jpeg_bytes: bytes) -> Optional[str]:
        """JPEG バイトを VLM に渡し、日本語描写を返す。失敗時は None。"""
        if not jpeg_bytes:
            return None

        image_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": DESCRIBE_PROMPT,
                    "images": [image_b64],
                }
            ],
            "stream": False,
            # gemma4 等の reasoning モデルは think を無効化しないと num_predict を
            # 思考トークンに使い果たし content が空になる。短い描写が欲しいので think=False。
            "think": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }

        try:
            resp = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            message = data.get("message", {}) or {}
            text = (message.get("content") or "").strip()
            # think 無効化が効かないモデル向けフォールバック: thinking から拾う
            if not text:
                text = (message.get("thinking") or "").strip()
            return text or None
        except Exception:
            # 接続失敗・タイムアウト・パース失敗は None (ループ継続用)
            return None
