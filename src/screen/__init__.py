"""
スクリーン認識モジュール
プライマリモニタをキャプチャ → Ollama VLM で日本語描写 → LLMコンテキストへ注入。

src/vision/ (カメラ入力) と同じアーキテクチャパターンを踏襲する。

2 つのモード:
  - local  : このサブPC自身の画面をキャプチャして VLM 描写する (ScreenContext)
  - remote : 別PC (メインPC) が push した画面を Web サーバーが描写した結果
             (data/screen/latest.json) を読むだけ (RemoteScreenContext)
モードは env SCREEN_CONTEXT_MODE (local|remote, default local) で切り替える。
"""
import os
from typing import Optional


def create_screen_context(mode: Optional[str] = None, **kwargs):
    """モードに応じて ScreenContext / RemoteScreenContext を生成するファクトリ。

    Args:
        mode: "local" | "remote"。None のとき env SCREEN_CONTEXT_MODE を読む
              (未設定なら "local")。
        **kwargs: local モードでは ScreenContext にそのまま渡す
                  (analysis_interval / base_url / model / stale_after など)。
                  remote モードではキャプチャも VLM 呼び出しもしないため
                  base_url / model / analysis_interval は無視し、stale_after のみ引き継ぐ。

    Returns:
        ScreenContext または RemoteScreenContext (同一の公開インターフェース)。
    """
    if mode is None:
        mode = os.environ.get("SCREEN_CONTEXT_MODE", "local")
    mode = (mode or "local").strip().lower()

    if mode == "remote":
        from src.screen.remote import RemoteScreenContext
        remote_kwargs = {}
        if "stale_after" in kwargs:
            remote_kwargs["stale_after"] = kwargs["stale_after"]
        return RemoteScreenContext(**remote_kwargs)

    from src.screen.context import ScreenContext
    return ScreenContext(**kwargs)
