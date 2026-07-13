"""
リモート画面コンテキスト (remote モード)

別PC (Windows メインPC など) 上のキャプチャエージェント
(scripts/screen_agent.py) がスクリーンショットを Web サーバーの
POST /api/screen/ingest に push し、サーバー側が VLM で 1 回だけ描写して
data/screen/latest.json に書き出す。

RemoteScreenContext はその latest.json を定期的に「読むだけ」で、
自分ではキャプチャも VLM 呼び出しも行わない (ローカルの ScreenContext と
同じ公開インターフェースを提供する)。

ファイルを読むだけなので負荷は軽く、ポーリング間隔はデフォルト 15 秒と短め。
captured_at が stale_after (デフォルト 600秒=10分) より古い / ファイルが無い
場合は get_context_text() は空文字を返す (ScreenContext と同じ鮮度ポリシー)。
"""
import copy
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LATEST_JSON = PROJECT_ROOT / "data" / "screen" / "latest.json"


@dataclass
class RemoteScreenState:
    """latest.json から読み取った最新のリモート画面状態"""
    description: str = ""            # VLM 描写テキスト
    captured_at: float = 0.0         # サーバーが画像を受信した時刻 (epoch秒)
    described_at: float = 0.0        # 描写が完了した時刻 (epoch秒)
    source: str = "remote"           # 供給元 (常に "remote")
    read_count: int = 0              # latest.json の読み取り成功回数


class RemoteScreenContext:
    """
    共有ファイル (data/screen/latest.json) を定期読取するだけの
    軽量スクリーンコンテキスト。

    ScreenContext と同じ公開インターフェース:
      start / stop / is_running / pause / resume /
      get_state / get_context_text / get_status
    """

    def __init__(
        self,
        poll_interval: float = 15.0,
        stale_after: float = 600.0,
        latest_path: Optional[Union[str, Path]] = None,
    ):
        """
        Args:
            poll_interval: latest.json の読取間隔（秒）。ファイルを読むだけなので短め
            stale_after: この秒数より古い描写はコンテキストに含めない (デフォルト 600秒=10分)
            latest_path: 読み取る JSON ファイルのパス。None なら data/screen/latest.json
        """
        self.poll_interval = poll_interval
        self.stale_after = stale_after
        self.latest_path = Path(latest_path) if latest_path else DEFAULT_LATEST_JSON

        self._state = RemoteScreenState()
        self._state_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._paused = False

    def start(self) -> bool:
        """バックグラウンドの読取ループを開始。ファイルを読むだけなので常に成功。"""
        self._running = True
        self._paused = False
        # 起動直後に一度読んでおく (即座にコンテキストを利用可能にする)
        self._read_once()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """停止"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    def pause(self) -> None:
        """読取を一時停止"""
        self._paused = True

    def resume(self) -> None:
        """読取を再開"""
        self._paused = False

    def get_state(self) -> RemoteScreenState:
        """現在の状態を取得 (thread-safe)"""
        with self._state_lock:
            return copy.copy(self._state)

    def get_context_text(self) -> str:
        """
        LLM のシステムプロンプトに注入する画面コンテキストテキストを生成。

        ScreenContext と同じ整形だが、「メインPCの画面」であることが分かる。
        未取得 / ファイル無し / captured_at が stale_after より古い場合は空文字。
        """
        if not self.is_running:
            return ""

        state = self.get_state()
        if not state.description or state.captured_at <= 0:
            return ""

        age = time.time() - state.captured_at
        if age > self.stale_after:
            return ""

        minutes = max(0, int(age / 60))
        lines = [
            "\n--- 画面情報 (メインPCの画面) ---",
            f"- ユーザーの画面(メインPC): {state.description} ({minutes}分前時点)",
        ]
        return "\n".join(lines)

    def get_status(self) -> dict:
        """API レスポンス用の状態辞書"""
        state = self.get_state()
        age = (time.time() - state.captured_at) if state.captured_at > 0 else None
        return {
            "mode": "remote",
            "running": self.is_running,
            "paused": self._paused,
            "description": state.description,
            "captured_at": state.captured_at,
            "described_at": state.described_at,
            "source": state.source,
            "age_seconds": age,
            "poll_interval": self.poll_interval,
            "stale_after": self.stale_after,
            "latest_path": str(self.latest_path),
            "read_count": state.read_count,
        }

    # --- 内部: バックグラウンド読取 ---

    def _poll_loop(self):
        while self._running:
            if not self._paused:
                self._read_once()
            time.sleep(self.poll_interval)

    def _read_once(self) -> bool:
        """latest.json を 1 回読んで状態を更新。有効な描写を読めたら True。

        - ファイル無し: 状態を空にする (captured_at=0)
        - 壊れた JSON / 読取失敗: 状態は維持 (更新しない)
        例外は外に漏らさない。
        """
        try:
            if not self.latest_path.exists():
                with self._state_lock:
                    self._state.description = ""
                    self._state.captured_at = 0.0
                return False
            raw = self.latest_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            description = (data.get("description") or "").strip()
            captured_at = float(data.get("captured_at") or 0.0)
        except Exception:
            # 壊れた JSON・部分書き込み・読取失敗は無視して前回状態を維持
            return False

        with self._state_lock:
            self._state.description = description
            self._state.captured_at = captured_at
            self._state.described_at = float(data.get("described_at") or 0.0)
            self._state.source = data.get("source") or "remote"
            self._state.read_count += 1

        return bool(description and captured_at > 0)
