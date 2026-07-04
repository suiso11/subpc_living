"""
スクリーンコンテキスト管理
スクリーンキャプチャ + VLM 描写を統合し、LLM に注入するコンテキストテキストを生成する。

バックグラウンドスレッドで定期的に画面をキャプチャして VLM で描写し、
「ユーザーが今 PC で何をしているか」を追跡する。

VLM 推論は重いため解析間隔は長め (デフォルト 90 秒)。
連続失敗が続いた場合は自動 pause して静かに無限失敗し続けないようにする。

設計は src/vision/context.py の VisionContext を踏襲している
(start/stop/is_running/pause/resume/get_state/get_context_text/get_status)。
"""
import copy
import threading
import time
from dataclasses import dataclass
from typing import Optional

from src.screen.capture import ScreenCapture
from src.screen.describer import ScreenDescriber


@dataclass
class ScreenState:
    """現在のスクリーン状態"""
    description: str = ""            # 最新の画面描写テキスト
    captured_at: float = 0.0         # 描写を取得した時刻 (epoch秒)
    analysis_count: int = 0          # 解析成功回数
    consecutive_failures: int = 0    # 連続失敗回数


class ScreenContext:
    """
    スクリーンコンテキストマネージャー

    画面キャプチャ → VLM 描写 をバックグラウンドで実行し、
    LLM プロンプトに注入するテキストコンテキストを提供する。
    """

    def __init__(
        self,
        analysis_interval: float = 90.0,
        base_url: str = "http://localhost:11434",
        model: Optional[str] = None,
        stale_after: float = 600.0,
        max_failures: int = 5,
        capture: Optional[ScreenCapture] = None,
        describer: Optional[ScreenDescriber] = None,
    ):
        """
        Args:
            analysis_interval: 解析間隔（秒）。VLM 推論が重いためデフォルト 90 秒
            base_url: Ollama サーバー URL
            model: VLM モデル名。None なら chat_config.json の model
            stale_after: 描写がこの秒数より古い場合はコンテキストに含めない (デフォルト 600秒=10分)
            max_failures: 連続失敗がこの回数に達したら自動 pause する
            capture: ScreenCapture の差し替え (テスト用)。None なら既定を生成
            describer: ScreenDescriber の差し替え (テスト用)。None なら既定を生成
        """
        self.capture = capture or ScreenCapture()
        self.describer = describer or ScreenDescriber(base_url=base_url, model=model)
        self.analysis_interval = analysis_interval
        self.stale_after = stale_after
        self.max_failures = max_failures

        self._state = ScreenState()
        self._state_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._paused = False

    def start(self) -> bool:
        """キャプチャ可否を確認し、バックグラウンド解析を開始"""
        if not self.capture.is_available():
            return False

        self._running = True
        self._paused = False
        self._thread = threading.Thread(target=self._analysis_loop, daemon=True)
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
        """解析を一時停止"""
        self._paused = True

    def resume(self) -> None:
        """解析を再開 (連続失敗カウントもリセット)"""
        with self._state_lock:
            self._state.consecutive_failures = 0
        self._paused = False

    def get_state(self) -> ScreenState:
        """現在のスクリーン状態を取得 (thread-safe)"""
        with self._state_lock:
            return copy.copy(self._state)

    def get_context_text(self) -> str:
        """
        LLM のシステムプロンプトに注入する画面コンテキストテキストを生成。

        Returns:
            画面情報を自然言語で記述した文字列。
            未取得 or 描写が stale_after より古い場合は空文字。
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
            "\n--- 画面情報 ---",
            f"- ユーザーの画面: {state.description} ({minutes}分前時点)",
        ]
        return "\n".join(lines)

    def get_status(self) -> dict:
        """API レスポンス用の状態辞書"""
        state = self.get_state()
        age = (time.time() - state.captured_at) if state.captured_at > 0 else None
        return {
            "running": self.is_running,
            "paused": self._paused,
            "description": state.description,
            "captured_at": state.captured_at,
            "age_seconds": age,
            "analysis_interval": self.analysis_interval,
            "analysis_count": state.analysis_count,
            "consecutive_failures": state.consecutive_failures,
            "model": self.describer.model,
        }

    # --- 内部: バックグラウンド解析 ---

    def _analysis_loop(self):
        """バックグラウンド解析ループ"""
        while self._running:
            if self._paused:
                time.sleep(self.analysis_interval)
                continue

            self._run_once()
            time.sleep(self.analysis_interval)

    def _run_once(self) -> bool:
        """1 回分のキャプチャ + 描写 + 状態更新。成功したら True。

        キャプチャや VLM 呼び出しが失敗しても例外は漏らさず、
        連続失敗を記録する (5回連続で自動 pause)。
        テストからも直接呼べるように分離している。
        """
        jpeg = None
        description = None
        try:
            jpeg = self.capture.capture()
            if jpeg is not None:
                description = self.describer.describe(jpeg)
        except Exception:
            description = None

        if description:
            self._record_success(description)
            return True
        else:
            self._record_failure()
            return False

    def _record_success(self, description: str) -> None:
        with self._state_lock:
            self._state.description = description
            self._state.captured_at = time.time()
            self._state.analysis_count += 1
            self._state.consecutive_failures = 0

    def _record_failure(self) -> None:
        should_pause = False
        with self._state_lock:
            self._state.consecutive_failures += 1
            if self._state.consecutive_failures >= self.max_failures:
                should_pause = True

        if should_pause and not self._paused:
            self._paused = True
            print(
                f"⚠️  Screen: {self.max_failures}回連続で描写に失敗したため"
                f"自動的に一時停止しました (resume で再開)"
            )
