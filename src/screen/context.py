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
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from src.screen.capture import ScreenCapture
from src.screen.describer import ScreenDescriber

logger = logging.getLogger(__name__)


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
        thread_factory: Optional[Callable[..., threading.Thread]] = None,
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
            thread_factory: 解析スレッド生成用ファクトリ (テスト用)。None なら threading.Thread
        """
        self.capture = capture or ScreenCapture()
        self.describer = describer or ScreenDescriber(base_url=base_url, model=model)
        self.analysis_interval = analysis_interval
        self.stale_after = stale_after
        self.max_failures = max_failures

        self._state = ScreenState()
        self._state_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._thread_factory = thread_factory or threading.Thread
        self._running = False
        self._paused = False
        self._stop_pending = False

    def start(self) -> bool:
        """キャプチャ可否を確認し、バックグラウンド解析を開始

        前回スレッドがまだ生存中 (stop の join がタイムアウトした等) の場合は
        重複起動を防ぐため False を返し、新規スレッドを生成しない。
        死亡が確認できたスレッドだけを置き換えて再起動する。

        可用性チェック・スレッド生成・スレッド起動の失敗は生例外を外に出さず
        False を返して部分状態を後始末する。ただし start が例外を投げてもスレッドが
        生存した場合 (部分起動) は True を返し、生存スレッドの所有権を破棄せず保持
        して二重起動を防ぐ。False を返すときは生存スレッドを保持しない
        (False は常に非生存を意味する)。
        """
        try:
            if not self.capture.is_available():
                return False
        except Exception:
            return False

        thread = self._thread
        if thread is not None:
            try:
                if thread.is_alive():
                    return False
            except Exception:
                return False

        try:
            thread = self._thread_factory(target=self._analysis_loop, daemon=True)
        except Exception:
            self._thread = None
            self._running = False
            self._paused = False
            self._stop_pending = False
            return False

        try:
            thread.start()
        except Exception:
            try:
                alive = thread.is_alive()
            except Exception:
                alive = False
            if alive:
                self._running = True
                self._paused = False
                self._stop_pending = False
                self._thread = thread
                return True
            self._running = False
            self._paused = False
            self._stop_pending = False
            self._thread = None
            return False
        if thread.is_alive():
            self._running = True
            self._paused = False
            self._stop_pending = False
            self._thread = thread
            return True
        self._running = False
        self._paused = False
        self._stop_pending = False
        self._thread = None
        return False

    def stop(self) -> None:
        """停止。join がタイムアウトしてもスレッド所有権は保持する。

        capture/describe がブロック中で join がタイムアウトした場合は _thread を
        保持したままにし、重複再起動を防ぐ (_stop_pending を立てる)。
        スレッドの死亡が確認できた時点で所有権を解放する (その後 start() で再起動可能)。
        何度呼んでも安全 (冪等)。
        """
        self._running = False
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout=5)
        if thread.is_alive():
            self._stop_pending = True
        else:
            self._thread = None
            self._stop_pending = False

    @property
    def is_running(self) -> bool:
        """所有するスレッドが実際に生存しているときだけ True。

        予期せぬスレッド死亡や停止中 (join タイムアウト) の状態を
        「稼働中」として報告しない。
        """
        if not self._running:
            return False
        thread = self._thread
        if thread is None:
            return False
        try:
            return thread.is_alive()
        except Exception:
            return False

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
        """API レスポンス用の状態辞書

        失敗メタデータは固定キー (bool) のみ。例外内容やパス等の可変情報は含めない。
        """
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
            "stop_pending": self._stop_pending,
            "thread_alive": self._thread.is_alive() if self._thread else False,
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
            logger.warning(
                "Screen: %s回連続で描写に失敗したため自動的に一時停止しました (resume で再開)",
                self.max_failures,
            )
