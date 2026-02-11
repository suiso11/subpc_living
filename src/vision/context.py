"""
映像コンテキスト管理
カメラ + 顔/感情検出を統合し、LLMに注入するコンテキストテキストを生成

バックグラウンドスレッドで定期的にフレームを解析し、
ユーザーの在席状況・表情をリアルタイムで追跡する。
"""
import copy
import threading
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from src.vision.camera import CameraCapture
from src.vision.detector import VisionAnalyzer, VisionResult


@dataclass
class VisionState:
    """現在の映像状態"""
    user_present: bool = False
    person_count: int = 0
    dominant_emotion: str = "unknown"
    dominant_emotion_ja: str = "不明"
    last_seen: float = 0.0          # ユーザーが最後に見えた時刻
    absent_since: float = 0.0       # ユーザーが離席し始めた時刻
    consecutive_emotion: str = ""   # 同じ感情が続いている場合
    emotion_streak: int = 0         # 同じ感情の連続回数
    analysis_count: int = 0         # 解析実行回数


class VisionContext:
    """
    映像コンテキストマネージャー

    カメラ → 顔検出 → 感情推定 をバックグラウンドで実行し、
    LLMプロンプトに注入するテキストコンテキストを提供する。

    解析間隔を設定して CPU 負荷を制御可能。
    """

    def __init__(
        self,
        camera_id: int = 0,
        analysis_interval: float = 2.0,
        emotion_model_path: Optional[str] = None,
        width: int = 640,
        height: int = 480,
    ):
        """
        Args:
            camera_id: カメラデバイスID
            analysis_interval: フレーム解析間隔（秒）。大きいほどCPU負荷低
            emotion_model_path: emotion-ferplus ONNX モデルのパス
            width: カメラ解像度（幅）
            height: カメラ解像度（高さ）
        """
        self.camera = CameraCapture(
            device_id=camera_id,
            width=width,
            height=height,
            fps=15,
        )
        self.analyzer = VisionAnalyzer(emotion_model_path=emotion_model_path)
        self.analysis_interval = analysis_interval

        self._state = VisionState()
        self._state_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_result: Optional[VisionResult] = None

    def start(self) -> bool:
        """カメラ + バックグラウンド解析を開始"""
        if not self.camera.start():
            return False

        self._running = True
        self._thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """停止してリソースを解放"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        self.camera.stop()

    @property
    def is_running(self) -> bool:
        return self._running and self.camera.is_running

    def get_state(self) -> VisionState:
        """現在の映像状態を取得 (thread-safe)"""
        with self._state_lock:
            return copy.copy(self._state)

    def get_last_result(self) -> Optional[VisionResult]:
        """最後のフレーム解析結果を取得"""
        with self._state_lock:
            return self._last_result

    def get_context_text(self) -> str:
        """
        LLMのシステムプロンプトに注入する映像コンテキストテキストを生成

        Returns:
            映像情報を自然言語で記述した文字列。カメラ非稼働時は空文字。
        """
        if not self.is_running:
            return ""

        state = self.get_state()
        now = time.time()

        lines = []
        lines.append("\n--- 現在の映像情報 ---")

        if state.user_present:
            if state.person_count == 1:
                lines.append("- ユーザーはカメラの前にいます")
            else:
                lines.append(f"- カメラの前に{state.person_count}人います")

            if state.dominant_emotion != "unknown":
                lines.append(f"- ユーザーの表情: {state.dominant_emotion_ja}")

                # 感情が継続している場合 (3回以上 = 約6秒以上)
                if state.emotion_streak >= 3:
                    lines.append("  (この表情がしばらく続いています)")
        else:
            if state.absent_since > 0:
                absent_sec = now - state.absent_since
                if absent_sec < 60:
                    lines.append("- ユーザーは少し前に離席しました")
                elif absent_sec < 300:
                    minutes = int(absent_sec / 60)
                    lines.append(f"- ユーザーは約{minutes}分前に離席しました")
                else:
                    minutes = int(absent_sec / 60)
                    lines.append(f"- ユーザーは{minutes}分以上離席しています")

                if state.last_seen > 0:
                    last_ts = time.strftime("%H:%M", time.localtime(state.last_seen))
                    lines.append(f"  (最後に確認: {last_ts})")
            else:
                lines.append("- ユーザーはカメラの前にいません")

        return "\n".join(lines)

    def get_status(self) -> dict:
        """APIレスポンス用の状態辞書"""
        state = self.get_state()
        return {
            "running": self.is_running,
            "user_present": state.user_present,
            "person_count": state.person_count,
            "emotion": state.dominant_emotion,
            "emotion_ja": state.dominant_emotion_ja,
            "emotion_detection": self.analyzer.has_emotion,
            "analysis_interval": self.analysis_interval,
            "analysis_count": state.analysis_count,
        }

    # --- 内部: バックグラウンド解析 ---

    def _analysis_loop(self):
        """バックグラウンド解析ループ"""
        # カメラ起動直後は安定するまで少し待つ
        time.sleep(0.5)

        while self._running:
            frame = self.camera.get_frame()
            if frame is not None:
                try:
                    result = self.analyzer.analyze(frame)
                    self._update_state(result)
                except Exception as e:
                    # 解析エラーは警告のみ（ループ継続）
                    pass

            time.sleep(self.analysis_interval)

    def _update_state(self, result: VisionResult):
        """解析結果から状態を更新 (thread-safe)"""
        with self._state_lock:
            self._last_result = result
            now = time.time()

            prev_present = self._state.user_present
            self._state.user_present = result.user_present
            self._state.person_count = result.person_count
            self._state.analysis_count += 1

            if result.user_present:
                self._state.last_seen = now
                self._state.absent_since = 0.0
                self._state.dominant_emotion = result.dominant_emotion
                self._state.dominant_emotion_ja = result.dominant_emotion_ja

                # 感情の連続カウント
                if result.dominant_emotion == self._state.consecutive_emotion:
                    self._state.emotion_streak += 1
                else:
                    self._state.consecutive_emotion = result.dominant_emotion
                    self._state.emotion_streak = 1
            else:
                # ユーザーが離席した瞬間を記録
                if prev_present and not result.user_present:
                    self._state.absent_since = now

                self._state.dominant_emotion = "unknown"
                self._state.dominant_emotion_ja = "不明"
                self._state.emotion_streak = 0
