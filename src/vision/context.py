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
from typing import Callable, Optional
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
        camera: Optional[CameraCapture] = None,
        analyzer: Optional[VisionAnalyzer] = None,
        thread_factory: Optional[Callable[..., threading.Thread]] = None,
    ):
        """
        Args:
            camera_id: カメラデバイスID
            analysis_interval: フレーム解析間隔（秒）。大きいほどCPU負荷低
            emotion_model_path: emotion-ferplus ONNX モデルのパス
            width: カメラ解像度（幅）
            height: カメラ解像度（高さ）
            camera: CameraCapture の差し替え (テスト用)。None なら既定を生成
            analyzer: VisionAnalyzer の差し替え (テスト用)。None なら既定を生成
            thread_factory: 解析スレッド生成用ファクトリ (テスト用)。None なら threading.Thread
        """
        self.camera = camera or CameraCapture(
            device_id=camera_id,
            width=width,
            height=height,
            fps=15,
        )
        self.analyzer = analyzer or VisionAnalyzer(emotion_model_path=emotion_model_path)
        self.analysis_interval = analysis_interval

        self._state = VisionState()
        self._state_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._thread_factory = thread_factory or threading.Thread
        self._running = False
        self._paused = False
        self._stop_pending = False
        self._last_result: Optional[VisionResult] = None

    def start(self) -> bool:
        """カメラ + バックグラウンド解析を開始

        前回スレッドがまだ生存中 (stop の join がタイムアウトした等) の場合は
        重複起動を防ぐため False を返し、新規スレッドを生成しない。
        カメラ起動・スレッド生成・スレッド起動の失敗は捕捉して後始末し False を返す。
        ただしスレッド起動例外でスレッドが実際に生存している場合は起動成功扱い
        (requested-running) として True を返し、生存中のカメラも落とさない。
        実際に生存したスレッドがあれば所有権を保持して重複起動を防ぐ。
        死亡が確認できたスレッドだけを置き換えて再起動する。
        """
        if self._thread is not None and self._thread.is_alive():
            return False

        thread: Optional[threading.Thread] = None
        try:
            if not self.camera.start():
                return False
            thread = self._thread_factory(target=self._analysis_loop, daemon=True)
            thread.start()
            if not thread.is_alive():
                # 起動直後に死亡 (予期せぬ worker 死亡) した場合は後始末して False
                self._abort_start(thread)
                return False
        except Exception:
            if thread is not None and thread.is_alive():
                # start 例外だがスレッドは生存: 起動成功扱い (requested-running) を保持し、
                # 生存中のカメラを落とさない (ネスト整合性の維持)
                self._thread = thread
                self._running = True
                self._paused = False
                self._stop_pending = False
                return True
            self._abort_start(thread)
            return False

        self._thread = thread
        self._running = True
        self._paused = False
        self._stop_pending = False
        return True

    def _abort_start(self, thread: Optional[threading.Thread]):
        """start() 失敗時のベストエフォート後始末。

        開始済みカメラは停止を試みる。実際に生存したスレッドがあれば所有権を
        保持し (stop_pending)、重複起動を防ぐ。未生存スレッドは参照を破棄する。
        """
        self._running = False
        self._paused = False
        if thread is not None and thread.is_alive():
            self._thread = thread
            self._stop_pending = True
        else:
            self._thread = None
            self._stop_pending = False
        try:
            self.camera.stop()
        except Exception:
            pass

    def stop(self):
        """停止してリソースを解放。join がタイムアウトしてもスレッド所有権は保持する。

        カメラを先に停止/解放して解析スレッドのブロック (camera.get_frame() 等) を
        解除してから join する。join がタイムアウトした場合は _thread を保持したまま
        にし、重複再起動を防ぐ (_stop_pending を立てる)。
        スレッドの死亡が確認できた時点で所有権を解放する (その後 start() で再起動可能)。
        何度呼んでも安全 (冪等)。
        """
        self._running = False
        self.camera.stop()  # 先にカメラを止めて解析スレッドをブロック解除する

        thread = self._thread
        if thread is not None:
            if thread.is_alive():
                thread.join(timeout=5)
                if thread.is_alive():
                    self._stop_pending = True
                else:
                    self._thread = None
                    self._stop_pending = False
            else:
                self._thread = None
                self._stop_pending = False

    @property
    def is_running(self) -> bool:
        """active (スレッド生存 かつ 停止要求なし かつ カメラ稼働) のときだけ True。

        解析スレッドが予期せず死んだ場合や stop_pending 中は False になる。
        """
        if not self._running:
            return False
        if self._stop_pending:
            return False
        if self._thread is None or not self._thread.is_alive():
            return False
        return self.camera.is_running

    def pause(self) -> None:
        """解析を一時停止（カメラは維持）"""
        self._paused = True

    def resume(self) -> None:
        """解析を再開"""
        self._paused = False

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
        """APIレスポンス用の状態辞書

        固定キーのみ。例外内容やパス等の可変情報は含めない。
        """
        state = self.get_state()
        return {
            "running": self.is_running,
            "paused": self._paused,
            "stop_pending": self._stop_pending,
            "thread_alive": self._thread.is_alive() if self._thread else False,
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
        """バックグラウンド解析ループ

        camera.get_frame / sleep / ループ本体の想定外例外は致命的とみなし、
        _running を False にしてカメラをベストエフォートで停止する (finally)。
        解析エラー (analyzer.analyze) は従来どおり警告のみでループを継続する。
        finally は解析スレッドが死亡する前に完了するため、生存中スレッドは
        start() の重複起動チェックで捕捉され、再起動が生存中の capture を
        上書きすることはない。
        """
        try:
            # カメラ起動直後は安定するまで少し待つ
            time.sleep(0.5)

            while self._running:
                if self._paused:
                    time.sleep(self.analysis_interval)
                    continue

                frame = self.camera.get_frame()
                if frame is not None:
                    try:
                        result = self.analyzer.analyze(frame)
                        self._update_state(result)
                    except Exception:
                        # 解析エラーは警告のみ（ループ継続）
                        pass

                time.sleep(self.analysis_interval)
        except Exception:
            # camera.get_frame / sleep / ループ本体の想定外例外は致命的
            self._running = False
        finally:
            # ベストエフォートでカメラを停止 (冪等)
            try:
                self.camera.stop()
            except Exception:
                pass

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
