"""
カメラキャプチャ
バックグラウンドスレッドで Webカメラからフレームを連続取得
"""
import threading
import time
import os
import numpy as np
from typing import Any, Callable, Optional

try:
    import cv2
    HAS_CV2 = True
    CAP_PROP_FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
    CAP_PROP_FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
    CAP_PROP_FPS = cv2.CAP_PROP_FPS
except ImportError:
    HAS_CV2 = False
    # cv2 未導入でもキャプチャファクトリを使うフェイクテストを可能にする固定値
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5

# status が返す状態
STOPPED = "stopped"
RUNNING = "running"
STOP_PENDING = "stop_pending"


class CameraCapture:
    """Webカメラキャプチャ（バックグラウンドスレッド）

    ライフサイクル契約:
    - ``status`` は "stopped" / "running" / "stop_pending" のいずれか
    - ``is_running`` は active (スレッド生存 かつ 停止要求なし かつ カメラ接続中) のときだけ True
    - ``stop()`` の join がタイムアウトしてスレッドが生き残っても ``_thread`` 参照は
      保持し、死亡が確認できるまで restart / 重複キャプチャをブロックする
    """

    def __init__(
        self,
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        capture_factory: Optional[Callable[[int], Any]] = None,
        thread_factory: Optional[Callable[..., threading.Thread]] = None,
    ):
        """
        Args:
            device_id: カメラデバイスID (/dev/video{N})
            width: キャプチャ幅
            height: キャプチャ高さ
            fps: フレームレート (CPU負荷制限用)
            capture_factory: VideoCapture 生成用ファクトリ (テスト用)。None なら cv2.VideoCapture
            thread_factory: キャプチャスレッド生成用ファクトリ (テスト用)。None なら threading.Thread
        """
        if not HAS_CV2:
            raise RuntimeError("opencv-python がインストールされていません: pip install opencv-python-headless")

        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps

        self._cap: Optional[Any] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_pending = False
        self._last_frame_time: float = 0.0
        self._frame_count: int = 0

        self._capture_factory = capture_factory
        self._thread_factory = thread_factory or threading.Thread

    def open(self) -> bool:
        """カメラデバイスを開く"""
        device_path = f"/dev/video{self.device_id}"
        if os.name != "nt" and not os.path.exists(device_path):
            print(f"  カメラデバイス {device_path} が存在しません")
            return False

        if self._capture_factory is not None:
            self._cap = self._capture_factory(self.device_id)
        else:
            self._cap = cv2.VideoCapture(self.device_id)
        if not self._cap.isOpened():
            print(f"  カメラ (device_id={self.device_id}) を開けません")
            self._cap = None
            return False

        self._cap.set(CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(CAP_PROP_FPS, self.fps)

        # 実際の設定値を取得
        actual_w = int(self._cap.get(CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(CAP_PROP_FPS)
        print(f"  カメラ: {actual_w}x{actual_h} @ {actual_fps:.0f}fps (device={self.device_id})")
        return True

    def start(self) -> bool:
        """バックグラウンドでフレーム取得を開始する。

        前回スレッドがまだ生存中 (stop の join がタイムアウトした等) は False を返し、
        重複キャプチャを防ぐ。死亡が確認できたスレッドだけを置き換えて再起動する。
        カメラ / スレッド生成・起動に失敗した場合は後始末して False を返す。
        起動例外時にスレッドが生存していた場合は起動成功扱い (requested-running) として
        ``True`` を返し、所有権 (``_thread``) を保持して ``stop()`` で後始末されるまで
        重複 start をブロックする。
        """
        if self._thread is not None and self._thread.is_alive():
            return False

        if not self.open():
            return False

        self._running = True
        self._stop_pending = False
        try:
            thread = self._thread_factory(target=self._capture_loop, daemon=True)
        except Exception:
            self._running = False
            self._stop_pending = False
            self._release_capture()
            return False

        # start() の例外時にも所有権を失わないよう、start より先に参照を保持する
        self._thread = thread
        try:
            thread.start()
        except Exception:
            if thread.is_alive():
                # start 例外だがスレッドは生存: 起動成功扱い (requested-running) を保持し、
                # カメラも解放しない
                self._running = True
                self._stop_pending = False
                return True
            # 未起動 / 死亡済み: 参照を解放して再起動可能にする
            self._running = False
            self._stop_pending = False
            self._thread = None
            self._release_capture()
            return False

        if not thread.is_alive():
            # 起動直後に死亡 (予期せぬ worker 死亡) した場合も後始末して再起動可能にする
            self._running = False
            self._stop_pending = False
            self._thread = None
            self._release_capture()
            return False

        return True

    def stop(self, timeout: float = 5.0):
        """カメラを停止する。

        停止シグナル (フラグ) とカメラ解放を先に実行してキャプチャスレッドのブロック
        (read 等) を解除してから join する。join がタイムアウトしてスレッドが生き残った
        場合は ``_thread`` 参照を保持し (stop_pending)、死亡が確認できるまで再起動・
        重複キャプチャをブロックする。何度呼んでも安全 (冪等)。
        """
        self._running = False
        self._release_capture()

        thread = self._thread
        if thread is not None:
            if thread.is_alive():
                try:
                    thread.join(timeout=timeout)
                except RuntimeError:
                    # 起動前に中断された (start 例外で保留された) スレッドは join 不能
                    pass
                if thread.is_alive():
                    self._stop_pending = True
                else:
                    self._thread = None
                    self._stop_pending = False
            else:
                self._thread = None
                self._stop_pending = False

    def get_frame(self) -> Optional[np.ndarray]:
        """最新フレームを取得 (thread-safe)"""
        with self._frame_lock:
            if self._frame is not None:
                return self._frame.copy()
            return None

    def get_jpeg(self, quality: int = 70) -> Optional[bytes]:
        """最新フレームをJPEGバイトで取得 (Web配信用)"""
        frame = self.get_frame()
        if frame is None:
            return None
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ret:
            return buf.tobytes()
        return None

    @property
    def status(self) -> str:
        """現在の状態: stopped / running / stop_pending"""
        if self._thread is None:
            return STOPPED
        if not self._thread.is_alive():
            return STOPPED
        if self._stop_pending:
            return STOP_PENDING
        return RUNNING

    @property
    def is_live(self) -> bool:
        """キャプチャスレッドが生存しているか"""
        return self._thread is not None and self._thread.is_alive()

    @property
    def stop_pending(self) -> bool:
        """停止要求済みだがスレッドの死亡が未確認か"""
        return self._stop_pending

    @property
    def is_running(self) -> bool:
        """active (スレッド生存 かつ 停止要求なし かつ カメラ接続中) のときだけ True。

        スレッドが予期せず死んだ場合や stop_pending 中は False になる。
        """
        if self.status != RUNNING:
            return False
        if self._cap is None:
            return False
        if hasattr(self._cap, 'isOpened'):
            return self._cap.isOpened()
        return bool(self._cap)

    @property
    def last_frame_time(self) -> float:
        return self._last_frame_time

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def _release_capture(self):
        """キャプチャを解放する (スレッドには触れない)。"""
        if self._cap is not None:
            if hasattr(self._cap, 'release'):
                try:
                    self._cap.release()
                except Exception:
                    pass
            self._cap = None

    def _capture_loop(self):
        """バックグラウンドフレーム取得ループ"""
        interval = 1.0 / self.fps
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                break

            ret, frame = self._cap.read()
            if ret:
                with self._frame_lock:
                    self._frame = frame
                    self._last_frame_time = time.time()
                    self._frame_count += 1

            time.sleep(interval)

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
