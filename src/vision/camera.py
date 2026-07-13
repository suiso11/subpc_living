"""
カメラキャプチャ
バックグラウンドスレッドで Webカメラからフレームを連続取得
"""
import threading
import time
import numpy as np
from typing import Optional

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class CameraCapture:
    """Webカメラキャプチャ（バックグラウンドスレッド）"""

    def __init__(
        self,
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
    ):
        """
        Args:
            device_id: カメラデバイスID (/dev/video{N})
            width: キャプチャ幅
            height: キャプチャ高さ
            fps: フレームレート (CPU負荷制限用)
        """
        if not HAS_CV2:
            raise RuntimeError("opencv-python がインストールされていません: pip install opencv-python-headless")

        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_frame_time: float = 0.0
        self._frame_count: int = 0

    def open(self) -> bool:
        """カメラデバイスを開く"""
        self._cap = cv2.VideoCapture(self.device_id)
        if not self._cap.isOpened():
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        # 実際の設定値を取得
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        print(f"  カメラ: {actual_w}x{actual_h} @ {actual_fps:.0f}fps (device={self.device_id})")
        return True

    def start(self) -> bool:
        """バックグラウンドでフレーム取得を開始"""
        if not self.open():
            return False

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """カメラを停止"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        if self._cap is not None:
            if hasattr(self._cap, 'release'):
                self._cap.release()
            self._cap = None

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
    def is_running(self) -> bool:
        if not self._running or self._cap is None:
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
        self.stop()
