"""
アイドル状態管理コントローラー
ユーザー無操作を検知して GPU 電力・リソースを動的に制御する。

- ユーザー操作がない場合 → GPU をアイドルモードに切替
- LLM 推論開始時 → GPU をアクティブモードに切替
- 長時間アイドル → Monitor 収集間隔を延長・Vision 解析を一時停止
"""
import time
import threading
import logging
from typing import Optional

from src.service.power import GpuPowerManager, create_all_managers

logger = logging.getLogger(__name__)


class IdleManager:
    """
    アイドル状態管理コントローラー

    ユーザーアクティビティを追跡し、一定時間無操作で
    GPU 電力制限をアイドルモードに切替える。
    LLM 推論時はアクティブモードに復帰させる。

    Args:
        idle_timeout: アイドル判定までの秒数 (デフォルト 300秒 = 5分)
        deep_idle_timeout: ディープアイドル判定までの秒数 (デフォルト 1800秒 = 30分)
        check_interval: チェック間隔 (秒)
    """

    # 状態定義
    STATE_ACTIVE = "active"
    STATE_IDLE = "idle"
    STATE_DEEP_IDLE = "deep_idle"

    def __init__(
        self,
        idle_timeout: float = 300.0,
        deep_idle_timeout: float = 1800.0,
        check_interval: float = 30.0,
    ):
        self.idle_timeout = idle_timeout
        self.deep_idle_timeout = deep_idle_timeout
        self.check_interval = check_interval

        self._state = self.STATE_ACTIVE
        self._last_activity: float = time.time()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # GPU 電力マネージャー
        self._gpu_managers: list[GpuPowerManager] = []

        # 外部コンポーネント参照 (任意で設定)
        self._monitor_context = None
        self._vision_context = None
        self._monitor_default_interval: float = 30.0
        self._monitor_idle_interval: float = 120.0

    def start(
        self,
        monitor_context=None,
        vision_context=None,
    ) -> None:
        """
        アイドルマネージャーを起動

        Args:
            monitor_context: MonitorContext (収集間隔の動的変更用)
            vision_context: VisionContext (解析の一時停止用)
        """
        self._monitor_context = monitor_context
        self._vision_context = vision_context

        if monitor_context is not None:
            self._monitor_default_interval = monitor_context.collect_interval

        # GPU マネージャー初期化
        self._gpu_managers = create_all_managers()
        if self._gpu_managers:
            names = [m.get_gpu_info().get("name", f"GPU{m.gpu_id}") for m in self._gpu_managers]
            logger.info(f"IdleManager: GPU電力管理対象: {', '.join(names)}")

        self._running = True
        self._thread = threading.Thread(target=self._check_loop, daemon=True)
        self._thread.start()
        logger.info("IdleManager: 起動 (idle=%ds, deep_idle=%ds)",
                     int(self.idle_timeout), int(self.deep_idle_timeout))

    def stop(self) -> None:
        """アイドルマネージャーを停止"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("IdleManager: 停止")

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def idle_seconds(self) -> float:
        """最後のアクティビティからの経過秒数"""
        return time.time() - self._last_activity

    def notify_activity(self) -> None:
        """ユーザー操作を通知。アイドル状態なら即座にアクティブに復帰。"""
        with self._lock:
            self._last_activity = time.time()
            if self._state != self.STATE_ACTIVE:
                old_state = self._state
                self._state = self.STATE_ACTIVE
                self._on_become_active(old_state)

    def notify_inference_start(self) -> None:
        """LLM 推論開始を通知 → GPU をアクティブモードに"""
        self.notify_activity()
        self._set_gpu_active()

    def notify_inference_end(self) -> None:
        """LLM 推論終了を通知 (アクティビティタイマーリセット)"""
        with self._lock:
            self._last_activity = time.time()

    def _check_loop(self) -> None:
        """バックグラウンドチェックループ"""
        while self._running:
            try:
                self._check_idle_state()
            except Exception as e:
                logger.error(f"IdleManager チェックエラー: {e}")
            time.sleep(self.check_interval)

    def _check_idle_state(self) -> None:
        """アイドル状態を判定して遷移"""
        with self._lock:
            elapsed = time.time() - self._last_activity

            if self._state == self.STATE_ACTIVE:
                if elapsed >= self.idle_timeout:
                    self._state = self.STATE_IDLE
                    self._on_become_idle()

            elif self._state == self.STATE_IDLE:
                if elapsed >= self.deep_idle_timeout:
                    self._state = self.STATE_DEEP_IDLE
                    self._on_become_deep_idle()

    def _on_become_idle(self) -> None:
        """アイドル状態に遷移した時の処理"""
        logger.info("IdleManager: → idle (GPU省電力モード)")
        self._set_gpu_idle()

    def _on_become_deep_idle(self) -> None:
        """ディープアイドル状態に遷移した時の処理"""
        logger.info("IdleManager: → deep_idle (リソース縮小)")
        # Monitor 収集間隔を延長
        if self._monitor_context is not None:
            try:
                self._monitor_context.collector.interval = self._monitor_idle_interval
                logger.info(f"  Monitor 収集間隔: {self._monitor_idle_interval}s")
            except Exception as e:
                logger.warning(f"  Monitor 間隔変更失敗: {e}")

        # Vision 解析を一時停止
        if self._vision_context is not None:
            try:
                self._vision_context.pause()
                logger.info("  Vision 解析: 一時停止")
            except AttributeError:
                pass  # pause メソッドがない場合は無視
            except Exception as e:
                logger.warning(f"  Vision 一時停止失敗: {e}")

    def _on_become_active(self, from_state: str) -> None:
        """アクティブ状態に復帰した時の処理"""
        logger.info(f"IdleManager: {from_state} → active")

        # GPU アクティブモードに復帰
        self._set_gpu_active()

        # Monitor 収集間隔を戻す
        if self._monitor_context is not None:
            try:
                self._monitor_context.collector.interval = self._monitor_default_interval
            except Exception:
                pass

        # Vision 解析を再開
        if self._vision_context is not None:
            try:
                self._vision_context.resume()
            except AttributeError:
                pass
            except Exception:
                pass

    def _set_gpu_idle(self) -> None:
        """全 GPU をアイドルモードに設定"""
        for mgr in self._gpu_managers:
            ok, msg = mgr.set_idle_mode()
            if ok:
                logger.info(f"  GPU{mgr.gpu_id}: idle ({mgr.idle_watts}W)")
            else:
                logger.warning(f"  GPU{mgr.gpu_id}: idle設定失敗 ({msg})")

    def _set_gpu_active(self) -> None:
        """全 GPU をアクティブモードに設定"""
        for mgr in self._gpu_managers:
            ok, msg = mgr.set_active_mode()
            if ok:
                logger.info(f"  GPU{mgr.gpu_id}: active ({mgr.active_watts}W)")
            else:
                logger.warning(f"  GPU{mgr.gpu_id}: active設定失敗 ({msg})")

    def get_status(self) -> dict:
        """APIレスポンス用の状態辞書"""
        return {
            "running": self.is_running,
            "state": self._state,
            "idle_seconds": round(self.idle_seconds, 0),
            "idle_timeout": self.idle_timeout,
            "deep_idle_timeout": self.deep_idle_timeout,
            "gpu_count": len(self._gpu_managers),
        }
