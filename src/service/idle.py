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

        # 起動直後は daemon が --initial-mode idle で GPU を idle にしているため、
        # IdleManager 側も IDLE から始める。最初の推論で active に遷移して wake が走る。
        self._state = self.STATE_IDLE
        # idle_timeout を過ぎた過去時刻にしておくことで、active 遷移後の
        # notify_inference_end → すぐ idle 判定される誤動作を防ぐ
        self._last_activity: float = time.time() - idle_timeout
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # GPU 電力マネージャー
        self._gpu_managers: list[GpuPowerManager] = []
        self._gpu_power_control_enabled = False
        self._gpu_power_control_reason = "GPU なし"

        # GPU復帰処理の重複抑制 (同じ wake を連打しない)
        self._wake_lock = threading.Lock()
        self._wake_in_progress = False

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
            print(f"[IdleManager] GPU電力管理対象: {', '.join(names)}", flush=True)
            ok, msg = self._gpu_managers[0].probe_power_control()
            if ok:
                self._gpu_power_control_enabled = True
                self._gpu_power_control_reason = "ok"
            else:
                self._gpu_power_control_enabled = False
                self._gpu_power_control_reason = msg
                print(f"[IdleManager] GPU電力制御は無効 ({self._gpu_power_control_reason})", flush=True)
        else:
            self._gpu_power_control_enabled = False
            self._gpu_power_control_reason = "nvidia-smi 未検出"

        self._running = True
        self._thread = threading.Thread(target=self._check_loop, daemon=True)
        self._thread.start()
        print(f"[IdleManager] 起動 (idle={int(self.idle_timeout)}s, deep_idle={int(self.deep_idle_timeout)}s)", flush=True)

    def stop(self) -> None:
        """アイドルマネージャーを停止"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        print("[IdleManager] 停止", flush=True)

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def gpu_power_control_enabled(self) -> bool:
        return self._gpu_power_control_enabled

    @property
    def gpu_power_control_reason(self) -> str:
        return self._gpu_power_control_reason

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

    def notify_inference_start(self, wait_for_gpu: bool = False) -> None:
        """LLM 推論開始を通知 → GPU をアクティブモードに

        ホットパスをブロックしないため、GPU 電力切替は別スレッドで実行する。
        daemon 呼び出し + nvidia-smi が数秒かかっても推論開始は待たされない。
        既にアクティブ状態なら GPU 切替は発行しない (冗長呼び出し抑制)。

        Args:
            wait_for_gpu: True の場合は GPU 電力切替の完了を待つ。
                STT のように直後に GPU を使う処理で指定する。
        """
        with self._lock:
            self._last_activity = time.time()
            transitioned = self._state != self.STATE_ACTIVE
            if transitioned:
                old_state = self._state
                self._state = self.STATE_ACTIVE
                print(f"[IdleManager] {old_state} → active (推論開始)", flush=True)
                self._restore_monitor_vision()
        wake_running = False
        if wait_for_gpu:
            with self._wake_lock:
                wake_running = self._wake_in_progress

        if wait_for_gpu and (transitioned or wake_running):
            self._set_gpu_active_blocking()
        elif transitioned:
            self._set_gpu_active_async()

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
                print(f"[IdleManager] チェックエラー: {e}", flush=True)
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
        print("[IdleManager] active → idle (GPU省電力モード)", flush=True)
        self._set_gpu_idle()

    def _on_become_deep_idle(self) -> None:
        """ディープアイドル状態に遷移した時の処理"""
        print("[IdleManager] idle → deep_idle (リソース縮小)", flush=True)
        # Monitor 収集間隔を延長
        if self._monitor_context is not None:
            try:
                self._monitor_context.collector.interval = self._monitor_idle_interval
            except Exception as e:
                print(f"[IdleManager] Monitor 間隔変更失敗: {e}", flush=True)

        # Vision 解析を一時停止
        if self._vision_context is not None:
            try:
                self._vision_context.pause()
            except AttributeError:
                pass  # pause メソッドがない場合は無視
            except Exception as e:
                print(f"[IdleManager] Vision 一時停止失敗: {e}", flush=True)

    def _on_become_active(self, from_state: str) -> None:
        """アクティブ状態に復帰した時の処理 (notify_activity 経由)

        GPU 復帰は非同期化し、Monitor/Vision のみ同期で戻す。
        """
        print(f"[IdleManager] {from_state} → active", flush=True)
        self._restore_monitor_vision()
        self._set_gpu_active_async()

    def _restore_monitor_vision(self) -> None:
        """Monitor 収集間隔と Vision 解析をアクティブ設定に戻す"""
        if self._monitor_context is not None:
            try:
                self._monitor_context.collector.interval = self._monitor_default_interval
            except Exception:
                pass
        if self._vision_context is not None:
            try:
                self._vision_context.resume()
            except AttributeError:
                pass
            except Exception:
                pass

    def _set_gpu_active_async(self) -> None:
        """GPU アクティブ化をバックグラウンドスレッドで実行 (重複抑制付き)"""
        if not self._gpu_power_control_enabled:
            return
        with self._wake_lock:
            if self._wake_in_progress:
                return  # 既に wake 中
            self._wake_in_progress = True

        def _worker() -> None:
            try:
                self._set_gpu_active()
            finally:
                with self._wake_lock:
                    self._wake_in_progress = False

        threading.Thread(target=_worker, daemon=True, name="gpu-wake").start()

    def _set_gpu_active_blocking(self) -> None:
        """GPU アクティブ化を同期実行する。

        STT 開始直前など、GPU が省電力のまま走ると初回処理が遅くなる経路で使う。
        """
        if not self._gpu_power_control_enabled:
            return
        waited_existing_wake = False
        while True:
            with self._wake_lock:
                if not self._wake_in_progress:
                    if waited_existing_wake:
                        return
                    self._wake_in_progress = True
                    break
            waited_existing_wake = True
            time.sleep(0.05)
        try:
            self._set_gpu_active()
        finally:
            with self._wake_lock:
                self._wake_in_progress = False

    def _set_gpu_idle(self) -> None:
        """全 GPU をアイドルモードに設定"""
        if not self._gpu_power_control_enabled:
            return
        for mgr in self._gpu_managers:
            if not mgr.power_control_available:
                continue
            ok, msg = mgr.set_idle_mode()
            if ok:
                print(f"[IdleManager] GPU{mgr.gpu_id}: idle ({mgr.idle_watts}W)", flush=True)
            else:
                print(f"[IdleManager] GPU{mgr.gpu_id}: idle設定失敗 ({msg})", flush=True)
                # 明示的な権限エラーのみ恒久無効化 (一時エラーは次回再試行)
                if "権限" in msg or "permission" in msg.lower() or "Insufficient" in msg:
                    self._gpu_power_control_enabled = False
                    self._gpu_power_control_reason = msg
                    print(f"[IdleManager] GPU電力制御を無効化 ({msg})", flush=True)
                    return

    def _set_gpu_active(self) -> None:
        """全 GPU をアクティブモードに設定"""
        if not self._gpu_power_control_enabled:
            return
        for mgr in self._gpu_managers:
            if not mgr.power_control_available:
                continue
            ok, msg = mgr.set_active_mode()
            if ok:
                print(f"[IdleManager] GPU{mgr.gpu_id}: active ({mgr.active_watts}W)", flush=True)
            else:
                print(f"[IdleManager] GPU{mgr.gpu_id}: active設定失敗 ({msg})", flush=True)
                # 明示的な権限エラーのみ恒久無効化 (一時エラーは次回再試行)
                if "権限" in msg or "permission" in msg.lower() or "Insufficient" in msg:
                    self._gpu_power_control_enabled = False
                    self._gpu_power_control_reason = msg
                    print(f"[IdleManager] GPU電力制御を無効化 ({msg})", flush=True)
                    return

    def get_status(self) -> dict:
        """APIレスポンス用の状態辞書"""
        return {
            "running": self.is_running,
            "state": self._state,
            "idle_seconds": round(self.idle_seconds, 0),
            "idle_timeout": self.idle_timeout,
            "deep_idle_timeout": self.deep_idle_timeout,
            "gpu_count": len(self._gpu_managers),
            "gpu_power_control_enabled": self._gpu_power_control_enabled,
            "gpu_power_control_reason": self._gpu_power_control_reason,
        }
