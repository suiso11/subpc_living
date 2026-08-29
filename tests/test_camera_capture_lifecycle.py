"""
CameraCapture の start/stop ライフサイクルを決定的に検証する。

実カメラ・実スレッドは使わず、FakeCapture / FakeThread で join タイムアウト・
スレッド死亡・重複起動防止・stop 冪等性・status/is_running の状態遷移を検証する。
"""
from __future__ import annotations

import unittest

import src.vision.camera as camera_mod
from src.vision.camera import CameraCapture, RUNNING, STOPPED, STOP_PENDING


class FakeCapture:
    """キャプチャデバイスのフェイク。"""

    def __init__(self, device_id: int, available: bool = True):
        self.device_id = device_id
        self._opened = available
        self.released = False
        self.set_calls = []
        self.get_calls = 0

    def isOpened(self) -> bool:
        return self._opened

    def set(self, prop, value):
        self.set_calls.append((prop, value))

    def get(self, prop):
        self.get_calls += 1
        return 0.0

    def read(self):
        return False, None

    def release(self):
        self.released = True
        self._opened = False


class FakeCaptureFactory:
    """生成した FakeCapture を記録するファクトリ。"""

    def __init__(self, available: bool = True):
        self.available = available
        self.instances = []

    def __call__(self, device_id: int) -> FakeCapture:
        cap = FakeCapture(device_id, available=self.available)
        self.instances.append(cap)
        return cap


class FakeThread:
    """join の挙動 (即死 or タイムアウト相当) を決定的に制御するフェイクスレッド。

    join_blocks=True の間は join() を呼んでも生き残り、join タイムアウトを再現する。
    kill() で明示的に死亡させられる。start_error を設定すると start() が例外を投げる。
    """

    def __init__(self, target=None, daemon=True):
        self.target = target
        self.daemon = daemon
        self._alive = False
        self.join_blocks = False
        self.starts_dead = False
        self.start_error = None
        self.stays_alive_on_error = False
        self.start_count = 0
        self.join_count = 0

    def start(self):
        self.start_count += 1
        self._alive = not self.starts_dead
        if self.start_error is not None:
            if not self.stays_alive_on_error:
                self._alive = False
            raise self.start_error

    def join(self, timeout=None):
        self.join_count += 1
        if not self.join_blocks:
            self._alive = False

    def is_alive(self) -> bool:
        return self._alive

    def kill(self):
        self._alive = False


def make_camera(**kwargs):
    factory = kwargs.pop("capture_factory", None) or FakeCaptureFactory()
    return CameraCapture(
        capture_factory=factory,
        thread_factory=FakeThread,
        **kwargs,
    ), factory


class CameraCaptureLifecycleTest(unittest.TestCase):
    def setUp(self):
        self._orig_has_cv2 = camera_mod.HAS_CV2
        camera_mod.HAS_CV2 = True

    def tearDown(self):
        camera_mod.HAS_CV2 = self._orig_has_cv2

    def test_initial_state_is_stopped(self):
        cam, _ = make_camera()
        self.assertEqual(cam.status, STOPPED)
        self.assertFalse(cam.is_running)
        self.assertFalse(cam.is_live)
        self.assertFalse(cam.stop_pending)
        self.assertIsNone(cam._thread)

    def test_start_opens_and_starts_thread(self):
        cam, factory = make_camera()
        self.assertTrue(cam.start())
        self.assertEqual(len(factory.instances), 1)
        self.assertTrue(factory.instances[0].isOpened())
        self.assertIsNotNone(cam._thread)
        self.assertTrue(cam._thread.is_alive())
        self.assertEqual(cam.status, RUNNING)
        self.assertTrue(cam.is_running)
        self.assertTrue(cam.is_live)
        self.assertFalse(cam.stop_pending)

    def test_start_fails_when_camera_unavailable(self):
        cam, factory = make_camera()
        factory.available = False
        self.assertFalse(cam.start())
        self.assertEqual(cam.status, STOPPED)
        self.assertFalse(cam.is_running)
        self.assertIsNone(cam._thread)

    def test_stop_releases_ownership_when_thread_dies(self):
        cam, factory = make_camera()
        self.assertTrue(cam.start())
        thread = cam._thread
        cam.stop()
        self.assertIsNone(cam._thread)
        self.assertEqual(cam.status, STOPPED)
        self.assertFalse(cam.is_running)
        self.assertFalse(cam.stop_pending)
        self.assertFalse(thread.is_alive())
        self.assertTrue(factory.instances[0].released)

    def test_stop_retains_live_thread_on_join_timeout(self):
        cam, factory = make_camera()
        self.assertTrue(cam.start())
        thread = cam._thread
        thread.join_blocks = True
        cam.stop()
        self.assertIs(cam._thread, thread)
        self.assertEqual(cam.status, STOP_PENDING)
        self.assertFalse(cam.is_running)
        self.assertTrue(cam.stop_pending)
        self.assertTrue(cam.is_live)
        self.assertTrue(thread.is_alive())
        self.assertTrue(factory.instances[0].released)

    def test_start_blocks_duplicate_while_thread_alive(self):
        cam, factory = make_camera()
        self.assertTrue(cam.start())
        first = cam._thread
        first.join_blocks = True
        cam.stop()
        self.assertTrue(first.is_alive())
        self.assertFalse(cam.start())
        self.assertIs(cam._thread, first)
        self.assertEqual(first.start_count, 1)
        self.assertEqual(len(factory.instances), 1)

    def test_restart_allowed_after_confirmed_death(self):
        cam, factory = make_camera()
        self.assertTrue(cam.start())
        first = cam._thread
        first.join_blocks = True
        cam.stop()
        self.assertTrue(cam.stop_pending)
        first.kill()
        self.assertFalse(first.is_alive())
        self.assertTrue(cam.start())
        self.assertIsNot(cam._thread, first)
        self.assertFalse(cam.stop_pending)
        self.assertTrue(cam.is_running)
        self.assertTrue(cam._thread.is_alive())
        self.assertEqual(len(factory.instances), 2)

    def test_second_stop_releases_ownership_after_death(self):
        cam, _ = make_camera()
        self.assertTrue(cam.start())
        thread = cam._thread
        thread.join_blocks = True
        cam.stop()
        self.assertTrue(cam.stop_pending)
        thread.kill()
        cam.stop()
        self.assertIsNone(cam._thread)
        self.assertFalse(cam.stop_pending)
        self.assertFalse(cam.is_running)
        self.assertEqual(cam.status, STOPPED)

    def test_stop_idempotent_without_thread(self):
        cam, _ = make_camera()
        cam.stop()
        cam.stop()
        self.assertFalse(cam.is_running)
        self.assertIsNone(cam._thread)
        self.assertFalse(cam.stop_pending)
        self.assertEqual(cam.status, STOPPED)

    def test_repeated_stop_idempotent_while_blocked(self):
        cam, _ = make_camera()
        self.assertTrue(cam.start())
        thread = cam._thread
        thread.join_blocks = True
        cam.stop()
        cam.stop()
        self.assertIs(cam._thread, thread)
        self.assertTrue(cam.stop_pending)
        self.assertEqual(thread.join_count, 2)

    def test_start_returns_false_when_thread_start_raises(self):
        cam, factory = make_camera()

        class _ThreadFactory:
            def __call__(self, target=None, daemon=True):
                t = FakeThread(target=target, daemon=daemon)
                t.start_error = RuntimeError("start boom")
                return t

        cam._thread_factory = _ThreadFactory()
        self.assertFalse(cam.start())
        self.assertIsNone(cam._thread)
        self.assertFalse(cam.is_running)
        self.assertTrue(factory.instances[0].released)

    def test_start_returns_true_retains_running_when_start_raises_after_marking_live(self):
        cam, factory = make_camera()
        calls = {"n": 0}

        def _LiveThreadFactory(target=None, daemon=True):
            calls["n"] += 1
            t = FakeThread(target=target, daemon=daemon)
            if calls["n"] == 1:
                t.start_error = RuntimeError("start boom")
                t.stays_alive_on_error = True
            return t

        cam._thread_factory = _LiveThreadFactory
        self.assertTrue(cam.start())
        # start() 例外だがスレッドは生存: 起動成功扱い (requested-running) を保持する
        retained = cam._thread
        self.assertIsNotNone(retained)
        self.assertTrue(retained.is_alive())
        self.assertTrue(cam.is_live)
        self.assertTrue(cam.is_running)
        self.assertFalse(cam.stop_pending)
        self.assertFalse(factory.instances[0].released)
        # 生存スレッドの所有権があるため重複 start はブロックされる
        self.assertFalse(cam.start())
        self.assertIs(cam._thread, retained)
        # stop() が保留スレッドを後始末して再起動可能にする
        cam.stop()
        self.assertIsNone(cam._thread)
        self.assertFalse(cam.stop_pending)
        self.assertFalse(cam.is_live)
        self.assertTrue(cam.start())
        self.assertIsNotNone(cam._thread)
        self.assertTrue(cam.is_running)

    def test_start_clears_ownership_when_start_raises_and_thread_dead(self):
        cam, factory = make_camera()
        calls = {"n": 0}

        def _ThreadFactory(target=None, daemon=True):
            calls["n"] += 1
            t = FakeThread(target=target, daemon=daemon)
            if calls["n"] == 1:
                t.start_error = RuntimeError("start boom")
            return t

        cam._thread_factory = _ThreadFactory
        self.assertFalse(cam.start())
        # 未起動 / 死亡済みなら参照を解放する
        self.assertIsNone(cam._thread)
        self.assertFalse(cam.is_live)
        self.assertFalse(cam.is_running)
        self.assertFalse(cam.stop_pending)
        self.assertTrue(factory.instances[0].released)
        # 後続 start は新しいスレッドで再試行できる
        self.assertTrue(cam.start())
        self.assertIsNotNone(cam._thread)
        self.assertTrue(cam.is_running)

    def test_unexpected_thread_death_marks_stopped(self):
        cam, _ = make_camera()
        self.assertTrue(cam.start())
        thread = cam._thread
        thread.kill()  # stop() を呼ばずに worker が予期せず死亡
        self.assertFalse(cam.is_running)
        self.assertFalse(cam.is_live)
        self.assertEqual(cam.status, STOPPED)
        # 死亡確認済みなので再起動できる
        self.assertTrue(cam.start())
        self.assertIsNot(cam._thread, thread)
        self.assertTrue(cam.is_running)

    def test_capture_released_before_join_unblocks_thread(self):
        cam, _ = make_camera()
        self.assertTrue(cam.start())
        thread = cam._thread
        # カメラ解放でスレッドが死ぬ (release が read ブロックを解除する想定)
        cam._cap.thread = thread
        cam._cap.release = lambda: (setattr(cam._cap, "_opened", False), thread.kill())
        thread.join_blocks = True
        cam.stop()
        # 解放が join より先に走ったためスレッドは死んでおり、所有権が解放される
        self.assertIsNone(cam._thread)
        self.assertFalse(cam.stop_pending)
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
