"""
VisionContext の start/stop ライフサイクルを決定的に検証する。

実スレッド・実カメラ・実モデルは使わず、FakeCamera / FakeAnalyzer / FakeThread で
join タイムアウト・スレッド死亡・重複起動防止・stop 冪等性・
固定 boolean のみの状態公開を検証する。
"""
from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from src.vision.context import VisionContext
from src.vision.detector import VisionResult


class FakeCamera:
    """常に利用可能なフェイクカメラ。

    start_error を設定すると start() が例外を投げる。stop_effect は stop() が
    呼ばれたときに実行されるコールバック (解析スレッドのブロック解除を再現)。
    """

    def __init__(self, available: bool = True, start_error: Exception = None):
        self.available = available
        self.start_error = start_error
        self._running = False
        self.start_count = 0
        self.stop_count = 0
        self.stop_effect = None

    def start(self) -> bool:
        self.start_count += 1
        if self.start_error is not None:
            raise self.start_error
        if not self.available:
            return False
        self._running = True
        return True

    def stop(self):
        self.stop_count += 1
        self._running = False
        if self.stop_effect is not None:
            self.stop_effect()

    @property
    def is_running(self) -> bool:
        return self._running

    def get_frame(self):
        return None


class FakeAnalyzer:
    """常に空結果を返すフェイクアナライザー。"""

    def __init__(self, has_emotion: bool = False):
        self.has_emotion = has_emotion

    def analyze(self, frame) -> VisionResult:
        return VisionResult()


class FakeThread:
    """join の挙動 (即死 or タイムアウト相当) を決定的に制御するフェイクスレッド。

    join_blocks=True の間は join() を呼んでも生き残り、join タイムアウトを再現する。
    kill() で明示的に死亡させられる。start_error を設定すると start() が例外を投げ、
    stays_alive_on_error=True なら例外後も生存を保つ。starts_dead=True なら
    start() 直後に死亡する。
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


def make_ctx(**kwargs):
    camera = kwargs.pop("camera", None)
    analyzer = kwargs.pop("analyzer", None)
    thread_factory = kwargs.pop("thread_factory", None)
    return VisionContext(
        camera=camera or FakeCamera(),
        analyzer=analyzer or FakeAnalyzer(),
        thread_factory=thread_factory or FakeThread,
        **kwargs,
    )


class VisionContextLifecycleTest(unittest.TestCase):
    def test_start_fails_when_camera_unavailable(self):
        ctx = make_ctx(camera=FakeCamera(available=False))
        self.assertFalse(ctx.start())
        self.assertFalse(ctx.is_running)
        self.assertIsNone(ctx._thread)

    def test_stop_releases_ownership_when_thread_dies(self):
        ctx = make_ctx()
        self.assertTrue(ctx.start())
        thread = ctx._thread
        self.assertIsNotNone(thread)
        self.assertTrue(thread.is_alive())

        ctx.stop()

        self.assertFalse(ctx.is_running)
        self.assertIsNone(ctx._thread)
        self.assertFalse(ctx._stop_pending)
        self.assertFalse(thread.is_alive())

    def test_stop_retains_ownership_on_join_timeout(self):
        ctx = make_ctx()
        self.assertTrue(ctx.start())
        thread = ctx._thread
        thread.join_blocks = True

        ctx.stop()

        # スレッドが生き残っている間は所有権を保持し、停止保留フラグが立つ
        self.assertFalse(ctx.is_running)
        self.assertIs(ctx._thread, thread)
        self.assertTrue(ctx._stop_pending)
        self.assertTrue(thread.is_alive())

    def test_start_refuses_duplicate_while_thread_alive(self):
        ctx = make_ctx()
        self.assertTrue(ctx.start())
        first = ctx._thread
        first.join_blocks = True
        ctx.stop()
        self.assertTrue(first.is_alive())

        # 生存スレッドがある限り再起動は拒否され、新規スレッドは生成されない
        self.assertFalse(ctx.start())
        self.assertIs(ctx._thread, first)
        self.assertEqual(first.start_count, 1)

    def test_restart_allowed_after_confirmed_death(self):
        ctx = make_ctx()
        self.assertTrue(ctx.start())
        first = ctx._thread
        first.join_blocks = True
        ctx.stop()
        self.assertTrue(ctx._stop_pending)

        # 死亡が確認されたら再起動できる
        first.kill()
        self.assertFalse(first.is_alive())
        self.assertTrue(ctx.start())
        self.assertIsNot(ctx._thread, first)
        self.assertFalse(ctx._stop_pending)
        self.assertTrue(ctx.is_running)
        self.assertTrue(ctx._thread.is_alive())

    def test_second_stop_releases_ownership_after_death(self):
        ctx = make_ctx()
        self.assertTrue(ctx.start())
        thread = ctx._thread
        thread.join_blocks = True
        ctx.stop()
        self.assertTrue(ctx._stop_pending)

        # 死亡後に2回目の stop を呼ぶと所有権が解放される
        thread.kill()
        ctx.stop()
        self.assertIsNone(ctx._thread)
        self.assertFalse(ctx._stop_pending)
        self.assertFalse(ctx.is_running)

    def test_stop_idempotent_without_thread(self):
        ctx = make_ctx()
        ctx.stop()
        ctx.stop()  # スレッド無しの stop は何度でも安全
        self.assertFalse(ctx.is_running)
        self.assertIsNone(ctx._thread)
        self.assertFalse(ctx._stop_pending)

    def test_repeated_stop_idempotent_while_blocked(self):
        ctx = make_ctx()
        self.assertTrue(ctx.start())
        thread = ctx._thread
        thread.join_blocks = True

        ctx.stop()
        ctx.stop()  # ブロック中の stop を繰り返しても状態を壊さない

        self.assertIs(ctx._thread, thread)
        self.assertTrue(ctx._stop_pending)
        self.assertEqual(thread.join_count, 2)

    def test_pause_resume_preserved(self):
        ctx = make_ctx()
        self.assertTrue(ctx.start())
        self.assertFalse(ctx._paused)

        ctx.pause()
        self.assertTrue(ctx._paused)
        self.assertTrue(ctx.get_status()["paused"])

        ctx.resume()
        self.assertFalse(ctx._paused)
        self.assertFalse(ctx.get_status()["paused"])
        ctx.stop()

    def test_status_exposes_fixed_boolean_metadata_only(self):
        ctx = make_ctx()
        self.assertTrue(ctx.start())
        thread = ctx._thread
        thread.join_blocks = True
        ctx.stop()

        status = ctx.get_status()

        fixed_keys = {
            "running",
            "paused",
            "stop_pending",
            "thread_alive",
            "user_present",
            "person_count",
            "emotion",
            "emotion_ja",
            "emotion_detection",
            "analysis_interval",
            "analysis_count",
        }
        self.assertEqual(set(status.keys()), fixed_keys)
        for key in (
            "running",
            "paused",
            "stop_pending",
            "thread_alive",
            "user_present",
            "emotion_detection",
        ):
            self.assertIsInstance(status[key], bool)
        self.assertFalse(status["running"])
        self.assertTrue(status["stop_pending"])
        self.assertTrue(status["thread_alive"])
        # 例外内容・パス等の可変文字列は含まれない
        for value in status.values():
            self.assertNotIsInstance(value, Exception)

    def test_start_returns_false_when_camera_start_raises(self):
        ctx = make_ctx(camera=FakeCamera(start_error=RuntimeError("camera boom")))
        self.assertFalse(ctx.start())
        self.assertIsNone(ctx._thread)
        self.assertFalse(ctx.is_running)
        # カメラ起動に失敗してもベストエフォートで後始末を試みる
        self.assertEqual(ctx.camera.stop_count, 1)

    def test_start_returns_false_when_thread_factory_raises(self):
        def boom(*args, **kwargs):
            raise RuntimeError("factory boom")

        ctx = make_ctx(thread_factory=boom)
        self.assertFalse(ctx.start())
        self.assertIsNone(ctx._thread)
        self.assertFalse(ctx.is_running)
        self.assertEqual(ctx.camera.stop_count, 1)

    def test_start_returns_false_when_thread_start_raises(self):
        class _Factory:
            def __call__(self, target=None, daemon=True):
                t = FakeThread(target=target, daemon=daemon)
                t.start_error = RuntimeError("start boom")
                return t

        ctx = make_ctx(thread_factory=_Factory())
        self.assertFalse(ctx.start())
        self.assertIsNone(ctx._thread)
        self.assertFalse(ctx.is_running)
        self.assertEqual(ctx.camera.stop_count, 1)

    def test_start_returns_false_when_thread_dies_immediately(self):
        class _Factory:
            def __call__(self, target=None, daemon=True):
                t = FakeThread(target=target, daemon=daemon)
                t.starts_dead = True
                return t

        ctx = make_ctx(thread_factory=_Factory())
        self.assertFalse(ctx.start())
        self.assertIsNone(ctx._thread)
        self.assertFalse(ctx.is_running)
        self.assertEqual(ctx.camera.stop_count, 1)

    def test_start_returns_true_when_thread_live_after_start_failure(self):
        class _Factory:
            def __init__(self):
                self.calls = 0

            def __call__(self, target=None, daemon=True):
                self.calls += 1
                t = FakeThread(target=target, daemon=daemon)
                if self.calls == 1:
                    t.start_error = RuntimeError("late start boom")
                    t.stays_alive_on_error = True
                return t

        ctx = make_ctx(thread_factory=_Factory())
        self.assertTrue(ctx.start())
        thread = ctx._thread
        self.assertIsNotNone(thread)
        self.assertTrue(thread.is_alive())
        self.assertTrue(ctx.is_running)
        self.assertFalse(ctx._stop_pending)
        # 生存中の解析スレッドに対してカメラを落とさない (ネスト整合性)
        self.assertEqual(ctx.camera.stop_count, 0)
        self.assertTrue(ctx.camera.is_running)
        # 実際に生存したスレッドがある限り再起動は拒否される
        self.assertFalse(ctx.start())
        self.assertIs(ctx._thread, thread)
        # 死亡確認後は再起動できる
        thread.kill()
        ctx.stop()
        self.assertIsNone(ctx._thread)
        self.assertFalse(ctx._stop_pending)
        self.assertTrue(ctx.start())
        self.assertIsNot(ctx._thread, thread)
        self.assertTrue(ctx.is_running)

    def test_unexpected_worker_death_makes_is_running_false(self):
        ctx = make_ctx()
        self.assertTrue(ctx.start())
        thread = ctx._thread
        thread.kill()  # stop() を呼ばずに解析スレッドが予期せず死亡
        self.assertFalse(ctx.is_running)
        # 死亡確認済みなので再起動できる
        self.assertTrue(ctx.start())
        self.assertIsNot(ctx._thread, thread)
        self.assertTrue(ctx.is_running)

    def test_stop_signals_camera_before_join_unblocks_thread(self):
        ctx = make_ctx()
        self.assertTrue(ctx.start())
        thread = ctx._thread
        # カメラ停止が解析スレッドのブロックを解除する想定 (kill で再現)
        ctx.camera.stop_effect = thread.kill
        thread.join_blocks = True
        ctx.stop()
        # カメラ停止が join より先に走ったためスレッドは死んでおり、所有権が解放される
        self.assertIsNone(ctx._thread)
        self.assertFalse(ctx._stop_pending)
        self.assertFalse(thread.is_alive())
        self.assertFalse(ctx.is_running)

    # --- 解析ループの finally 後始末 (canary) ---

    def test_loop_marks_running_false_and_stops_camera_on_get_frame_error(self):
        class BoomCamera(FakeCamera):
            def get_frame(self):
                raise RuntimeError("frame boom")

        camera = BoomCamera()
        ctx = make_ctx(camera=camera)
        with patch("src.vision.context.time.sleep"):
            ctx._running = True
            ctx._analysis_loop()
        self.assertFalse(ctx._running)
        self.assertEqual(camera.stop_count, 1)
        self.assertFalse(camera.is_running)

    def test_loop_marks_running_false_and_stops_camera_on_sleep_error(self):
        camera = FakeCamera()
        ctx = make_ctx(camera=camera)
        with patch(
            "src.vision.context.time.sleep",
            side_effect=RuntimeError("sleep boom"),
        ):
            ctx._running = True
            ctx._analysis_loop()
        self.assertFalse(ctx._running)
        self.assertEqual(camera.stop_count, 1)

    def test_loop_continues_silently_after_analyzer_errors(self):
        calls = {"analyze": 0}

        class FramesCamera(FakeCamera):
            def __init__(self):
                super().__init__()
                self.count = 0

            def get_frame(self):
                self.count += 1
                if self.count <= 2:
                    return "frame"
                raise RuntimeError("frame boom")

        class BoomAnalyzer(FakeAnalyzer):
            def analyze(self, frame):
                calls["analyze"] += 1
                raise RuntimeError("analyzer boom")

        camera = FramesCamera()
        ctx = make_ctx(camera=camera, analyzer=BoomAnalyzer())
        with patch("src.vision.context.time.sleep"):
            ctx._running = True
            ctx._analysis_loop()
        # アナライザーエラーではループは止まらず2回解析する
        self.assertEqual(calls["analyze"], 2)
        # 想定外の get_frame 例外でのみ致命的後始末が走る
        self.assertFalse(ctx._running)
        self.assertEqual(camera.stop_count, 1)

    # --- 想定外 worker 死亡の後始末 ---

    def test_unexpected_worker_death_cleans_up_camera_and_allows_restart(self):
        class BoomCamera(FakeCamera):
            def get_frame(self):
                raise RuntimeError("frame boom")

        camera = BoomCamera()
        ctx = make_ctx(camera=camera, thread_factory=threading.Thread)
        self.assertTrue(ctx.start())
        thread = ctx._thread
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(ctx.is_running)
        self.assertFalse(camera.is_running)
        self.assertEqual(camera.stop_count, 1)
        # 死亡確認後はカメラを新規起動して再起動できる
        self.assertTrue(ctx.start())
        self.assertEqual(camera.start_count, 2)
        ctx.stop()

    def test_restart_cannot_overwrite_live_capture_while_loop_alive(self):
        class BlockingCamera(FakeCamera):
            def __init__(self):
                super().__init__()
                self.block = threading.Event()
                self.entered = threading.Event()

            def get_frame(self):
                self.entered.set()
                self.block.wait(10)
                return None

        camera = BlockingCamera()
        ctx = make_ctx(camera=camera, thread_factory=threading.Thread)
        self.assertTrue(ctx.start())
        first = ctx._thread
        # ループスレッドが get_frame でブロックするのを待つ
        self.assertTrue(camera.entered.wait(5))
        # ブロック中は再起動しても新しい capture を作らない
        self.assertFalse(ctx.start())
        self.assertEqual(camera.start_count, 1)
        self.assertIs(ctx._thread, first)
        # 停止しても生存中は所有権を保持する
        ctx.stop()
        self.assertTrue(first.is_alive())
        self.assertTrue(ctx._stop_pending)
        # スレッドが死んだ時点で finally の後始末が走る
        camera.block.set()
        first.join(timeout=5)
        self.assertFalse(first.is_alive())
        self.assertGreaterEqual(camera.stop_count, 1)


if __name__ == "__main__":
    unittest.main()