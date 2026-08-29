"""
ScreenContext の start/stop ライフサイクルを決定的に検証する。

実スレッド・実キャプチャは使わず、FakeThread / FakeCapture / FakeDescriber で
join タイムアウト・スレッド死亡・重複起動防止・stop 冪等性・
固定 boolean のみの状態公開を検証する。
"""
from __future__ import annotations

import unittest

from src.screen.context import ScreenContext


class FakeCapture:
    """常に利用可能なフェイクキャプチャ。"""

    def __init__(self, available: bool = True):
        self.available = available

    def is_available(self) -> bool:
        return self.available

    def capture(self):
        return b"jpeg-bytes"


class FakeDescriber:
    """常に描写を返すフェイク描写器。"""

    def __init__(self, model: str = "fake-vlm"):
        self.model = model

    def describe(self, jpeg_bytes):
        return "ターミナルで作業しています。"


class FakeThread:
    """join の挙動 (即死 or タイムアウト相当) を決定的に制御するフェイクスレッド。

    join_blocks=True の間は join() を呼んでも生き残り、join タイムアウトを再現する。
    kill() で明示的に死亡させられる。
    """

    def __init__(self, target=None, daemon=True):
        self.target = target
        self.daemon = daemon
        self._alive = False
        self.join_blocks = False
        self.start_count = 0
        self.join_count = 0

    def start(self):
        self.start_count += 1
        self._alive = True

    def join(self, timeout=None):
        self.join_count += 1
        if not self.join_blocks:
            self._alive = False

    def is_alive(self) -> bool:
        return self._alive

    def kill(self):
        self._alive = False


class _RaisingStartThread(FakeThread):
    """start() が例外を投げるフェイクスレッド。

    raise_alive=True のときは生存状態にしてから例外を投げ (部分起動)、
    False のときは起動せずに例外を投げる。
    """

    def __init__(self, target=None, daemon=True, raise_alive: bool = False):
        super().__init__(target=target, daemon=daemon)
        self._raise_alive = raise_alive

    def start(self):
        self.start_count += 1
        if self._raise_alive:
            self._alive = True
        raise RuntimeError("secret start detail")


class _ImmediateDeathThread(FakeThread):
    """start() 直後に即死するフェイクスレッド (予期せぬスレッド死亡)。"""

    def start(self):
        super().start()
        self._alive = False


def make_ctx(**kwargs):
    capture = kwargs.pop("capture", None)
    describer = kwargs.pop("describer", None)
    return ScreenContext(
        capture=capture or FakeCapture(),
        describer=describer or FakeDescriber(),
        thread_factory=FakeThread,
        **kwargs,
    )


class ScreenContextLifecycleTest(unittest.TestCase):
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

    def test_availability_unavailable_returns_false_without_state(self):
        ctx = make_ctx(capture=FakeCapture(available=False))
        self.assertFalse(ctx.start())
        self.assertFalse(ctx.is_running)
        self.assertIsNone(ctx._thread)
        self.assertFalse(ctx._stop_pending)
        self.assertFalse(ctx._paused)

    def test_availability_exception_returns_false(self):
        class _RaisingCapture(FakeCapture):
            def is_available(self):
                raise RuntimeError("secret availability detail")

        ctx = make_ctx(capture=_RaisingCapture())
        started = ctx.start()
        self.assertFalse(started)
        self.assertFalse(ctx.is_running)
        self.assertIsNone(ctx._thread)
        self.assertFalse(ctx._stop_pending)

    def test_thread_factory_failure_cleans_partial_state(self):
        def bad_factory(target=None, daemon=True):
            raise RuntimeError("secret factory detail")

        ctx = make_ctx()
        ctx._thread_factory = bad_factory
        started = ctx.start()
        self.assertFalse(started)
        self.assertFalse(ctx.is_running)
        self.assertIsNone(ctx._thread)
        self.assertFalse(ctx._stop_pending)
        self.assertFalse(ctx._paused)

    def test_thread_start_failure_cleans_partial_state(self):
        ctx = make_ctx()
        ctx._thread_factory = lambda target=None, daemon=True: _RaisingStartThread(
            target=target, daemon=daemon
        )
        started = ctx.start()
        self.assertFalse(started)
        self.assertFalse(ctx.is_running)
        self.assertIsNone(ctx._thread)
        self.assertFalse(ctx._stop_pending)
        self.assertFalse(ctx._paused)

    def test_partial_start_returns_true_and_retains_live_ownership(self):
        ctx = make_ctx()
        ctx._thread_factory = lambda target=None, daemon=True: _RaisingStartThread(
            target=target, daemon=daemon, raise_alive=True
        )
        started = ctx.start()
        self.assertTrue(started)
        self.assertTrue(ctx.is_running)
        self.assertFalse(ctx._stop_pending)
        self.assertFalse(ctx._paused)
        thread = ctx._thread
        self.assertIsNotNone(thread)
        self.assertTrue(thread.is_alive())

        # 生存スレッドが残っている限り再起動は拒否される
        self.assertFalse(ctx.start())
        self.assertIs(ctx._thread, thread)

        # 死亡を確認できた時点で所有権が解放され再起動できる
        thread.kill()
        ctx.stop()
        self.assertIsNone(ctx._thread)
        self.assertFalse(ctx._stop_pending)
        ctx._thread_factory = FakeThread
        self.assertTrue(ctx.start())
        self.assertTrue(ctx.is_running)

    def test_immediate_death_returns_false_and_clears(self):
        ctx = make_ctx()
        ctx._thread_factory = lambda target=None, daemon=True: _ImmediateDeathThread(
            target=target, daemon=daemon
        )
        self.assertFalse(ctx.start())
        self.assertIsNone(ctx._thread)
        # 生存スレッドが無いときは「稼働中」として報告しない
        self.assertFalse(ctx.is_running)
        self.assertFalse(ctx.get_status()["running"])
        # 死亡確認後は所有権が解放され再起動できる
        ctx.stop()
        self.assertIsNone(ctx._thread)
        self.assertFalse(ctx._stop_pending)
        ctx._thread_factory = FakeThread
        self.assertTrue(ctx.start())
        self.assertTrue(ctx.is_running)

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
            "description",
            "captured_at",
            "age_seconds",
            "analysis_interval",
            "analysis_count",
            "consecutive_failures",
            "model",
            "stop_pending",
            "thread_alive",
        }
        self.assertEqual(set(status.keys()), fixed_keys)
        for key in ("running", "paused", "stop_pending", "thread_alive"):
            self.assertIsInstance(status[key], bool)
        self.assertFalse(status["running"])
        self.assertTrue(status["stop_pending"])
        self.assertTrue(status["thread_alive"])
        # 例外内容・パス等の可変文字列は含まれない
        for value in status.values():
            self.assertNotIsInstance(value, Exception)


if __name__ == "__main__":
    unittest.main()