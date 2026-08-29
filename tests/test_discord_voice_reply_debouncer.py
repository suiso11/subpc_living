from __future__ import annotations

import asyncio
import unittest

from src.discord_bot.voice_reply_debouncer import (
    VoiceReplyDebouncer,
    VoiceReplyGenerationGate,
)


class _FakeHandle:
    def __init__(self, loop: "_FakeLoop", when: float, callback, args) -> None:
        self._loop = loop
        self.when = when
        self.callback = callback
        self.args = args
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class _FakeLoop:
    """Deterministic stand-in for an asyncio loop's time()/call_later()."""

    def __init__(self) -> None:
        self._now = 0.0
        self._handles: list[_FakeHandle] = []

    def time(self) -> float:
        return self._now

    def call_later(self, delay: float, callback, *args) -> _FakeHandle:
        handle = _FakeHandle(self, self._now + delay, callback, args)
        self._handles.append(handle)
        return handle

    def advance(self, seconds: float) -> None:
        """Advance the clock, firing any due (non-cancelled) timers in order."""
        target = self._now + seconds
        while True:
            due = [
                h
                for h in self._handles
                if not h.cancelled and h.when <= target + 1e-9
            ]
            if not due:
                break
            due.sort(key=lambda h: h.when)
            handle = due[0]
            self._handles.remove(handle)
            self._now = max(self._now, handle.when)
            handle.callback(*handle.args)
        self._now = target


class VoiceReplyDebouncerTest(unittest.TestCase):
    def _make(self, *, debounce_ms=3000, max_ms=10000):
        loop = _FakeLoop()
        flushed: list[tuple[object, str]] = []
        debouncer = VoiceReplyDebouncer(
            debounce_ms=debounce_ms,
            max_ms=max_ms,
            on_flush=lambda msg, text: flushed.append((msg, text)),
            loop=loop,
        )
        return loop, debouncer, flushed

    def test_merges_fragments_and_fires_once(self) -> None:
        loop, debouncer, flushed = self._make(debounce_ms=3000, max_ms=10000)

        debouncer.submit("alice", "m1", "文脈の読み飛ばしっていうよりも")
        loop.advance(1.0)
        debouncer.submit("alice", "m2", "文脈が途切れてるのか")
        loop.advance(1.0)
        debouncer.submit("alice", "m3", "続いてるのかな、判断が。")

        # Not enough silence yet -> no reply.
        self.assertEqual(flushed, [])

        loop.advance(3.0)
        self.assertEqual(len(flushed), 1)
        msg, text = flushed[0]
        # Latest fragment's message is used for the reply.
        self.assertEqual(msg, "m3")
        self.assertEqual(
            text,
            "文脈の読み飛ばしっていうよりも文脈が途切れてるのか続いてるのかな、判断が。",
        )

    def test_timer_resets_on_each_fragment(self) -> None:
        loop, debouncer, flushed = self._make(debounce_ms=3000, max_ms=10000)

        debouncer.submit("alice", "m1", "あー")
        loop.advance(2.0)  # < debounce, no fire
        self.assertEqual(flushed, [])
        debouncer.submit("alice", "m2", "えーと")
        loop.advance(2.0)  # timer was reset, still < debounce since last fragment
        self.assertEqual(flushed, [])
        debouncer.submit("alice", "m3", "そうだね")
        loop.advance(2.999)
        self.assertEqual(flushed, [])
        loop.advance(0.001)
        self.assertEqual(len(flushed), 1)
        self.assertEqual(flushed[0][1], "あーえーとそうだね")

    def test_max_ms_forces_fire_while_still_talking(self) -> None:
        loop, debouncer, flushed = self._make(debounce_ms=3000, max_ms=10000)

        # A fragment every 2s keeps resetting the 3s debounce, but max_ms=10s
        # must force a flush at 10s from the first fragment.
        debouncer.submit("alice", "m1", "1")
        for i in range(2, 8):
            loop.advance(2.0)
            debouncer.submit("alice", f"m{i}", str(i))
            if loop.time() < 10.0:
                self.assertEqual(flushed, [], f"fired too early at t={loop.time()}")

        # By t=10s the max window elapses.
        loop.advance(0.5)
        self.assertEqual(len(flushed), 1)
        # All fragments submitted up to and including the max deadline are merged.
        self.assertTrue(flushed[0][1].startswith("1"))

    def test_disabled_debounce_fires_immediately_per_fragment(self) -> None:
        loop, debouncer, flushed = self._make(debounce_ms=0, max_ms=10000)
        self.assertFalse(debouncer.enabled)

        debouncer.submit("alice", "m1", "断片1")
        debouncer.submit("alice", "m2", "断片2")

        # No clock advance needed: each fragment replies right away, separately.
        self.assertEqual(flushed, [("m1", "断片1"), ("m2", "断片2")])

    def test_separate_speakers_buffer_independently(self) -> None:
        loop, debouncer, flushed = self._make(debounce_ms=3000, max_ms=10000)

        debouncer.submit("alice", "a1", "アリス")
        loop.advance(1.0)
        debouncer.submit("bob", "b1", "ボブ")
        loop.advance(3.0)

        # Both fire once each, independently merged.
        self.assertEqual(len(flushed), 2)
        by_text = {text for _msg, text in flushed}
        self.assertEqual(by_text, {"アリス", "ボブ"})

    def test_flush_all_fires_pending(self) -> None:
        loop, debouncer, flushed = self._make(debounce_ms=3000, max_ms=10000)
        debouncer.submit("alice", "m1", "途中")
        self.assertEqual(flushed, [])
        debouncer.flush_all()
        self.assertEqual(flushed, [("m1", "途中")])
        # Buffer cleared; a stray timer fire must not double-flush.
        loop.advance(5.0)
        self.assertEqual(len(flushed), 1)


class _FakeTask:
    """Minimal deterministic stand-in for an asyncio.Task used by the gate.

    Supports ``done()`` / ``cancel()`` / ``add_done_callback()`` only, so the gate's
    sync revoke and task-tracking can be tested without any event loop.
    """

    def __init__(self, done: bool = False) -> None:
        self._done = done
        self.cancel_calls = 0
        self._done_callbacks: list = []

    def done(self) -> bool:
        return self._done

    def cancel(self) -> None:
        self.cancel_calls += 1

    def add_done_callback(self, callback) -> None:
        self._done_callbacks.append(callback)

    def finish(self) -> None:
        self._done = True
        for callback in self._done_callbacks:
            callback(self)


class VoiceReplyGenerationGateTest(unittest.TestCase):
    """VoiceReplyGenerationGate の世代管理・revoke・タスク追跡を検証する。"""

    def test_activate_returns_incrementing_generation(self) -> None:
        gate = VoiceReplyGenerationGate()
        self.assertFalse(gate.active)
        self.assertEqual(gate.generation, 0)
        gen1 = gate.activate()
        self.assertEqual(gen1, 1)
        self.assertTrue(gate.active)
        gen2 = gate.activate()
        self.assertEqual(gen2, 2)

    def test_is_active_requires_same_generation(self) -> None:
        gate = VoiceReplyGenerationGate()
        gen = gate.activate()
        self.assertTrue(gate.is_active(gen))
        self.assertFalse(gate.is_active(gen - 1))
        self.assertFalse(gate.is_active(gen + 1))
        self.assertFalse(gate.is_active(0))

    def test_reactivate_invalidates_old_generation(self) -> None:
        gate = VoiceReplyGenerationGate()
        gen1 = gate.activate()
        gen2 = gate.activate()
        self.assertFalse(gate.is_active(gen1))
        self.assertTrue(gate.is_active(gen2))

    def test_revoke_deactivates_and_increments_generation(self) -> None:
        async def scenario() -> None:
            gate = VoiceReplyGenerationGate()
            gen = gate.activate()
            await gate.revoke()
            self.assertFalse(gate.active)
            self.assertEqual(gate.generation, gen + 1)
            self.assertFalse(gate.is_active(gen))

        asyncio.run(scenario())

    def test_revoke_with_no_tracked_tasks_returns_immediately(self) -> None:
        async def scenario() -> None:
            gate = VoiceReplyGenerationGate()
            gen = gate.activate()
            await gate.revoke()
            self.assertFalse(gate.active)
            self.assertEqual(gate.generation, gen + 1)

        asyncio.run(scenario())

    def test_revoke_cancels_and_awaits_multiple_tracked_tasks(self) -> None:
        async def scenario() -> None:
            gate = VoiceReplyGenerationGate()
            gen = gate.activate()
            started = asyncio.Event()
            release = asyncio.Event()

            async def blocker() -> None:
                started.set()
                await release.wait()

            # 「バリア」: 両方の返信タスクが実行中になるのを待ってから revoke する。
            task_a = asyncio.create_task(blocker())
            task_b = asyncio.create_task(blocker())
            await started.wait()
            gate.track(task_a)
            gate.track(task_b)
            await gate.revoke()
            self.assertFalse(gate.active)
            self.assertNotEqual(gate.generation, gen)
            self.assertTrue(task_a.cancelled())
            self.assertTrue(task_b.cancelled())

        asyncio.run(scenario())

    def test_revoke_is_bounded_when_task_ignores_cancellation(self) -> None:
        async def scenario() -> None:
            # timeout=0: revoke は待ちに時間上限を設け、即座に返らなければならない。
            gate = VoiceReplyGenerationGate(revoke_timeout=0)
            gen = gate.activate()
            started = asyncio.Event()

            async def stubborn() -> None:
                started.set()
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    await asyncio.sleep(0.01)

            task = asyncio.create_task(stubborn())
            await started.wait()
            gate.track(task)
            await gate.revoke()
            self.assertFalse(gate.active)
            self.assertNotEqual(gate.generation, gen)
            # キャンセルを無視しても待ちは上限内。後片付けだけ行う。
            done, _ = await asyncio.wait([task], timeout=1)
            self.assertIn(task, done)

        asyncio.run(scenario())

    def test_revoke_sync_cancels_running_tasks_skips_done(self) -> None:
        gate = VoiceReplyGenerationGate()
        gen = gate.activate()
        running = _FakeTask()
        done = _FakeTask(done=True)
        gate.track(running)
        gate.track(done)
        gate.revoke_sync()
        self.assertFalse(gate.active)
        self.assertEqual(gate.generation, gen + 1)
        self.assertEqual(running.cancel_calls, 1)
        self.assertEqual(done.cancel_calls, 0)

    def test_revoke_cancels_tracked_autoread_child(self) -> None:
        gate = VoiceReplyGenerationGate()
        gate.activate()
        autoread_task = _FakeTask()
        gate.track(autoread_task)

        gate.revoke_sync()

        self.assertEqual(autoread_task.cancel_calls, 1)

    def test_track_discards_finished_tasks_via_done_callback(self) -> None:
        gate = VoiceReplyGenerationGate()
        gate.activate()
        task = _FakeTask()
        gate.track(task)
        self.assertIn(task, gate._tasks)
        task.finish()
        self.assertNotIn(task, gate._tasks)


if __name__ == "__main__":
    unittest.main()
