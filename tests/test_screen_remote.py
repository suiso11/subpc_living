"""
remote モード (RemoteScreenContext) と ingest エンドポイントの検証。

- RemoteScreenContext: 鮮度判定 / ファイル無し / 壊れた JSON
- POST /api/screen/ingest: 認証 (未設定403・不一致403・一致200) /
  マジックバイト検証 / サイズ上限
- ファクトリ create_screen_context のモード切替
- scripts/screen_agent.py の純粋ロジック (should_send / image_hash)
- ingest 描写の単一飛行 / 世代ゲート (fake loop・fake future・blocking describer)

実 Ollama は呼ばない (describer をモック)。
"""
from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stdout
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.perception import SensorPolicy
from src.screen.remote import RemoteScreenContext
from src.screen import create_screen_context
from src.screen.context import ScreenContext


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


# --------------------------- RemoteScreenContext ---------------------------

class RemoteScreenContextTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.latest = Path(self.tmp.name) / "latest.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _ctx(self, screen_ingest=True, **kwargs) -> RemoteScreenContext:
        # 明示 true を既定にし、env 非依存のテスト注入にする
        return RemoteScreenContext(
            latest_path=self.latest, screen_ingest=screen_ingest, **kwargs
        )

    def test_context_text_empty_when_not_running(self):
        _write_json(self.latest, {"description": "作業中", "captured_at": time.time()})
        ctx = self._ctx()
        # start していない (_running=False) → 空
        self.assertEqual(ctx.get_context_text(), "")

    def test_fresh_description_rendered(self):
        _write_json(self.latest, {
            "description": "VSCodeでコードを書いています。",
            "captured_at": time.time(),
            "described_at": time.time(),
            "source": "remote",
        })
        ctx = self._ctx(poll_interval=0.01)
        self.assertTrue(ctx.start())
        text = ctx.get_context_text()
        self.assertIn("画面情報", text)
        self.assertIn("メインPC", text)
        self.assertIn("VSCodeでコードを書いています。", text)
        self.assertTrue(text.startswith("\n"))
        ctx.stop()

    def test_missing_file_gives_empty(self):
        ctx = self._ctx()
        ctx._running = True
        # ファイルが存在しない
        self.assertFalse(ctx._read_once())
        self.assertEqual(ctx.get_context_text(), "")

    def test_broken_json_keeps_empty(self):
        self.latest.write_text("{ this is not valid json ", encoding="utf-8")
        ctx = self._ctx()
        ctx._running = True
        self.assertFalse(ctx._read_once())
        self.assertEqual(ctx.get_context_text(), "")

    def test_stale_description_gives_empty(self):
        _write_json(self.latest, {
            "description": "古い作業",
            "captured_at": time.time() - 11 * 60,  # 11分前
        })
        ctx = self._ctx(stale_after=600.0)
        ctx._running = True
        ctx._read_once()
        self.assertEqual(ctx.get_context_text(), "")

    def test_within_stale_window_rendered(self):
        _write_json(self.latest, {
            "description": "まだ有効",
            "captured_at": time.time() - 9 * 60,  # 9分前
        })
        ctx = self._ctx(stale_after=600.0, poll_interval=0.01)
        self.assertTrue(ctx.start())
        text = ctx.get_context_text()
        self.assertIn("画面情報", text)
        self.assertIn("(9分前時点)", text)
        ctx.stop()

    def test_get_status_fields(self):
        _write_json(self.latest, {"description": "x", "captured_at": time.time()})
        ctx = self._ctx(poll_interval=0.01)
        self.assertTrue(ctx.start())
        status = ctx.get_status()
        self.assertEqual(status["mode"], "remote")
        self.assertTrue(status["running"])
        self.assertEqual(status["source"], "remote")
        self.assertIsNotNone(status["age_seconds"])
        ctx.stop()

    def test_source_always_remote_regardless_of_latest_json_canary(self):
        # canary: latest.json に改ざんされた source が入っていても、
        # RemoteScreenContext は常に固定 "remote" を返す。
        _write_json(self.latest, {
            "description": "作業中",
            "captured_at": time.time(),
            "described_at": time.time(),
            "source": "evil-tampered-source",
        })
        ctx = self._ctx(poll_interval=0.01)
        self.assertTrue(ctx.start())
        self.assertEqual(ctx.get_state().source, "remote")
        self.assertEqual(ctx.get_status()["source"], "remote")
        ctx.stop()


# --------------------------- ライフサイクル (fail safe) ---------------------------

class _FakeThread:
    """実スレッドを立てずにライフサイクルを決定的に検証するためのフェイク。
    is_alive は明示制御、join は呼び出し回数を記録するだけ。"""

    def __init__(self, alive: bool = True):
        self.alive = alive
        self.join_calls = 0

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout=None):
        self.join_calls += 1


class RemoteScreenLifecycleTest(unittest.TestCase):
    """RemoteScreenContext の fail-safe ライフサイクルをフェイクスレッドで決定的に検証。

    - start: 生存中の読取スレッドがあると拒否 (False) し、既存スレッドを上書きしない
    - stop: join タイムアウト時は ownership を保持し、多重呼び出しは idempotent
    - 再 start: スレッドの死が確認された後でのみ許可
    - is_running: 実スレッドの生存を真実として報告
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.latest = Path(self.tmp.name) / "latest.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _ctx(self, **kwargs) -> RemoteScreenContext:
        return RemoteScreenContext(latest_path=self.latest, screen_ingest=True, **kwargs)

    def test_start_rejects_while_reader_thread_alive(self):
        ctx = self._ctx()
        ctx._thread = _FakeThread(alive=True)
        self.assertTrue(ctx.is_running)
        self.assertFalse(ctx.start())
        # 既存の読取スレッドを上書きしない
        self.assertIsInstance(ctx._thread, _FakeThread)

    def test_start_rejected_until_thread_death_confirmed(self):
        ctx = self._ctx(poll_interval=0.01)
        self.assertTrue(ctx.start())
        self.assertTrue(ctx.is_running)
        # stop しない限り再 start は拒否される
        self.assertFalse(ctx.start())
        ctx.stop()
        self.assertIsNone(ctx._thread)
        # 確認死後の再 start は許可される
        self.assertTrue(ctx.start())
        ctx.stop()

    def test_start_allowed_when_thread_confirmed_dead(self):
        ctx = self._ctx(poll_interval=0.01)
        fake = _FakeThread(alive=True)
        ctx._thread = fake
        fake.alive = False  # 死が確認された
        self.assertFalse(ctx.is_running)
        self.assertTrue(ctx.start())
        self.assertIsInstance(ctx._thread, threading.Thread)
        ctx.stop()

    def test_stop_retains_ownership_on_join_timeout(self):
        ctx = self._ctx()
        fake = _FakeThread(alive=True)
        ctx._thread = fake
        ctx.stop()
        # join タイムアウトで生存 → ownership を保持 (None にしない)
        self.assertIs(ctx._thread, fake)
        self.assertEqual(fake.join_calls, 1)
        # stop を要求したら is_running は即時 False (スレッドが生存していても)
        self.assertFalse(ctx.is_running)
        self.assertTrue(ctx.thread_alive)
        self.assertTrue(ctx.stop_pending)

    def test_repeated_stop_idempotent_and_retries_join(self):
        ctx = self._ctx()
        fake = _FakeThread(alive=True)
        ctx._thread = fake
        ctx.stop()
        ctx.stop()
        ctx.stop()
        self.assertIs(ctx._thread, fake)
        self.assertEqual(fake.join_calls, 3)
        # 生存中は stop_pending が継続する
        self.assertTrue(ctx.stop_pending)
        self.assertFalse(ctx.is_running)

    def test_stop_releases_only_after_confirmed_death(self):
        ctx = self._ctx()
        fake = _FakeThread(alive=True)
        ctx._thread = fake
        fake.alive = False
        ctx.stop()
        self.assertIsNone(ctx._thread)
        self.assertFalse(ctx.is_running)
        self.assertFalse(ctx.thread_alive)
        self.assertFalse(ctx.stop_pending)

    def test_stop_without_thread_is_noop(self):
        ctx = self._ctx()
        ctx.stop()
        self.assertIsNone(ctx._thread)
        self.assertFalse(ctx.is_running)

    def test_is_running_reflects_thread_liveness_truthfully(self):
        ctx = self._ctx()
        self.assertFalse(ctx.is_running)
        ctx._thread = _FakeThread(alive=True)
        self.assertTrue(ctx.is_running)
        ctx._thread = _FakeThread(alive=False)
        self.assertFalse(ctx.is_running)
        ctx._thread = None
        self.assertFalse(ctx.is_running)

    def test_is_running_false_after_stop_requested_while_thread_alive(self):
        # stop 要求後 (stop_pending) はスレッドが生存していても is_running は False
        ctx = self._ctx()
        fake = _FakeThread(alive=True)
        ctx._thread = fake
        self.assertTrue(ctx.is_running)
        ctx.stop()
        self.assertFalse(ctx.is_running)
        self.assertTrue(ctx.thread_alive)
        self.assertTrue(ctx.stop_pending)

    def test_start_rejects_stop_pending_live_thread_without_overwrite(self):
        # stop 保留中に生存する旧読取スレッドは上書きしない (start は False)
        ctx = self._ctx()
        fake = _FakeThread(alive=True)
        ctx._thread = fake
        ctx.stop()
        self.assertTrue(ctx.stop_pending)
        self.assertFalse(ctx.start())
        self.assertIs(ctx._thread, fake)
        self.assertTrue(ctx.thread_alive)
        self.assertFalse(ctx.is_running)

    def test_repeated_stop_clears_pending_after_death(self):
        # 繰り返し stop でスレッドの死を確認できたら pending を解除する
        ctx = self._ctx()
        fake = _FakeThread(alive=True)
        ctx._thread = fake
        ctx.stop()
        self.assertTrue(ctx.stop_pending)
        fake.alive = False
        ctx.stop()
        self.assertIsNone(ctx._thread)
        self.assertFalse(ctx.stop_pending)
        self.assertFalse(ctx.is_running)

    def test_start_clears_stale_pending_and_restarts_after_death(self):
        # 死を確認できた stop_pending の残骸からは再 start できる
        ctx = self._ctx(poll_interval=0.01)
        fake = _FakeThread(alive=True)
        ctx._thread = fake
        ctx.stop()
        self.assertTrue(ctx.stop_pending)
        fake.alive = False
        self.assertTrue(ctx.start())
        self.assertIsInstance(ctx._thread, threading.Thread)
        self.assertFalse(ctx.stop_pending)
        ctx.stop()

    def test_status_exposes_fixed_booleans(self):
        ctx = self._ctx()
        fake = _FakeThread(alive=True)
        ctx._thread = fake
        self.assertTrue(ctx.is_running)
        status = ctx.get_status()
        self.assertTrue(status["running"])
        self.assertTrue(status["thread_alive"])
        self.assertFalse(status["stop_pending"])
        ctx.stop()
        status = ctx.get_status()
        # stop 要求後: running は False でも thread_alive は生存の真実を報告する
        self.assertFalse(status["running"])
        self.assertTrue(status["thread_alive"])
        self.assertTrue(status["stop_pending"])


# --------------------------- partial start の硬化 ---------------------------

class _ControlledThread:
    """threading.Thread を差し替えて start の成否・生存を決定的に制御するフェイク。

    start 時に任意の例外を投げられ、is_alive を明示制御できる。join は何もしない。
    """

    def __init__(self, target=None, daemon=None):
        self.target = target
        self.daemon = daemon
        self.start_raise = None
        self.alive_after_start = True
        self.start_called = False

    def start(self):
        self.start_called = True
        if self.start_raise is not None:
            raise self.start_raise

    def is_alive(self):
        return self.alive_after_start

    def join(self, timeout=None):
        pass


class RemoteScreenPartialStartTest(unittest.TestCase):
    """start() の partial start 硬化をフェイクスレッドで決定的に検証。

    - ownership は thread.start() より前に self._thread に確定される
    - factory / start の例外は握られ、生存スレッドは保持される
    - 死んだ / 未 start / 即死のスレッドは回収され、start は False
    - 生存スレッドが保持されている間は再 start が拒否され上書きしない
    - stop は未 start で保持されたスレッドを安全に回収する
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.latest = Path(self.tmp.name) / "latest.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _ctx(self, **kwargs) -> RemoteScreenContext:
        return RemoteScreenContext(latest_path=self.latest, screen_ingest=True, **kwargs)

    def test_ownership_assigned_before_thread_start_and_alive_retained(self):
        # start() が例外を投げても、呼び出し時点で ownership が確定しており、
        # 実際に生存しているスレッドは破棄されない。
        captured = {}

        class _ObservingThread(_ControlledThread):
            def start(self):
                captured["ownership_at_start"] = ctx._thread is self
                self.start_called = True
                raise RuntimeError("start interrupted")

        with mock.patch("src.screen.remote.threading.Thread", _ObservingThread):
            ctx = self._ctx()
            self.assertTrue(ctx.start())
        self.assertTrue(captured["ownership_at_start"])
        # 例外があっても生存スレッドは ownership を保持したまま True
        self.assertIsInstance(ctx._thread, _ObservingThread)
        self.assertTrue(ctx.is_running)
        # 生存中は再 start を拒否し、上書きもしない
        self.assertFalse(ctx.start())
        self.assertIsInstance(ctx._thread, _ObservingThread)

    def test_factory_exception_returns_false_and_cleans_state(self):
        def _boom(*args, **kwargs):
            raise RuntimeError("thread factory failed")

        with mock.patch("src.screen.remote.threading.Thread", _boom):
            ctx = self._ctx()
            self.assertFalse(ctx.start())
        self.assertIsNone(ctx._thread)
        self.assertFalse(ctx.is_running)

    def test_start_exception_with_dead_thread_cleans_and_returns_false(self):
        def _factory(*args, **kwargs):
            t = _ControlledThread(*args, **kwargs)
            t.start_raise = RuntimeError("start failed")
            t.alive_after_start = False
            return t

        with mock.patch("src.screen.remote.threading.Thread", _factory):
            ctx = self._ctx()
            self.assertFalse(ctx.start())
        # 未 start のまま死んだスレッドは回収され、再 start が許可される
        self.assertIsNone(ctx._thread)
        self.assertFalse(ctx.is_running)

    def test_immediate_death_returns_false_and_cleans(self):
        class _DeadOnArrival(_ControlledThread):
            def is_alive(self):
                return False

        with mock.patch("src.screen.remote.threading.Thread", _DeadOnArrival):
            ctx = self._ctx()
            self.assertFalse(ctx.start())
        self.assertIsNone(ctx._thread)
        self.assertFalse(ctx.is_running)

    def test_start_success_alive_returns_true_and_retains(self):
        with mock.patch("src.screen.remote.threading.Thread", _ControlledThread):
            ctx = self._ctx()
            self.assertTrue(ctx.start())
        self.assertIsInstance(ctx._thread, _ControlledThread)
        self.assertTrue(ctx.is_running)

    def test_stop_reaps_never_started_retained_thread(self):
        class _NeverStarted:
            def is_alive(self):
                return False

            def join(self, timeout=None):
                raise RuntimeError("cannot join thread before it is started")

        ctx = self._ctx()
        never = _NeverStarted()
        ctx._thread = never
        ctx.stop()  # 例外を漏らさず回収
        self.assertIsNone(ctx._thread)
        self.assertFalse(ctx.is_running)

    def test_repeated_stop_after_reaping_never_started_is_noop(self):
        class _NeverStarted:
            def is_alive(self):
                return False

            def join(self, timeout=None):
                raise RuntimeError("cannot join thread before it is started")

        ctx = self._ctx()
        ctx._thread = _NeverStarted()
        ctx.stop()
        ctx.stop()
        ctx.stop()
        self.assertIsNone(ctx._thread)
        self.assertFalse(ctx.is_running)


# --------------------------- screen_ingest ゲート ---------------------------

class RemoteScreenIngestGateTest(unittest.TestCase):
    """RemoteScreenContext は共有 SensorPolicy.screen_ingest が明示 true のときだけ
    latest.json を読取・公開する。false / 未設定 / 不正値は fail closed で、既存の
    legacy 描写は get_state / get_status / get_context_text のどこからも出ない。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.latest = Path(self.tmp.name) / "latest.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _disabled(self, **kwargs) -> RemoteScreenContext:
        return RemoteScreenContext(latest_path=self.latest, screen_ingest=False, **kwargs)

    def test_disabled_hides_fresh_legacy_description(self):
        _write_json(self.latest, {
            "description": "レガシー画面描写",
            "captured_at": time.time(),
            "described_at": time.time(),
            "source": "remote",
        })
        ctx = self._disabled()
        self.assertFalse(ctx.start())
        self.assertFalse(ctx.is_running)
        self.assertIsNone(ctx._thread)
        self.assertEqual(ctx.get_context_text(), "")
        self.assertEqual(ctx.get_state().description, "")
        status = ctx.get_status()
        self.assertFalse(status["running"])
        self.assertFalse(status["screen_ingest"])
        self.assertEqual(status["description"], "")
        self.assertFalse(ctx._read_once())
        self.assertEqual(ctx.get_state().read_count, 0)

    def test_stale_legacy_description_hidden_when_disabled(self):
        _write_json(self.latest, {
            "description": "古いレガシー描写",
            "captured_at": time.time() - 99999,
        })
        ctx = self._disabled()
        self.assertFalse(ctx.start())
        self.assertEqual(ctx.get_context_text(), "")
        self.assertEqual(ctx.get_state().captured_at, 0.0)
        self.assertEqual(ctx.get_status()["description"], "")

    def test_disabled_never_parses_file(self):
        _write_json(self.latest, {"description": "x", "captured_at": time.time()})
        ctx = self._disabled()
        with mock.patch(
            "src.screen.remote.json.loads",
            side_effect=AssertionError("must not parse latest.json"),
        ) as loads:
            self.assertFalse(ctx._read_once())
            self.assertFalse(ctx.start())
        loads.assert_not_called()

    def test_disabled_starts_no_thread_and_skips_read(self):
        ctx = self._disabled()
        with mock.patch.object(
            RemoteScreenContext,
            "_read_once",
            side_effect=AssertionError("must not read latest.json"),
        ) as read_once:
            self.assertFalse(ctx.start())
        read_once.assert_not_called()
        self.assertIsNone(ctx._thread)
        self.assertFalse(ctx.is_running)

    def test_disabled_never_constructs_thread(self):
        def _boom(*args, **kwargs):
            raise AssertionError("must not construct thread")

        with mock.patch("src.screen.remote.threading.Thread", _boom):
            ctx = self._disabled()
            self.assertFalse(ctx.start())
        self.assertIsNone(ctx._thread)
        self.assertFalse(ctx.is_running)

    def test_env_default_disables_ingest(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            ctx = RemoteScreenContext(latest_path=self.latest)
        self.assertFalse(ctx.screen_ingest)

    def test_env_canonical_true_enables_ingest(self):
        with mock.patch.dict(
            os.environ, {"SENSOR_SCREEN_INGEST_ENABLED": "true"}, clear=True
        ):
            ctx = RemoteScreenContext(latest_path=self.latest)
        self.assertTrue(ctx.screen_ingest)
        _write_json(self.latest, {"description": "d", "captured_at": time.time()})
        ctx._running = True
        self.assertTrue(ctx._read_once())

    def test_env_canonical_non_true_disables_ingest(self):
        for value in ("", "false", "0", "1", "yes", "on", "no"):
            with self.subTest(value=value):
                with mock.patch.dict(
                    os.environ, {"SENSOR_SCREEN_INGEST_ENABLED": value}, clear=True
                ):
                    ctx = RemoteScreenContext(latest_path=self.latest)
                self.assertFalse(ctx.screen_ingest)

    def test_mode_and_legacy_env_do_not_enable_ingest(self):
        # SCREEN_CONTEXT_MODE / WEB_SCREEN_CONTEXT_ENABLED / screen_capture では
        # ingest は有効化されない (canonical 名のみ・legacy 別名なし)。
        for env in (
            {"SCREEN_CONTEXT_MODE": "remote"},
            {"SCREEN_CONTEXT_MODE": "remote", "WEB_SCREEN_CONTEXT_ENABLED": "true"},
            {"SCREEN_CONTEXT_MODE": "remote", "SENSOR_SCREEN_CAPTURE_ENABLED": "true"},
        ):
            with self.subTest(env=env):
                with mock.patch.dict(os.environ, env, clear=True):
                    ctx = RemoteScreenContext(latest_path=self.latest)
                self.assertFalse(ctx.screen_ingest)

    def test_canonical_presence_overrides_absent_legacy(self):
        for value in ("", "false", "0", "1", "yes", "on", "no"):
            with self.subTest(value=value):
                with mock.patch.dict(os.environ, {
                    "SENSOR_SCREEN_INGEST_ENABLED": value,
                    "SCREEN_CONTEXT_MODE": "remote",
                    "WEB_SCREEN_CONTEXT_ENABLED": "true",
                }, clear=True):
                    ctx = RemoteScreenContext(latest_path=self.latest)
                self.assertFalse(ctx.screen_ingest)

    def test_canonical_true_beats_any_other_env(self):
        with mock.patch.dict(os.environ, {
            "SENSOR_SCREEN_INGEST_ENABLED": "true",
            "WEB_SCREEN_CONTEXT_ENABLED": "false",
            "SCREEN_CONTEXT_MODE": "local",
        }, clear=True):
            ctx = RemoteScreenContext(latest_path=self.latest)
        self.assertTrue(ctx.screen_ingest)

    def test_explicit_true_overrides_env_false(self):
        with mock.patch.dict(
            os.environ, {"SENSOR_SCREEN_INGEST_ENABLED": "false"}, clear=True
        ):
            ctx = RemoteScreenContext(latest_path=self.latest, screen_ingest=True)
        self.assertTrue(ctx.screen_ingest)
        _write_json(self.latest, {"description": "d", "captured_at": time.time()})
        ctx._running = True
        self.assertTrue(ctx._read_once())

    def test_explicit_false_overrides_env_true(self):
        with mock.patch.dict(
            os.environ, {"SENSOR_SCREEN_INGEST_ENABLED": "true"}, clear=True
        ):
            ctx = RemoteScreenContext(latest_path=self.latest, screen_ingest=False)
        self.assertFalse(ctx.screen_ingest)

    def test_poll_loop_disabled_returns_without_reading(self):
        ctx = self._disabled()
        with mock.patch.object(
            RemoteScreenContext,
            "_read_once",
            side_effect=AssertionError("must not read"),
        ) as read_once:
            ctx._poll_loop()
        read_once.assert_not_called()


# --------------------------- ファクトリ ---------------------------

class FactoryTest(unittest.TestCase):
    def test_default_is_local(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SCREEN_CONTEXT_MODE", None)
            ctx = create_screen_context()
            self.assertIsInstance(ctx, ScreenContext)

    def test_explicit_remote(self):
        ctx = create_screen_context(mode="remote")
        self.assertIsInstance(ctx, RemoteScreenContext)

    def test_env_remote(self):
        with mock.patch.dict(os.environ, {"SCREEN_CONTEXT_MODE": "remote"}):
            ctx = create_screen_context()
            self.assertIsInstance(ctx, RemoteScreenContext)

    def test_remote_ignores_local_kwargs(self):
        # local 向け kwargs (base_url/model/analysis_interval) を渡しても remote は無視
        ctx = create_screen_context(
            mode="remote",
            analysis_interval=90.0,
            base_url="http://x:1",
            model="foo",
            stale_after=123.0,
        )
        self.assertIsInstance(ctx, RemoteScreenContext)
        self.assertEqual(ctx.stale_after, 123.0)

    def test_remote_forwards_explicit_screen_ingest_true(self):
        ctx = create_screen_context(mode="remote", screen_ingest=True)
        self.assertIsInstance(ctx, RemoteScreenContext)
        self.assertTrue(ctx.screen_ingest)

    def test_remote_forwards_explicit_screen_ingest_false(self):
        ctx = create_screen_context(mode="remote", screen_ingest=False)
        self.assertFalse(ctx.screen_ingest)

    def test_remote_env_ingest_canonical_true_enables(self):
        with mock.patch.dict(os.environ, {
            "SCREEN_CONTEXT_MODE": "remote",
            "SENSOR_SCREEN_INGEST_ENABLED": "true",
        }, clear=True):
            ctx = create_screen_context()
        self.assertIsInstance(ctx, RemoteScreenContext)
        self.assertTrue(ctx.screen_ingest)

    def test_remote_env_canonical_false_overrides_legacy(self):
        with mock.patch.dict(os.environ, {
            "SCREEN_CONTEXT_MODE": "remote",
            "SENSOR_SCREEN_INGEST_ENABLED": "false",
            "WEB_SCREEN_CONTEXT_ENABLED": "true",
        }, clear=True):
            ctx = create_screen_context()
        self.assertIsInstance(ctx, RemoteScreenContext)
        self.assertFalse(ctx.screen_ingest)

    def test_remote_start_fails_closed_without_ingest(self):
        with mock.patch.dict(os.environ, {"SCREEN_CONTEXT_MODE": "remote"}, clear=True):
            ctx = create_screen_context()
        self.assertIsInstance(ctx, RemoteScreenContext)
        self.assertFalse(ctx.screen_ingest)
        self.assertFalse(ctx.start())
        self.assertFalse(ctx.is_running)


# --------------------------- ingest エンドポイント ---------------------------

class IngestEndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from starlette.testclient import TestClient
        from src.web import server
        cls.server = server
        cls.TestClient = TestClient

    def setUp(self):
        # latest.jpg / latest.json をテンポラリに向ける
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.server.SCREEN_DIR = d
        self.server.SCREEN_LATEST_JPG = d / "latest.jpg"
        self.server.SCREEN_LATEST_JSON = d / "latest.json"
        self._orig_ingest = {
            "_ingest_active_generation": self.server._ingest_active_generation,
            "_ingest_future": self.server._ingest_future,
            "_ingest_generation": self.server._ingest_generation,
            "_ingest_accepting": self.server._ingest_accepting,
            "_ingest_done_events": self.server._ingest_done_events,
            "screen_ingest_describer": self.server.screen_ingest_describer,
        }
        self.server._ingest_active_generation = None
        self.server._ingest_future = None
        self.server._ingest_generation = 0
        self.server._ingest_accepting = True
        self.server._ingest_done_events = {}
        self._orig_policy = self.server.sensor_policy
        self.server.sensor_policy = SensorPolicy(screen_ingest=True)

        class _FakeDescriber:
            model = "fake-vlm"

            def describe(self, jpeg):
                return "メインPCでブラウザを見ています。"

        self.server.screen_ingest_describer = _FakeDescriber()

    def _drain_ingest(self, timeout: float = 2.0) -> None:
        """teardown で実行中 ingest worker の完了を bounded に待つ (後続テストへの干渉防止)。

        TestClient の実 executor で走る worker を確実に後始末させる。実データ・実モデルは
        使わず、worker 本体の完了 Event のみを待つ。
        """
        generation = self.server._ingest_active_generation
        if generation is None:
            return
        event = self.server._ingest_done_events.get(generation)
        if event is not None:
            event.wait(timeout)

    def tearDown(self):
        self._drain_ingest()
        for name, value in self._orig_ingest.items():
            setattr(self.server, name, value)
        self.server.sensor_policy = self._orig_policy
        self.tmp.cleanup()

    def _client(self):
        return self.TestClient(self.server.app)

    _JPEG = b"\xff\xd8\xff" + b"\x00" * 200

    def test_token_unset_is_403(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SCREEN_INGEST_TOKEN", None)
            r = self._client().post(
                "/api/screen/ingest",
                content=self._JPEG,
                headers={"X-Screen-Token": "whatever", "Content-Type": "image/jpeg"},
            )
        self.assertEqual(r.status_code, 403)

    def test_token_mismatch_is_403(self):
        with mock.patch.dict(os.environ, {"SCREEN_INGEST_TOKEN": "secret"}):
            r = self._client().post(
                "/api/screen/ingest",
                content=self._JPEG,
                headers={"X-Screen-Token": "wrong", "Content-Type": "image/jpeg"},
            )
        self.assertEqual(r.status_code, 403)

    def test_valid_token_and_jpeg_is_200_and_writes_json(self):
        with mock.patch.dict(os.environ, {"SCREEN_INGEST_TOKEN": "secret"}):
            r = self._client().post(
                "/api/screen/ingest",
                content=self._JPEG,
                headers={"X-Screen-Token": "secret", "Content-Type": "image/jpeg"},
            )
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertTrue(body["ok"])
            self.assertTrue(body["described"])
            # 生JPEGは永続化しない (latest.jpg は書き込まれない)
            self.assertFalse(self.server.SCREEN_LATEST_JPG.exists())
            # 描写はバックグラウンド → latest.json 生成を待つ (bounded <=2s)
            deadline = time.time() + 2.0
            while time.time() < deadline and not self.server.SCREEN_LATEST_JSON.exists():
                time.sleep(0.05)
            self.assertTrue(self.server.SCREEN_LATEST_JSON.exists())
            data = json.loads(self.server.SCREEN_LATEST_JSON.read_text(encoding="utf-8"))
            self.assertEqual(data["source"], "remote")
            self.assertEqual(data["description"], "メインPCでブラウザを見ています。")
            self.assertIn("captured_at", data)
            self.assertIn("described_at", data)

    def test_policy_disabled_returns_403_even_with_valid_token(self):
        self.server.sensor_policy = SensorPolicy(screen_ingest=False)
        with mock.patch.dict(os.environ, {"SCREEN_INGEST_TOKEN": "secret"}):
            r = self._client().post(
                "/api/screen/ingest",
                content=self._JPEG,
                headers={"X-Screen-Token": "secret", "Content-Type": "image/jpeg"},
            )
        self.assertEqual(r.status_code, 403)
        self.assertFalse(self.server.SCREEN_LATEST_JPG.exists())
        self.assertFalse(self.server.SCREEN_LATEST_JSON.exists())

    def test_policy_disabled_rejects_before_reading_body(self):
        # token 単独では有効化されない。oversize body でも 413 ではなく 403 になる
        # = body を読まず・書き込みもスケジュールもしないことを示す。
        self.server.sensor_policy = SensorPolicy(screen_ingest=False)
        big = b"\xff\xd8\xff" + b"0" * (8 * 1024 * 1024 + 1)
        with mock.patch.dict(os.environ, {"SCREEN_INGEST_TOKEN": "secret"}):
            r = self._client().post(
                "/api/screen/ingest",
                content=big,
                headers={"X-Screen-Token": "secret", "Content-Type": "image/jpeg"},
            )
        self.assertEqual(r.status_code, 403)
        self.assertFalse(self.server.SCREEN_LATEST_JPG.exists())
        self.assertFalse(self.server.SCREEN_LATEST_JSON.exists())

    def test_policy_none_fails_closed(self):
        self.server.sensor_policy = None
        with mock.patch.dict(os.environ, {"SCREEN_INGEST_TOKEN": "secret"}):
            r = self._client().post(
                "/api/screen/ingest",
                content=self._JPEG,
                headers={"X-Screen-Token": "secret", "Content-Type": "image/jpeg"},
            )
        self.assertEqual(r.status_code, 403)

    def test_error_payload_exposes_no_env_or_token_values(self):
        self.server.sensor_policy = SensorPolicy(screen_ingest=False)
        with mock.patch.dict(os.environ, {"SCREEN_INGEST_TOKEN": "secret"}):
            r = self._client().post(
                "/api/screen/ingest",
                content=self._JPEG,
                headers={"X-Screen-Token": "secret", "Content-Type": "image/jpeg"},
            )
        self.assertEqual(r.json(), {"error": "forbidden"})

    def test_bad_magic_bytes_is_400(self):
        with mock.patch.dict(os.environ, {"SCREEN_INGEST_TOKEN": "secret"}):
            r = self._client().post(
                "/api/screen/ingest",
                content=b"not-a-jpeg-at-all",
                headers={"X-Screen-Token": "secret", "Content-Type": "image/jpeg"},
            )
        self.assertEqual(r.status_code, 400)

    def test_oversize_is_413(self):
        big = b"\xff\xd8\xff" + b"0" * (8 * 1024 * 1024 + 1)
        with mock.patch.dict(os.environ, {"SCREEN_INGEST_TOKEN": "secret"}):
            r = self._client().post(
                "/api/screen/ingest",
                content=big,
                headers={"X-Screen-Token": "secret", "Content-Type": "image/jpeg"},
            )
        self.assertEqual(r.status_code, 413)


# --------------------------- ingest 描写ライフサイクル (fake loop/future) ---------------------------

class _StuckFuture(asyncio.Future):
    """キャンセルしても完了しない Future (実行中 worker が cancel 後も続くことを模す)。

    cancel は呼び出しを記録するだけで Future を未完了のまま保ち、テストが明示的に
    set_result するまで pending を続ける。実スレッド・実 executor は使わない。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cancel_calls = 0

    def cancel(self, *args, **kwargs):
        self.cancel_calls += 1
        return False


class _CancelledFuture(asyncio.Future):
    """``cancel()`` で done (cancelled) になる Future。

    実 ``run_in_executor`` の Future は cancel で done になっても下位 executor worker は
    継続していることがある。cancel 済み Future の done だけでは完了判定できないことを
    検証するために使う (完了判定は worker 本体の Event のみ)。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cancel_calls = 0

    def cancel(self, *args, **kwargs):
        self.cancel_calls += 1
        super().cancel(*args, **kwargs)
        return True


class _FakeLoop:
    """``run_in_executor`` を差し替えるフェイク。

    提出を記録して Future を返すか、``raise_on_submit`` が設定されていれば例外を投げる。
    実 executor は使わない。
    """

    def __init__(self):
        self.submits = 0
        self.submitted_args = None
        self.raise_on_submit = None
        self.future_factory = _StuckFuture

    def run_in_executor(self, executor, func, *args):
        self.submits += 1
        self.submitted_args = args
        if self.raise_on_submit is not None:
            raise self.raise_on_submit
        return self.future_factory()


class ScreenIngestLifecycleTest(unittest.TestCase):
    """ingest 描写の単一飛行 / 世代ゲートを fake loop・fake future・blocking describer で
    決定的に検証。

    - 提出 (run_in_executor) 失敗: 単一飛行を原子的に解除し、後続提出を詰まらせない
    - シャットダウン中の実行: 受付 revoke → cancel → 完了 Event の bounded wait タイムアウト
      で ownership 保持 (cancel 済み Future の done でも worker 継続中は解除しない)
    - シャットダウン後の遅延完了: latest.json を書き込まない
    - restart 時の旧 worker との重なり防止: ownership 保持中は新規提出を拒否
    - revoke/コミット境界: revoke 先行なら書き込み抑止、コミット先行なら revoke が後続
    - 正常完了: ownership を解除し、restart (新世代) で再提出できる
    """

    _JPEG = b"\xff\xd8\xff" + b"\x00" * 32

    @classmethod
    def setUpClass(cls):
        from starlette.testclient import TestClient
        from src.web import server
        cls.server = server
        cls.TestClient = TestClient

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.server.SCREEN_DIR = d
        self.server.SCREEN_LATEST_JPG = d / "latest.jpg"
        self.server.SCREEN_LATEST_JSON = d / "latest.json"
        self._orig = {
            "_ingest_active_generation": self.server._ingest_active_generation,
            "_ingest_future": self.server._ingest_future,
            "_ingest_generation": self.server._ingest_generation,
            "_ingest_accepting": self.server._ingest_accepting,
            "_ingest_done_events": self.server._ingest_done_events,
            "screen_ingest_describer": self.server.screen_ingest_describer,
            "sensor_policy": self.server.sensor_policy,
        }
        self.server._ingest_active_generation = None
        self.server._ingest_future = None
        self.server._ingest_generation = 0
        self.server._ingest_accepting = True
        self.server._ingest_done_events = {}
        self.server.sensor_policy = SensorPolicy(screen_ingest=True)

        class _FakeDescriber:
            def describe(self, jpeg):
                return "canary late description"

        self.server.screen_ingest_describer = _FakeDescriber()

    def tearDown(self):
        for name, value in self._orig.items():
            setattr(self.server, name, value)
        self.tmp.cleanup()

    def _run(self, coro):
        asyncio.run(coro)

    def _client(self):
        return self.TestClient(self.server.app)

    def test_submit_failure_resets_single_flight_atomically(self):
        async def scenario():
            self.server._start_ingest_generation()
            loop = _FakeLoop()
            loop.raise_on_submit = RuntimeError("executor is shut down")
            with self.assertRaises(RuntimeError):
                self.server._submit_ingest_describe(loop, self._JPEG, time.time())
            # 原子的に解除され、保持 Future も残らない
            self.assertIsNone(self.server._ingest_active_generation)
            self.assertIsNone(self.server._ingest_future)
            # 失敗後は新たに提出できる (single-flight が詰まらない)
            loop.raise_on_submit = None
            self.assertTrue(
                self.server._submit_ingest_describe(loop, self._JPEG, time.time())
            )

        self._run(scenario())

    def test_endpoint_submit_failure_returns_fixed_503(self):
        with mock.patch.dict(os.environ, {"SCREEN_INGEST_TOKEN": "secret"}):
            with mock.patch.object(
                self.server,
                "_submit_ingest_describe",
                side_effect=RuntimeError("boom token=super-secret-canary-9f4b"),
            ):
                r = self._client().post(
                    "/api/screen/ingest",
                    content=self._JPEG,
                    headers={"X-Screen-Token": "secret", "Content-Type": "image/jpeg"},
                )
        self.assertEqual(r.status_code, 503)
        body = r.json()
        self.assertEqual(body["error"], "screen ingest describe submit failed")
        self.assertNotIn("super-secret-canary-9f4b", r.text)
        self.assertNotIn("boom", r.text)

    def test_submit_after_revoke_rejects_before_registering_anything(self):
        async def scenario():
            self.server._start_ingest_generation()
            loop = _FakeLoop()
            self.assertTrue(
                self.server._submit_ingest_describe(loop, self._JPEG, time.time())
            )
            await self.server._stop_ingest_describe(timeout=0.05)
            before_active = self.server._ingest_active_generation
            before_future = self.server._ingest_future
            before_events = dict(self.server._ingest_done_events)
            # revoke 後の提出は受付検査で何も登録せずに拒否される (固定 unavailable)。
            with self.assertRaises(ConnectionError):
                self.server._submit_ingest_describe(loop, self._JPEG, time.time())
            self.assertIs(self.server._ingest_active_generation, before_active)
            self.assertIs(self.server._ingest_future, before_future)
            self.assertEqual(self.server._ingest_done_events, before_events)
            # 繰り返し拒否しても Event / Future / 単一飛行は増えない。
            with self.assertRaises(ConnectionError):
                self.server._submit_ingest_describe(loop, self._JPEG, time.time())
            self.assertEqual(self.server._ingest_done_events, before_events)

        self._run(scenario())

    def test_endpoint_after_revoke_returns_fixed_503_and_never_described(self):
        async def revoke():
            self.server._start_ingest_generation()
            await self.server._stop_ingest_describe(timeout=0.01)

        asyncio.run(revoke())
        with mock.patch.dict(os.environ, {"SCREEN_INGEST_TOKEN": "secret"}):
            r = self._client().post(
                "/api/screen/ingest",
                content=self._JPEG,
                headers={"X-Screen-Token": "secret", "Content-Type": "image/jpeg"},
            )
        # revoke 後の受信は固定 503 (unavailable) で、described は一切返さない。
        self.assertEqual(r.status_code, 503)
        body = r.json()
        self.assertEqual(body["error"], "screen ingest describe submit failed")
        self.assertEqual(body["error_type"], "unavailable")
        self.assertNotIn("described", body)
        self.assertNotIn("secret", r.text)
        # revoke 済み提出は単一飛行状態・Future・完了 Event を作らない。
        self.assertIsNone(self.server._ingest_active_generation)
        self.assertIsNone(self.server._ingest_future)
        self.assertEqual(self.server._ingest_done_events, {})

    def test_shutdown_during_work_keeps_ownership_and_revokes_writes(self):
        async def scenario():
            self.server._start_ingest_generation()
            fut = _StuckFuture()
            loop = _FakeLoop()
            loop.future_factory = lambda: fut
            self.assertTrue(
                self.server._submit_ingest_describe(loop, self._JPEG, time.time())
            )
            self.assertIs(self.server._ingest_future, fut)
            submitted_gen = self.server._ingest_generation
            await self.server._stop_ingest_describe(timeout=0.05)
            # revoke: 提出世代の結果はもう受け付けられない
            self.assertFalse(self.server._ingest_results_accepted(submitted_gen))
            # cancel されたが worker 継続中 → ownership (単一飛行 + Future) を保持
            self.assertEqual(fut.cancel_calls, 1)
            self.assertIsNotNone(self.server._ingest_active_generation)
            self.assertIs(self.server._ingest_future, fut)

        self._run(scenario())

    def test_late_completion_after_shutdown_does_not_write(self):
        async def scenario():
            self.server._start_ingest_generation()
            fut = _StuckFuture()
            loop = _FakeLoop()
            loop.future_factory = lambda: fut
            self.assertTrue(
                self.server._submit_ingest_describe(loop, self._JPEG, time.time())
            )
            submitted_gen = self.server._ingest_generation
            await self.server._stop_ingest_describe(timeout=0.05)
            # シャットダウン後に worker が遅延完了しても latest.json は書かれない
            self.server._describe_ingested(self._JPEG, time.time(), submitted_gen)
            self.assertFalse(self.server.SCREEN_LATEST_JSON.exists())
            # 自分自身の ownership だけを後始末する
            self.assertIsNone(self.server._ingest_active_generation)
            self.assertIsNone(self.server._ingest_future)

        self._run(scenario())

    def test_describe_writes_only_while_generation_current(self):
        async def scenario():
            self.server._start_ingest_generation()
            loop = _FakeLoop()
            self.assertTrue(
                self.server._submit_ingest_describe(loop, self._JPEG, time.time())
            )
            current_gen = self.server._ingest_generation
            self.server._describe_ingested(self._JPEG, time.time(), current_gen)
            self.assertTrue(self.server.SCREEN_LATEST_JSON.exists())
            self.assertIsNone(self.server._ingest_active_generation)

        self._run(scenario())

    def test_normal_completion_cleanup_and_restart(self):
        async def scenario():
            self.server._start_ingest_generation()
            fut = _StuckFuture()
            loop = _FakeLoop()
            loop.future_factory = lambda: fut
            self.assertTrue(
                self.server._submit_ingest_describe(loop, self._JPEG, time.time())
            )
            gen = self.server._ingest_generation
            # worker 本体が正常完了し finally で完了 Event を set / ownership を解除する
            self.server._describe_ingested(self._JPEG, time.time(), gen)
            fut.set_result(None)  # asyncio Future 側も完了 (補助)
            await self.server._stop_ingest_describe(timeout=1.0)
            self.assertIsNone(self.server._ingest_active_generation)
            self.assertIsNone(self.server._ingest_future)
            # restart: 新世代で再提出できる
            self.server._start_ingest_generation()
            self.assertTrue(
                self.server._submit_ingest_describe(loop, self._JPEG, time.time())
            )
            self.assertIsNotNone(self.server._ingest_active_generation)

        self._run(scenario())

    def test_cancelled_future_done_but_worker_running_keeps_ownership(self):
        async def scenario():
            self.server._start_ingest_generation()
            fut = _CancelledFuture()
            loop = _FakeLoop()
            loop.future_factory = lambda: fut
            self.assertTrue(
                self.server._submit_ingest_describe(loop, self._JPEG, time.time())
            )
            gen = self.server._ingest_generation
            await self.server._stop_ingest_describe(timeout=0.05)
            # cancel で run_in_executor Future は done (cancelled) になっても、
            # 下位 worker の完了 Event は未 set → ownership は保持される。
            self.assertTrue(fut.done())
            self.assertTrue(fut.cancelled())
            self.assertEqual(fut.cancel_calls, 1)
            self.assertIsNotNone(self.server._ingest_active_generation)
            self.assertIs(self.server._ingest_future, fut)
            self.assertIsNotNone(self.server._ingest_done_events.get(gen))

        self._run(scenario())

    def test_restart_overlap_prevented_while_old_worker_running(self):
        async def scenario():
            self.server._start_ingest_generation()
            fut = _StuckFuture()
            loop = _FakeLoop()
            loop.future_factory = lambda: fut
            self.assertTrue(
                self.server._submit_ingest_describe(loop, self._JPEG, time.time())
            )
            old_gen = self.server._ingest_generation
            # 旧 worker が未完了のまま stop タイムアウト → ownership 保持
            await self.server._stop_ingest_describe(timeout=0.05)
            self.assertIsNotNone(self.server._ingest_active_generation)
            # restart しても新規提出は実行中の旧 worker と重ならないよう拒否される
            self.server._start_ingest_generation()
            self.assertFalse(
                self.server._submit_ingest_describe(loop, self._JPEG, time.time())
            )
            # 旧 worker が finally まで到達すると ownership が解除され、再提出できる
            self.server._describe_ingested(self._JPEG, time.time(), old_gen)
            self.assertIsNone(self.server._ingest_active_generation)
            self.assertIsNone(self.server._ingest_future)
            self.assertTrue(
                self.server._submit_ingest_describe(loop, self._JPEG, time.time())
            )

        self._run(scenario())

    def test_revoke_before_commit_suppresses_write(self):
        # revoke がコミットより先に lock を取ったとき、最終 acceptance check が失敗して
        # latest.json への書き込みは抑止される (シャットダウン後のコミットは起きない)。
        entered = threading.Event()
        release = threading.Event()

        class _BlockingDescriber:
            def describe(self, jpeg):
                entered.set()
                release.wait(2.0)
                return "canary revoke-first description"

        self.server.screen_ingest_describer = _BlockingDescriber()
        gen = 7
        self.server._ingest_active_generation = gen
        self.server._ingest_future = None
        self.server._ingest_generation = gen
        self.server._ingest_accepting = True
        self.server._ingest_done_events = {gen: threading.Event()}

        t = threading.Thread(
            target=lambda: self.server._describe_ingested(
                self._JPEG, time.time(), gen
            )
        )
        t.start()
        try:
            self.assertTrue(entered.wait(2.0))
            with self.server._ingest_describe_lock:
                self.server._ingest_accepting = False
                self.server._ingest_generation = gen + 1
        finally:
            # どんな失敗でも worker を解放して join し、テストがハングしないようにする。
            release.set()
            t.join(2.0)
        self.assertFalse(t.is_alive())
        self.assertFalse(self.server.SCREEN_LATEST_JSON.exists())
        self.assertIsNone(self.server._ingest_active_generation)
        self.assertIsNone(self.server._ingest_future)

    def test_commit_before_revoke_keeps_write(self):
        # worker がコミットを先に lock 下で完了させたとき、revoke はその完了後に続き、
        # 書き込み済み結果は残る (revoke がコミット済み結果を壊さない)。
        entered = threading.Event()
        release = threading.Event()

        class _BlockingDescriber:
            def describe(self, jpeg):
                entered.set()
                release.wait(2.0)
                return "canary commit-first description"

        self.server.screen_ingest_describer = _BlockingDescriber()
        gen = 8
        self.server._ingest_active_generation = gen
        self.server._ingest_future = None
        self.server._ingest_generation = gen
        self.server._ingest_accepting = True
        done_event = threading.Event()
        self.server._ingest_done_events = {gen: done_event}

        t = threading.Thread(
            target=lambda: self.server._describe_ingested(
                self._JPEG, time.time(), gen
            )
        )
        t.start()
        try:
            self.assertTrue(entered.wait(2.0))
            release.set()
            # worker 本体の finally が完了 Event を set するのはコミット後。それを待つことで
            # コミット完了後に revoke が続く順序を決定的に検証できる。
            self.assertTrue(done_event.wait(2.0))
        finally:
            # どんな失敗でも worker を解放して join し、テストがハングしないようにする。
            release.set()
            t.join(2.0)
        self.assertFalse(t.is_alive())
        self.assertTrue(self.server.SCREEN_LATEST_JSON.exists())
        # コミット後に revoke (stop 相当) しても書き込み済み結果は残る
        with self.server._ingest_describe_lock:
            self.server._ingest_accepting = False
            self.server._ingest_generation = gen + 1
        self.assertTrue(self.server.SCREEN_LATEST_JSON.exists())
        self.assertIsNone(self.server._ingest_active_generation)
        self.assertIsNone(self.server._ingest_future)


# --------------------------- ingest status と policy gate ---------------------------

class IngestStatusPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from starlette.testclient import TestClient
        from src.web import server
        cls.server = server
        cls.TestClient = TestClient

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.server.SCREEN_DIR = d
        self.server.SCREEN_LATEST_JPG = d / "latest.jpg"
        self.server.SCREEN_LATEST_JSON = d / "latest.json"
        self._orig_policy = self.server.sensor_policy
        self._orig_screen = self.server.screen

    def tearDown(self):
        self.server.sensor_policy = self._orig_policy
        self.server.screen = self._orig_screen
        self.tmp.cleanup()

    def _client(self):
        return self.TestClient(self.server.app)

    def _stale_json(self):
        _write_json(self.server.SCREEN_LATEST_JSON, {
            "description": "古い画面の描写",
            "captured_at": time.time() - 100,
            "described_at": time.time() - 100,
            "source": "remote",
        })

    def test_status_when_ingest_disabled_hides_stale_description(self):
        self._stale_json()
        self.server.sensor_policy = SensorPolicy(screen_ingest=False)
        body = self._client().get("/api/screen/status").json()
        self.assertEqual(body["ingest"], {"enabled": False})
        self.assertNotIn("description", body["ingest"])
        self.assertNotIn("available", body["ingest"])
        self.assertNotIn("token_configured", body["ingest"])

    def test_status_when_ingest_disabled_removes_legacy_latest_jpg(self):
        self.server.SCREEN_LATEST_JPG.write_bytes(b"\xff\xd8\xff" + b"\x00" * 16)
        self.server.sensor_policy = SensorPolicy(screen_ingest=False)
        self._client().get("/api/screen/status")
        self.assertFalse(self.server.SCREEN_LATEST_JPG.exists())

    def test_status_when_ingest_enabled_minimizes_privacy_schema(self):
        # プライバシー最小化スキーマ: VLM 描写テキスト (description) は有効時でも露出しない。
        # available / source / timestamps / token_configured のみ残す。
        _write_json(self.server.SCREEN_LATEST_JSON, {
            "description": "作業中",
            "captured_at": time.time(),
            "described_at": time.time(),
            "source": "remote",
        })
        self.server.sensor_policy = SensorPolicy(screen_ingest=True)
        with mock.patch.dict(os.environ, {"SCREEN_INGEST_TOKEN": "secret"}):
            body = self._client().get("/api/screen/status").json()
        ingest = body["ingest"]
        self.assertEqual(ingest["available"], True)
        self.assertEqual(ingest["source"], "remote")
        self.assertIn("captured_at", ingest)
        self.assertIn("described_at", ingest)
        self.assertIs(ingest["token_configured"], True)
        self.assertNotIn("description", ingest)
        self.assertNotIn("jpg_exists", ingest)
        self.assertNotIn("model", ingest)

    def test_status_when_ingest_enabled_vlm_text_absent_canary(self):
        # canary: どんな最新 JSON でも VLM 描写テキストが status に漏れないことを保証。
        _write_json(self.server.SCREEN_LATEST_JSON, {
            "description": "絶対に漏れてはいけない秘密の画面内容",
            "captured_at": time.time(),
            "described_at": time.time(),
            "source": "remote",
        })
        self.server.sensor_policy = SensorPolicy(screen_ingest=True)
        body = self._client().get("/api/screen/status").json()
        raw = json.dumps(body, ensure_ascii=False)
        self.assertNotIn("絶対に漏れてはいけない", raw)
        self.assertNotIn("description", body["ingest"])

    def test_status_when_ingest_disabled_omits_token_configured(self):
        self.server.sensor_policy = SensorPolicy(screen_ingest=False)
        with mock.patch.dict(os.environ, {"SCREEN_INGEST_TOKEN": "secret"}):
            body = self._client().get("/api/screen/status").json()
        self.assertNotIn("token_configured", body["ingest"])

    def test_remove_legacy_latest_jpg_is_best_effort_and_silent(self):
        self.server.SCREEN_LATEST_JPG.write_bytes(b"\xff\xd8\xff" + b"\x00" * 16)
        self.server._remove_legacy_latest_jpg()
        self.assertFalse(self.server.SCREEN_LATEST_JPG.exists())


# --------------------------- screen 構築・start の policy gate ---------------------------

class _FakeScreen:
    def __init__(self) -> None:
        self.started = False

    def start(self) -> bool:
        self.started = True
        return True

    def get_status(self) -> dict:
        return {"mode": "local", "model": "fake-vlm", "analysis_interval": 90.0}


class ScreenPolicyGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimpleNamespace(
            ollama_base_url="http://unused.invalid", model="config-model"
        )

    def test_disabled_default_does_not_construct_or_start_screen(self):
        with mock.patch(
            "src.web.server.create_screen_context",
            side_effect=AssertionError("must not construct screen"),
        ):
            ctx = self._init(SensorPolicy(screen_capture=False))
        self.assertIsNone(ctx)

    def test_enabled_constructs_and_starts_screen(self):
        fake = _FakeScreen()
        with mock.patch("src.web.server.create_screen_context", return_value=fake):
            ctx = self._init(SensorPolicy(screen_capture=True))
        self.assertIs(ctx, fake)
        self.assertTrue(fake.started)

    def test_policy_none_fails_closed(self):
        with mock.patch(
            "src.web.server.create_screen_context",
            side_effect=AssertionError("must not construct screen"),
        ):
            ctx = self._init(None)
        self.assertIsNone(ctx)

    def test_enabled_but_non_ollama_backend_skips(self):
        with mock.patch(
            "src.web.server.create_screen_context",
            side_effect=AssertionError("must not construct screen"),
        ):
            ctx = self._init(SensorPolicy(screen_capture=True), kind="openai_compatible")
        self.assertIsNone(ctx)

    def test_start_failure_returns_none(self):
        class _Unstartable:
            def start(self) -> bool:
                return False

        with mock.patch("src.web.server.create_screen_context", return_value=_Unstartable()):
            ctx = self._init(SensorPolicy(screen_capture=True))
        self.assertIsNone(ctx)

    def _init(self, policy, *, kind="ollama"):
        from src.web import server
        return server._init_screen_from_policy(
            policy,
            config=self.config,
            primary_provider_kind=kind,
        )


# --------------------------- screen_agent 純粋ロジック ---------------------------

def _load_screen_agent():
    path = PROJECT_ROOT / "scripts" / "screen_agent.py"
    spec = importlib.util.spec_from_file_location("screen_agent", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ScreenAgentLogicTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = _load_screen_agent()

    def test_image_hash_stable_and_distinct(self):
        a = self.agent
        self.assertEqual(a.image_hash(b"abc"), a.image_hash(b"abc"))
        self.assertNotEqual(a.image_hash(b"abc"), a.image_hash(b"abd"))

    def test_should_send_first_time(self):
        a = self.agent
        self.assertTrue(a.should_send("h1", None, None, now=100.0))

    def test_should_send_when_hash_changed(self):
        a = self.agent
        self.assertTrue(a.should_send("h2", "h1", 100.0, now=110.0, min_resend_interval=600.0))

    def test_skip_when_same_hash_within_interval(self):
        a = self.agent
        self.assertFalse(a.should_send("h1", "h1", 100.0, now=200.0, min_resend_interval=600.0))

    def test_resend_when_same_hash_after_interval(self):
        a = self.agent
        self.assertTrue(a.should_send("h1", "h1", 100.0, now=100.0 + 601.0, min_resend_interval=600.0))

    def test_next_backoff_caps(self):
        a = self.agent
        self.assertEqual(a.next_backoff(200.0), 300.0)  # 400 -> capped at 300
        self.assertEqual(a.next_backoff(5.0), 10.0)


class _ExplodingScreenAgentCallable:
    def __init__(self, name: str):
        self.calls = 0
        self.name = name

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError(f"{self.name}_CANARY")


class _ExplodingScreenAgentDependency:
    def __init__(self, name: str):
        self.name = name

    def __bool__(self):
        raise AssertionError(f"{self.name}_CANARY")


class ScreenAgentSourceGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = _load_screen_agent()

    def _args(self, **overrides):
        values = {
            "enable_screen_capture": False,
            "url": "http://URL_CANARY.invalid",
            "token": "TOKEN_CANARY",
            "max_edge": 1344,
            "once": False,
            "interval": 0.0,
            "min_resend": 600.0,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def _assert_no_sensitive_log(self, logs: str, *values: str):
        for value in values:
            self.assertNotIn(value, logs)

    def test_default_false_invalid_one_yes_on_are_disabled_before_dependencies(self):
        values = (None, "false", "invalid", "1", "yes", "on")
        for value in values:
            with self.subTest(value="default" if value is None else value):
                capture = _ExplodingScreenAgentCallable("CAPTURE")
                send = _ExplodingScreenAgentCallable("SEND")
                deps = {
                    name: _ExplodingScreenAgentDependency(name.upper())
                    for name in ("HAS_MSS", "HAS_PIL", "HAS_HTTPX")
                }
                env = {} if value is None else {
                    "SENSOR_SCREEN_CAPTURE_ENABLED": value,
                }
                output = io.StringIO()
                with mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch.object(self.agent, "capture_jpeg", capture), \
                    mock.patch.object(self.agent, "send_jpeg", send), \
                    mock.patch.object(self.agent, "HAS_MSS", deps["HAS_MSS"]), \
                    mock.patch.object(self.agent, "HAS_PIL", deps["HAS_PIL"]), \
                    mock.patch.object(self.agent, "HAS_HTTPX", deps["HAS_HTTPX"]), \
                    redirect_stdout(output):
                    result = self.agent.run(self._args(once=True))

                self.assertEqual(result, 4)
                self.assertEqual(capture.calls, 0)
                self.assertEqual(send.calls, 0)
                self._assert_no_sensitive_log(
                    output.getvalue(), "URL_CANARY", "TOKEN_CANARY", "CANARY"
                )

    def test_token_or_once_alone_does_not_enable_capture(self):
        for name, overrides in (
            ("token", {"token": "TOKEN_CANARY", "url": ""}),
            ("once", {"once": True, "url": "", "token": ""}),
        ):
            with self.subTest(name=name):
                capture = _ExplodingScreenAgentCallable("CAPTURE")
                send = _ExplodingScreenAgentCallable("SEND")
                env = {}
                output = io.StringIO()
                with mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch.object(self.agent, "capture_jpeg", capture), \
                    mock.patch.object(self.agent, "send_jpeg", send), \
                    mock.patch.object(self.agent, "HAS_MSS", _ExplodingScreenAgentDependency("MSS")), \
                    mock.patch.object(self.agent, "HAS_PIL", _ExplodingScreenAgentDependency("PIL")), \
                    mock.patch.object(self.agent, "HAS_HTTPX", _ExplodingScreenAgentDependency("HTTPX")), \
                    redirect_stdout(output):
                    result = self.agent.run(self._args(**overrides))

                self.assertEqual(result, 4)
                self.assertEqual(capture.calls, 0)
                self.assertEqual(send.calls, 0)
                self._assert_no_sensitive_log(
                    output.getvalue(), "URL_CANARY", "TOKEN_CANARY", "CANARY"
                )

    def test_exact_true_whitespace_and_case_enable_fake_once(self):
        for value in ("true", " TRUE ", "TrUe", "\ttrue\n"):
            with self.subTest(value=repr(value)):
                captured = []
                sent = []
                jpeg = b"\xff\xd8JPEG_BYTES_CANARY"

                def capture_jpeg(*args, **kwargs):
                    captured.append((args, kwargs))
                    return jpeg

                def send_jpeg(url, token, body):
                    sent.append((url, token, body))

                output = io.StringIO()
                with mock.patch.dict(
                    os.environ,
                    {"SENSOR_SCREEN_CAPTURE_ENABLED": value},
                    clear=True,
                ), mock.patch.object(self.agent, "HAS_MSS", True), \
                    mock.patch.object(self.agent, "HAS_PIL", True), \
                    mock.patch.object(self.agent, "HAS_HTTPX", True), \
                    mock.patch.object(self.agent, "capture_jpeg", capture_jpeg), \
                    mock.patch.object(self.agent, "send_jpeg", send_jpeg), \
                    redirect_stdout(output):
                    result = self.agent.run(self._args(once=True))

                self.assertEqual(result, 0)
                self.assertEqual(len(captured), 1)
                self.assertEqual(len(sent), 1)
                self.assertEqual(sent[0], ("http://URL_CANARY.invalid", "TOKEN_CANARY", jpeg))
                self._assert_no_sensitive_log(
                    output.getvalue(),
                    "http://URL_CANARY.invalid",
                    "TOKEN_CANARY",
                    self.agent.image_hash(jpeg),
                    "JPEG_BYTES_CANARY",
                    "raw exception",
                )

    def test_cli_flag_overrides_false_environment(self):
        captured = []
        sent = []
        jpeg = b"\xff\xd8CLI_BYTES_CANARY"

        def capture_jpeg(*args, **kwargs):
            captured.append(True)
            return jpeg

        def send_jpeg(url, token, body):
            sent.append((url, token, body))

        output = io.StringIO()
        with mock.patch.dict(
            os.environ,
            {"SENSOR_SCREEN_CAPTURE_ENABLED": "false"},
            clear=True,
        ), mock.patch.object(self.agent, "HAS_MSS", True), \
            mock.patch.object(self.agent, "HAS_PIL", True), \
            mock.patch.object(self.agent, "HAS_HTTPX", True), \
            mock.patch.object(self.agent, "capture_jpeg", capture_jpeg), \
            mock.patch.object(self.agent, "send_jpeg", send_jpeg), \
            redirect_stdout(output):
            result = self.agent.run(self._args(enable_screen_capture=True, once=True))

        self.assertEqual(result, 0)
        self.assertEqual(len(captured), 1)
        self.assertEqual(sent, [("http://URL_CANARY.invalid", "TOKEN_CANARY", jpeg)])
        self._assert_no_sensitive_log(
            output.getvalue(),
            "http://URL_CANARY.invalid",
            "TOKEN_CANARY",
            self.agent.image_hash(jpeg),
            "CLI_BYTES_CANARY",
            "raw exception",
        )

    def test_failure_logs_expose_no_raw_ascii_or_cp932_exception(self):
        class AsciiCaptureFailure(RuntimeError):
            pass

        class Cp932SendFailure(RuntimeError):
            pass

        failures = (
            ("capture", AsciiCaptureFailure(
                "ASCII_RAW_EXCEPTION_CANARY url=URL_CANARY token=TOKEN_CANARY"
            )),
            ("send", Cp932SendFailure(
                "CP932生例外CANARY url=URL_CANARY token=TOKEN_CANARY"
            )),
        )
        for stage, failure in failures:
            with self.subTest(stage=stage):
                captured = []
                sent = []
                jpeg = b"\xff\xd8FAILURE_BYTES_CANARY"

                def capture_jpeg(*args, **kwargs):
                    captured.append(True)
                    if stage == "capture":
                        raise failure
                    return jpeg

                def send_jpeg(url, token, body):
                    sent.append((url, token, body))
                    if stage == "send":
                        raise failure

                output = io.StringIO()
                with mock.patch.dict(os.environ, {}, clear=True), \
                    mock.patch.object(self.agent, "HAS_MSS", True), \
                    mock.patch.object(self.agent, "HAS_PIL", True), \
                    mock.patch.object(self.agent, "HAS_HTTPX", True), \
                    mock.patch.object(self.agent, "capture_jpeg", capture_jpeg), \
                    mock.patch.object(self.agent, "send_jpeg", send_jpeg), \
                    redirect_stdout(output):
                    result = self.agent.run(
                        self._args(enable_screen_capture=True, once=True)
                    )

                logs = output.getvalue()
                self.assertEqual(result, 1)
                self.assertEqual(len(captured), 1)
                self.assertEqual(len(sent), 0 if stage == "capture" else 1)
                self.assertIn(
                    "capture failed" if stage == "capture" else "send failed",
                    logs,
                )
                self._assert_no_sensitive_log(
                    logs,
                    str(failure),
                    failure.args[0],
                    "URL_CANARY",
                    "TOKEN_CANARY",
                    "FAILURE_BYTES_CANARY",
                    "raw exception",
                )

    def test_scripts_and_static_copy_are_byte_equal(self):
        script = PROJECT_ROOT / "scripts" / "screen_agent.py"
        static = PROJECT_ROOT / "src" / "web" / "static" / "screen_agent.py"
        self.assertEqual(script.read_bytes(), static.read_bytes())


if __name__ == "__main__":
    unittest.main()
