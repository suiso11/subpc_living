"""Monitor Context Provider / ChatSession互換wiringのテスト。"""
from __future__ import annotations

import io
import json
import os
import sqlite3
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.chat.session import ChatSession
from src.context import MonitorContextProvider, MonitorSource
from src.context.builder import ContextBuilder
from src.context.contracts import ContextBlock
from src.context.providers.monitor import MonitorContextProvider as ProviderImpl
from src.context.providers.monitor import MonitorSource as SourceImpl
import src.monitor.collector as collector_mod
from src.monitor.collector import SystemCollector, SystemMetrics
from src.monitor.context import MonitorContext
from src.monitor.storage import MetricsStorage, MetricsStorageError


class _FakeMonitor:
    def __init__(
        self,
        text: str = "\n--- サブPCの現在の状態 ---\n- CPU: 10% (低負荷)",
    ) -> None:
        self._text = text
        self.calls: int = 0

    def get_context_text(self) -> str:
        self.calls += 1
        return self._text


class _EmptyMonitor:
    def get_context_text(self) -> str:
        return ""


class _NonStrMonitor:
    def get_context_text(self) -> str:
        return None  # type: ignore[return-value]


class _BrokenMonitor:
    def get_context_text(self) -> str:
        raise RuntimeError("secret monitor body")


class _FakeVision:
    def get_context_text(self) -> str:
        return "\n[Vision] 現在の視界"


class _FakeScreen:
    def get_context_text(self) -> str:
        return "\n[Screen] 画面の内容"


class MonitorSourceProtocolTest(unittest.TestCase):
    def test_monitors_conform_to_monitor_source(self) -> None:
        self.assertIsInstance(_FakeMonitor(), MonitorSource)
        self.assertIsInstance(_BrokenMonitor(), MonitorSource)

    def test_unrelated_object_does_not_conform(self) -> None:
        self.assertNotIsInstance(object(), MonitorSource)


class MonitorContextProviderTest(unittest.TestCase):
    def test_returns_block_with_expected_metadata(self) -> None:
        monitor = _FakeMonitor()
        result = MonitorContextProvider.collect(monitor)
        self.assertIsNotNone(result)
        self.assertEqual(result.source, "monitor")
        self.assertEqual(result.sensitivity, "personal")
        self.assertIs(result.local_only, True)
        self.assertIsInstance(result.content, str)
        self.assertEqual(result.content, monitor._text)
        self.assertEqual(monitor.calls, 1)

    def test_empty_text_returns_none(self) -> None:
        self.assertIsNone(MonitorContextProvider.collect(_EmptyMonitor()))

    def test_whitespace_text_returns_none(self) -> None:
        self.assertIsNone(MonitorContextProvider.collect(_FakeMonitor("   \n  ")))

    def test_non_str_result_returns_none(self) -> None:
        self.assertIsNone(MonitorContextProvider.collect(_NonStrMonitor()))

    def test_exception_returns_none(self) -> None:
        self.assertIsNone(MonitorContextProvider.collect(_BrokenMonitor()))

    def test_exception_logs_type_only_not_body(self) -> None:
        with self.assertLogs(
            "src.context.providers.monitor", level="WARNING"
        ) as captured:
            result = MonitorContextProvider.collect(_BrokenMonitor())
        self.assertIsNone(result)
        self.assertEqual(len(captured.output), 1)
        self.assertIn("RuntimeError", captured.output[0])
        self.assertNotIn("secret monitor body", captured.output[0])


class MonitorCloudFilterTest(unittest.TestCase):
    def test_cloud_target_excludes_personal_local_only_monitor(self) -> None:
        builder = ContextBuilder("base")
        monitor_block = MonitorContextProvider.collect(_FakeMonitor())
        public_block = ContextBlock(
            source="screen", content="公開", sensitivity="public"
        )
        result = builder.build_system_content(
            [monitor_block, public_block], privacy="cloud_allowed", target_local=False
        )
        self.assertEqual(result, "base公開")


class ChatSessionMonitorWiringTest(unittest.TestCase):
    def test_exact_payload_with_vision_monitor_screen_order(self) -> None:
        monitor = _FakeMonitor()
        session = ChatSession(
            system_prompt="sys",
            vision_context=_FakeVision(),
            monitor_context=monitor,
            screen_context=_FakeScreen(),
        )
        session.add_user_message("hi")
        session.add_assistant_message("a")
        self.assertEqual(
            session.build_messages(),
            [
                {
                    "role": "system",
                    "content": (
                        "sys\n[Vision] 現在の視界"
                        "\n--- サブPCの現在の状態 ---\n- CPU: 10% (低負荷)"
                        "\n[Screen] 画面の内容"
                    ),
                },
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "a"},
            ],
        )

    def test_empty_monitor_payload_matches_no_monitor(self) -> None:
        session = ChatSession(
            system_prompt="sys",
            vision_context=_FakeVision(),
            monitor_context=_EmptyMonitor(),
            screen_context=_FakeScreen(),
        )
        session.add_user_message("hi")
        session.add_assistant_message("a")
        self.assertEqual(
            session.build_messages(),
            [
                {
                    "role": "system",
                    "content": "sys\n[Vision] 現在の視界\n[Screen] 画面の内容",
                },
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "a"},
            ],
        )

    def test_broken_monitor_does_not_stop_conversation(self) -> None:
        session = ChatSession(
            system_prompt="sys",
            vision_context=_FakeVision(),
            monitor_context=_BrokenMonitor(),
            screen_context=_FakeScreen(),
        )
        session.add_user_message("hi")
        session.add_assistant_message("a")
        with self.assertLogs("src.context.providers.monitor", level="WARNING"):
            messages = session.build_messages()
        self.assertEqual(
            messages,
            [
                {
                    "role": "system",
                    "content": "sys\n[Vision] 現在の視界\n[Screen] 画面の内容",
                },
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "a"},
            ],
        )

    def test_no_monitor_unchanged(self) -> None:
        session = ChatSession(
            system_prompt="sys",
            vision_context=_FakeVision(),
            screen_context=_FakeScreen(),
        )
        session.add_user_message("hi")
        session.add_assistant_message("a")
        self.assertEqual(
            session.build_messages(),
            [
                {
                    "role": "system",
                    "content": "sys\n[Vision] 現在の視界\n[Screen] 画面の内容",
                },
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "a"},
            ],
        )


class RootContextPublicAPITest(unittest.TestCase):
    def test_monitor_provider_exported_from_root(self) -> None:
        import src.context

        self.assertIn("MonitorContextProvider", src.context.__all__)
        self.assertIs(src.context.MonitorContextProvider, ProviderImpl)

    def test_monitor_source_exported_from_root(self) -> None:
        import src.context

        self.assertIn("MonitorSource", src.context.__all__)
        self.assertIs(src.context.MonitorSource, SourceImpl)
        self.assertIsInstance(_FakeMonitor(), src.context.MonitorSource)

    def test_monitor_exported_from_providers(self) -> None:
        import src.context.providers

        self.assertIn("MonitorContextProvider", src.context.providers.__all__)
        self.assertIn("MonitorSource", src.context.providers.__all__)


class ProcessDetailsCollectorPolicyTest(unittest.TestCase):
    """プロセス詳細収集の opt-in (SENSOR_PROCESS_DETAILS_ENABLED) を固定。

    実プロセスは検査しない。psutil を偽データでモックし、収集経路の分岐だけを確認する。
    """

    @staticmethod
    def _fake_procs():
        p1 = mock.Mock()
        p1.info = {"pid": 1234, "name": "secret-app", "cpu_percent": 88.0}
        p2 = mock.Mock()
        p2.info = {"pid": 5678, "name": "another-app", "cpu_percent": 44.0}
        return [p1, p2]

    def _collect(self, enabled: bool | None):
        procs = self._fake_procs()
        process_iter = mock.Mock(return_value=procs)
        fake_psutil = SimpleNamespace(
            pids=lambda: [1, 2, 3],
            process_iter=process_iter,
            cpu_percent=lambda **kw: 50.0,
            cpu_freq=lambda: None,
            getloadavg=lambda: (1.0, 1.0, 1.0),
            virtual_memory=lambda: SimpleNamespace(
                total=8 * 1024 ** 3, used=2 * 1024 ** 3, percent=25.0
            ),
            swap_memory=lambda: SimpleNamespace(percent=0.0),
            disk_usage=lambda path: SimpleNamespace(
                total=100 * 1024 ** 3, used=50 * 1024 ** 3, percent=50.0
            ),
            disk_io_counters=lambda: SimpleNamespace(read_bytes=0, write_bytes=0),
            net_io_counters=lambda: SimpleNamespace(bytes_sent=0, bytes_recv=0),
            sensors_temperatures=lambda: {},
            NoSuchProcess=Exception,
            AccessDenied=Exception,
        )
        kwargs = {} if enabled is None else {"process_details_enabled": enabled}
        with mock.patch.object(collector_mod, "psutil", fake_psutil, create=True), mock.patch.object(
            collector_mod, "HAS_PSUTIL", True
        ):
            collector = SystemCollector(interval=9999.0, **kwargs)
            with mock.patch.object(SystemCollector, "_collect_gpu", side_effect=lambda m: m):
                metrics = collector.collect_once()
        return collector, metrics, process_iter

    def test_default_is_disabled_and_collects_no_details(self) -> None:
        collector, metrics, process_iter = self._collect(enabled=None)
        self.assertFalse(collector._process_details_enabled)
        self.assertEqual(metrics.top_cpu_processes, [])
        self.assertEqual(metrics.process_count, 3)
        process_iter.assert_not_called()

    def test_false_collects_no_details(self) -> None:
        collector, metrics, process_iter = self._collect(enabled=False)
        self.assertFalse(collector._process_details_enabled)
        self.assertEqual(metrics.top_cpu_processes, [])
        self.assertEqual(metrics.process_count, 3)
        process_iter.assert_not_called()

    def test_true_collects_top_detail_list(self) -> None:
        collector, metrics, process_iter = self._collect(enabled=True)
        self.assertTrue(collector._process_details_enabled)
        self.assertEqual(
            metrics.top_cpu_processes,
            [
                {"name": "secret-app", "pid": 1234, "cpu_percent": 88.0},
                {"name": "another-app", "pid": 5678, "cpu_percent": 44.0},
            ],
        )
        self.assertEqual(metrics.process_count, 3)
        process_iter.assert_called_once()


class _FakeThread:
    """実際には実行しない収集スレッド。start/join/is_alive を記録する。"""

    def __init__(self, target=None, daemon=True) -> None:
        self._target = target
        self._daemon = daemon
        self._alive = False
        self.started = 0
        self.join_timeouts: list[float | None] = []

    def start(self) -> None:
        self.started += 1
        self._alive = True

    def join(self, timeout=None) -> None:
        self.join_timeouts.append(timeout)

    def is_alive(self) -> bool:
        return self._alive

    def mark_dead(self) -> None:
        self._alive = False


class SystemCollectorLifecycleTest(unittest.TestCase):
    """SystemCollector のライフサイクルを実スレッド・実psutilなしで決定的に検証。

    収集は注入した collect_fn、スレッドは注入した thread_factory の偽物で代替し、
    プロセス・ファイル・ネットワーク等の実データには一切触れない。
    """

    _FAKE_PSUTIL = SimpleNamespace(
        disk_io_counters=lambda: SimpleNamespace(read_bytes=0, write_bytes=0),
        net_io_counters=lambda: SimpleNamespace(bytes_sent=0, bytes_recv=0),
    )

    def _make_collector(self, join_timeout: float = 5.0):
        threads: list[_FakeThread] = []

        def factory() -> _FakeThread:
            t = _FakeThread()
            threads.append(t)
            return t

        def collect() -> SystemMetrics:
            return SystemMetrics(timestamp=0.0, process_count=3)

        collector = SystemCollector(
            interval=9999.0,
            join_timeout=join_timeout,
            thread_factory=factory,
            collect_fn=collect,
        )
        return collector, threads

    def setUp(self) -> None:
        patchers = [
            mock.patch.object(collector_mod, "psutil", self._FAKE_PSUTIL, create=True),
            mock.patch.object(collector_mod, "HAS_PSUTIL", True),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)

    def test_duplicate_start_does_not_create_second_thread(self) -> None:
        collector, threads = self._make_collector()
        self.assertTrue(collector.start())
        self.assertTrue(collector.start())
        self.assertTrue(collector.is_running)
        self.assertEqual(len(threads), 1)
        self.assertEqual(threads[0].started, 1)
        self.assertIs(collector._thread, threads[0])

    def test_stop_after_normal_run_reclaims_thread(self) -> None:
        collector, threads = self._make_collector()
        collector.start()
        threads[0].mark_dead()
        collector.stop()
        self.assertFalse(collector.is_running)
        self.assertIsNone(collector._thread)

    def test_stop_timeout_retains_live_thread(self) -> None:
        collector, threads = self._make_collector(join_timeout=5.0)
        collector.start()
        collector.stop()  # ブロックされていて生存し続ける
        self.assertFalse(collector.is_running)
        self.assertTrue(collector.thread_alive)
        self.assertTrue(collector.stop_pending)
        self.assertIs(collector._thread, threads[0])
        self.assertEqual(threads[0].join_timeouts, [5.0])

    def test_repeated_stop_is_idempotent(self) -> None:
        collector, threads = self._make_collector()
        collector.start()
        threads[0].mark_dead()
        collector.stop()
        collector.stop()
        collector.stop()
        self.assertIsNone(collector._thread)
        self.assertEqual(threads[0].join_timeouts.count(5.0), 1)

    def test_reclaim_happens_only_after_death(self) -> None:
        collector, threads = self._make_collector()
        collector.start()
        for _ in range(2):
            collector.stop()  # 生存中の間は回収されない
            self.assertIs(collector._thread, threads[0])
        threads[0].mark_dead()
        collector.stop()
        self.assertIsNone(collector._thread)

    def test_start_refuses_to_reactivate_stop_pending_live_worker(self) -> None:
        collector, threads = self._make_collector()
        self.assertTrue(collector.start())
        collector.stop()  # タイムアウトで生存継続 (stop_pending)
        self.assertFalse(collector.is_running)
        self.assertTrue(collector.thread_alive)
        self.assertTrue(collector.stop_pending)
        # 停止保留中の旧 worker は reactivate しない。_running も変更しない。
        self.assertFalse(collector.start())
        self.assertFalse(collector.is_running)
        self.assertTrue(collector.thread_alive)
        self.assertIs(collector._thread, threads[0])
        self.assertEqual(len(threads), 1)
        # 死を確認できた繰り返し stop で pending が解消し、再起動できる
        threads[0].mark_dead()
        collector.stop()
        self.assertFalse(collector.stop_pending)
        self.assertIsNone(collector._thread)
        self.assertTrue(collector.start())
        self.assertTrue(collector.is_running)
        self.assertEqual(len(threads), 2)

    def test_repeated_stop_clears_pending_after_death(self) -> None:
        collector, threads = self._make_collector()
        collector.start()
        collector.stop()
        self.assertTrue(collector.stop_pending)
        self.assertTrue(collector.thread_alive)
        threads[0].mark_dead()
        collector.stop()
        self.assertFalse(collector.stop_pending)
        self.assertFalse(collector.thread_alive)
        self.assertIsNone(collector._thread)
        self.assertFalse(collector.is_running)

    def test_thread_alive_and_stop_pending_properties(self) -> None:
        collector, _ = self._make_collector()
        self.assertFalse(collector.thread_alive)
        self.assertFalse(collector.stop_pending)

    def test_restart_allowed_after_thread_reclaimed(self) -> None:
        collector, threads = self._make_collector()
        self.assertTrue(collector.start())
        threads[0].mark_dead()
        collector.stop()
        self.assertTrue(collector.start())
        self.assertTrue(collector.is_running)
        self.assertEqual(len(threads), 2)
        self.assertIs(collector._thread, threads[1])

    def test_thread_factory_failure_does_not_expose_raw_exception(self) -> None:
        collector, threads = self._make_collector()

        def bad_factory():
            raise RuntimeError("secret factory detail")

        collector._thread_factory = bad_factory
        started = collector.start()  # 生例外を露出しない
        self.assertFalse(started)
        self.assertFalse(collector.is_running)
        self.assertIsNone(collector._thread)

    def test_is_running_false_when_thread_dies_unexpectedly(self) -> None:
        collector, threads = self._make_collector()
        self.assertTrue(collector.start())
        threads[0].mark_dead()
        # 生存スレッドが無いときは「稼働中」として報告しない
        self.assertFalse(collector.is_running)
        collector.stop()
        self.assertIsNone(collector._thread)
        self.assertTrue(collector.start())
        self.assertTrue(collector.is_running)

    def test_partial_start_retains_live_ownership(self) -> None:
        collector, threads = self._make_collector()

        class _PartialStartThread(_FakeThread):
            def start(self) -> None:
                self.started += 1
                self._alive = True
                raise RuntimeError("secret partial start detail")

        collector._thread_factory = lambda: _PartialStartThread()
        started = collector.start()
        # start() が例外を投げてもスレッドが生存していれば保持し True を返す
        self.assertTrue(started)
        self.assertTrue(collector.is_running)
        thread = collector._thread
        self.assertIsNotNone(thread)
        self.assertTrue(thread.is_alive())
        self.assertIs(collector._callback, None)
        # 生存スレッドが残っている限り再起動は True のまま
        self.assertTrue(collector.start())
        self.assertIs(collector._thread, thread)
        # 死亡を確認できた時点で所有権が解放され再起動できる
        thread.mark_dead()
        collector.stop()
        self.assertIsNone(collector._thread)
        collector._thread_factory = lambda: _FakeThread()
        self.assertTrue(collector.start())
        self.assertTrue(collector.is_running)

    def test_thread_start_failure_returns_false(self) -> None:
        collector, _ = self._make_collector()

        class _BadStartThread(_FakeThread):
            def start(self) -> None:
                raise RuntimeError("secret start detail")

        collector._thread_factory = lambda: _BadStartThread()
        started = collector.start()
        self.assertFalse(started)
        self.assertFalse(collector.is_running)
        self.assertIsNone(collector._thread)
        self.assertIsNone(collector._callback)

    def test_immediate_death_returns_false_and_clears(self) -> None:
        collector, _ = self._make_collector()

        class _ImmediateDeathThread(_FakeThread):
            def start(self) -> None:
                self.started += 1
                self._alive = True
                raise RuntimeError("secret start detail")

            def is_alive(self) -> bool:
                # 起動直後に即死したと見なす
                return False

        collector._thread_factory = lambda: _ImmediateDeathThread()
        started = collector.start()
        self.assertFalse(started)
        self.assertFalse(collector.is_running)
        self.assertIsNone(collector._thread)
        self.assertIsNone(collector._callback)
        # 即死後は再起動できる
        collector._thread_factory = lambda: _FakeThread()
        self.assertTrue(collector.start())
        self.assertTrue(collector.is_running)

    def test_immediate_death_after_normal_start_returns_false_and_clears(self) -> None:
        collector, _ = self._make_collector()

        class _ImmediateDeathThread(_FakeThread):
            def is_alive(self) -> bool:
                return False

        collector._thread_factory = lambda: _ImmediateDeathThread()
        started = collector.start()
        self.assertFalse(started)
        self.assertFalse(collector.is_running)
        self.assertIsNone(collector._thread)
        self.assertIsNone(collector._callback)

    def test_join_failure_retains_thread_without_raw_exception(self) -> None:
        collector, threads = self._make_collector()
        collector.start()

        def bad_join(timeout=None):
            raise RuntimeError("secret join detail")

        threads[0].join = bad_join
        collector.stop()  # 生例外を露出しない
        self.assertFalse(collector.is_running)
        self.assertTrue(collector.stop_pending)
        self.assertIs(collector._thread, threads[0])

    def test_injectable_collect_fn_is_used(self) -> None:
        collector, _ = self._make_collector()
        metrics = collector._collect_fn()
        self.assertEqual(metrics.process_count, 3)


class ProcessDetailsStoragePolicyTest(unittest.TestCase):
    """ストレージの書込・読出し境界でのプロセス詳細 redact を固定。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = str(Path(self._tmp.name) / "metrics.db")

    def _make_storage(self, enabled: bool) -> MetricsStorage:
        storage = MetricsStorage(db_path=self.db_path, process_details_enabled=enabled)
        storage.initialize()
        self.addCleanup(storage.close)
        return storage

    def _insert_legacy_row_with_details(self, storage: MetricsStorage) -> None:
        detail = json.dumps([{"name": "legacy-app", "pid": 999, "cpu_percent": 5.0}])
        with storage._cursor() as cur:
            cur.execute(
                "INSERT INTO metrics (timestamp, process_count, top_processes) VALUES (?, ?, ?)",
                (time.time(), 5, detail),
            )

    def test_legacy_row_with_details_redacted_when_disabled(self) -> None:
        storage = self._make_storage(enabled=False)
        self._insert_legacy_row_with_details(storage)
        row = storage.get_latest_row()
        self.assertIsNotNone(row)
        self.assertEqual(row["top_processes"], [])
        self.assertEqual(row["process_count"], 5)

    def test_legacy_row_with_details_exposed_when_enabled(self) -> None:
        storage = self._make_storage(enabled=True)
        self._insert_legacy_row_with_details(storage)
        row = storage.get_latest_row()
        self.assertIsNotNone(row)
        self.assertEqual(
            row["top_processes"], [{"name": "legacy-app", "pid": 999, "cpu_percent": 5.0}]
        )

    def test_store_metrics_never_writes_details_when_disabled(self) -> None:
        storage = self._make_storage(enabled=False)
        metrics = SystemMetrics(timestamp=time.time(), process_count=3)
        metrics.top_cpu_processes = [{"name": "secret-app", "pid": 1, "cpu_percent": 9.0}]
        storage.store_metrics(metrics)
        with storage._cursor() as cur:
            cur.execute("SELECT top_processes FROM metrics ORDER BY id DESC LIMIT 1")
            raw = cur.fetchone()[0]
        self.assertEqual(json.loads(raw), [])
        self.assertEqual(storage.get_latest_row()["top_processes"], [])

    def test_store_metrics_writes_details_when_enabled(self) -> None:
        storage = self._make_storage(enabled=True)
        metrics = SystemMetrics(timestamp=time.time(), process_count=3)
        metrics.top_cpu_processes = [{"name": "secret-app", "pid": 1, "cpu_percent": 9.0}]
        storage.store_metrics(metrics)
        row = storage.get_latest_row()
        self.assertEqual(
            row["top_processes"], [{"name": "secret-app", "pid": 1, "cpu_percent": 9.0}]
        )


class ProcessDetailsMonitorContextPolicyTest(unittest.TestCase):
    """MonitorContext の opt-in 解決 (default / false / true) を固定。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = str(Path(self._tmp.name) / "metrics.db")

    def _make(self, env: dict[str, str] | None = None, explicit: bool | None = None):
        with mock.patch.dict(os.environ, env or {}, clear=True):
            with mock.patch.object(collector_mod, "HAS_PSUTIL", True):
                return MonitorContext(
                    db_path=self.db_path, process_details_enabled=explicit
                )

    def test_default_env_is_fail_closed(self) -> None:
        ctx = self._make()
        self.assertFalse(ctx.process_details_enabled)
        self.assertFalse(ctx.collector._process_details_enabled)
        self.assertFalse(ctx.storage._process_details_enabled)

    def test_env_false_is_fail_closed(self) -> None:
        ctx = self._make(env={"SENSOR_PROCESS_DETAILS_ENABLED": "false"})
        self.assertFalse(ctx.process_details_enabled)
        self.assertFalse(ctx.collector._process_details_enabled)
        self.assertFalse(ctx.storage._process_details_enabled)

    def test_env_true_enables(self) -> None:
        ctx = self._make(env={"SENSOR_PROCESS_DETAILS_ENABLED": "true"})
        self.assertTrue(ctx.process_details_enabled)
        self.assertTrue(ctx.collector._process_details_enabled)
        self.assertTrue(ctx.storage._process_details_enabled)

    def test_explicit_false_overrides_env_true(self) -> None:
        ctx = self._make(env={"SENSOR_PROCESS_DETAILS_ENABLED": "true"}, explicit=False)
        self.assertFalse(ctx.process_details_enabled)
        self.assertFalse(ctx.collector._process_details_enabled)
        self.assertFalse(ctx.storage._process_details_enabled)


class ProcessNameReaderErrorPrivacyTest(unittest.TestCase):
    """_process_name_reader の例外メッセージに PID を含めないことを固定。"""

    def test_unavailable_error_contains_no_pid(self) -> None:
        from src.perception.sources import (
            ActivitySourceUnavailableError,
            _process_name_reader,
        )

        errors = {
            "NoSuchProcess": type("NoSuchProcess", (Exception,), {}),
            "AccessDenied": type("AccessDenied", (Exception,), {}),
            "ZombieProcess": type("ZombieProcess", (Exception,), {}),
        }

        def process(pid: int):
            raise errors["NoSuchProcess"](pid)

        fake_psutil = SimpleNamespace(
            Process=process,
            NoSuchProcess=errors["NoSuchProcess"],
            AccessDenied=errors["AccessDenied"],
            ZombieProcess=errors["ZombieProcess"],
        )
        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
            with self.assertRaises(ActivitySourceUnavailableError) as cm:
                _process_name_reader(98765)
        self.assertNotIn("98765", str(cm.exception))
        self.assertNotIn("pid", str(cm.exception).lower())


class MetricsStorageLifecycleErrorPrivacyTest(unittest.TestCase):
    """ストレージライフサイクル障害から DBパス・SQLiteメッセージが漏れないことを固定。

    実プロダクションDBには触れず、一時ディレクトリ + モック注入のみを使う。
    """

    CANARY_PATH = "C:/canary/secret/metrics.db"
    CANARY_MSG = "canary sqlite detail"

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = str(Path(self._tmp.name) / "metrics.db")

    def test_initialize_failure_never_leaks_db_path(self) -> None:
        storage = MetricsStorage(db_path=self.db_path, process_details_enabled=False)
        with mock.patch(
            "sqlite3.connect",
            side_effect=sqlite3.OperationalError(
                f"unable to open database file: {self.CANARY_PATH}"
            ),
        ):
            with self.assertRaises(MetricsStorageError) as cm:
                storage.initialize()
        msg = str(cm.exception)
        self.assertIn("OperationalError", msg)
        self.assertNotIn(self.CANARY_PATH, msg)
        self.assertNotIn("unable to open", msg)

    def test_operation_failure_never_leaks_sqlite_message(self) -> None:
        storage = MetricsStorage(db_path=self.db_path, process_details_enabled=False)
        self.addCleanup(storage.close)
        fake_cursor = mock.Mock()
        fake_cursor.execute.side_effect = sqlite3.OperationalError(self.CANARY_MSG)
        fake_conn = mock.Mock()
        fake_conn.cursor.return_value = fake_cursor
        storage._conn = fake_conn
        with self.assertRaises(MetricsStorageError) as cm:
            storage.get_record_count()
        msg = str(cm.exception)
        self.assertIn("OperationalError", msg)
        self.assertNotIn(self.CANARY_MSG, msg)

    def test_not_initialized_uses_fixed_message_without_path(self) -> None:
        storage = MetricsStorage(db_path=self.db_path, process_details_enabled=False)
        with self.assertRaises(MetricsStorageError) as cm:
            storage.get_record_count()
        self.assertIn("not initialized", str(cm.exception))
        self.assertNotIn(str(storage.db_path), str(cm.exception))


class MonitorContextDiagnosticsPrivacyTest(unittest.TestCase):
    """MonitorContext の起動/コールバック異常から DBパス・SQLiteメッセージ・
    プロセス詳細が漏れないことを固定。実プロセス・実DBには触れない。"""

    CANARY_PATH = "C:/canary/secret/metrics.db"
    CANARY_MSG = "canary sqlite detail"

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = str(Path(self._tmp.name) / "metrics.db")

    def _make_ctx(self, storage: mock.Mock) -> MonitorContext:
        with mock.patch.object(collector_mod, "HAS_PSUTIL", True):
            ctx = MonitorContext(
                db_path=self.db_path,
                process_details_enabled=False,
                sleep_fn=lambda _delay: None,
            )
        ctx.storage = storage
        return ctx

    def test_start_failure_prints_type_only_no_db_path(self) -> None:
        storage = mock.Mock(spec=MetricsStorage)
        storage.initialize.side_effect = sqlite3.OperationalError(
            f"unable to open database file: {self.CANARY_PATH}"
        )
        ctx = self._make_ctx(storage)
        with redirect_stdout(io.StringIO()) as buf:
            started = ctx.start()
        out = buf.getvalue()
        self.assertFalse(started)
        self.assertIn("OperationalError", out)
        self.assertNotIn(self.CANARY_PATH, out)
        self.assertNotIn("unable to open", out)

    def test_callback_failure_prints_type_only_no_sqlite_message(self) -> None:
        storage = mock.Mock(spec=MetricsStorage)
        storage.store_metrics.side_effect = sqlite3.OperationalError(self.CANARY_MSG)
        ctx = self._make_ctx(storage)
        with redirect_stdout(io.StringIO()) as buf:
            ctx._on_metrics(SystemMetrics(timestamp=time.time(), process_count=1))
        out = buf.getvalue()
        self.assertIn("OperationalError", out)
        self.assertNotIn(self.CANARY_MSG, out)

    def test_callback_periodic_notice_stays_type_only(self) -> None:
        storage = mock.Mock(spec=MetricsStorage)
        storage.store_metrics.side_effect = RuntimeError(self.CANARY_MSG)
        ctx = self._make_ctx(storage)
        with redirect_stdout(io.StringIO()) as buf:
            for _ in range(100):
                ctx._on_metrics(SystemMetrics(timestamp=time.time(), process_count=1))
        out = buf.getvalue()
        self.assertEqual(out.count("RuntimeError"), 2)
        self.assertNotIn(self.CANARY_MSG, out)

    def _assert_cp932_safe(self, text: str) -> None:
        text.encode("cp932")  # エンコード不可なら UnicodeEncodeError

    def test_collector_start_failure_cleans_up_collector_and_storage(self) -> None:
        storage = mock.Mock(spec=MetricsStorage)
        ctx = self._make_ctx(storage)
        with mock.patch.object(ctx.collector, "start", return_value=False) as fake_start, \
                mock.patch.object(ctx.collector, "stop") as fake_stop:
            started = ctx.start()
        self.assertFalse(started)
        self.assertFalse(ctx.is_running)
        fake_start.assert_called_once_with(callback=ctx._on_metrics)
        fake_stop.assert_called_once()
        storage.close.assert_called_once()

    def test_storage_init_failure_still_stops_collector_and_closes_storage(self) -> None:
        storage = mock.Mock(spec=MetricsStorage)
        storage.initialize.side_effect = sqlite3.OperationalError("boom")
        ctx = self._make_ctx(storage)
        with mock.patch.object(ctx.collector, "start") as fake_start, \
                mock.patch.object(ctx.collector, "stop") as fake_stop:
            started = ctx.start()
        self.assertFalse(started)
        fake_start.assert_not_called()
        fake_stop.assert_called_once()
        storage.close.assert_called_once()

    def test_start_failure_diagnostic_is_cp932_safe_type_only(self) -> None:
        storage = mock.Mock(spec=MetricsStorage)
        storage.initialize.side_effect = sqlite3.OperationalError(
            f"unable to open database file: {self.CANARY_PATH} ({self.CANARY_MSG})"
        )
        ctx = self._make_ctx(storage)
        with redirect_stdout(io.StringIO()) as buf:
            started = ctx.start()
        out = buf.getvalue()
        self.assertFalse(started)
        self._assert_cp932_safe(out)
        self.assertNotIn("⚠", out)
        self.assertNotIn(self.CANARY_PATH, out)
        self.assertNotIn(self.CANARY_MSG, out)
        self.assertNotIn("unable to open", out)
        self.assertEqual(out.count("OperationalError"), 1)

    def test_collector_unavailable_diagnostic_is_cp932_safe(self) -> None:
        storage = mock.Mock(spec=MetricsStorage)
        ctx = self._make_ctx(storage)
        with mock.patch.object(ctx.collector, "start", return_value=False):
            with redirect_stdout(io.StringIO()) as buf:
                started = ctx.start()
        out = buf.getvalue()
        self.assertFalse(started)
        self._assert_cp932_safe(out)
        self.assertNotIn("⚠", out)
        self.assertNotIn(self.CANARY_PATH, out)
        self.assertIn("collector unavailable", out)

    def test_callback_diagnostic_is_cp932_safe_type_only(self) -> None:
        storage = mock.Mock(spec=MetricsStorage)
        storage.store_metrics.side_effect = RuntimeError(self.CANARY_MSG)
        ctx = self._make_ctx(storage)
        with redirect_stdout(io.StringIO()) as buf:
            ctx._on_metrics(SystemMetrics(timestamp=time.time(), process_count=1))
        out = buf.getvalue()
        self._assert_cp932_safe(out)
        self.assertNotIn("⚠", out)
        self.assertNotIn(self.CANARY_MSG, out)
        self.assertIn("RuntimeError", out)


class MonitorContextStopTruthfulTest(unittest.TestCase):
    """MonitorContext.stop の順序契約を固定。

    - stop は collector を先に停止し、worker 生存中 (stop_pending) は storage を
      開いたまま所有する (生存 worker のコールバックが storage を書き続けるため)
    - stop 要求後 is_running は即時 False (worker が生存していても)
    - 死を確認できた繰り返し stop でのみ storage を閉じる
    - status は固定 bool (running / thread_alive / stop_pending) を公開する
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = str(Path(self._tmp.name) / "metrics.db")

    def _ctx(self) -> MonitorContext:
        with mock.patch.object(collector_mod, "HAS_PSUTIL", True):
            ctx = MonitorContext(db_path=self.db_path, process_details_enabled=False)
        return ctx

    class _LiveThread:
        alive = True

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout=None) -> None:
            pass

    def _live_collector(self, ctx: MonitorContext):
        t = self._LiveThread()
        ctx.collector._thread = t
        ctx.collector._running = True
        ctx._running = True
        return t

    def test_stop_keeps_storage_open_while_collector_thread_alive(self) -> None:
        ctx = self._ctx()
        self._live_collector(ctx)
        self.assertTrue(ctx.is_running)
        with mock.patch.object(ctx.storage, "close") as close:
            ctx.stop()
        # collector を先に停止。worker 生存中は storage を閉じない。
        close.assert_not_called()
        self.assertFalse(ctx.is_running)
        self.assertTrue(ctx.collector.thread_alive)
        self.assertTrue(ctx.collector.stop_pending)

    def test_repeated_stop_closes_storage_only_after_worker_death(self) -> None:
        ctx = self._ctx()
        t = self._live_collector(ctx)
        with mock.patch.object(ctx.storage, "close") as close:
            ctx.stop()
            close.assert_not_called()
            self.assertTrue(ctx.collector.stop_pending)
            t.alive = False
            ctx.stop()  # 死を確認できた繰り返し stop でのみ storage を閉じる
            close.assert_called_once()
            self.assertFalse(ctx.collector.stop_pending)
            self.assertIsNone(ctx.collector._thread)
            self.assertFalse(ctx.is_running)

    def test_is_running_false_immediately_after_stop(self) -> None:
        ctx = self._ctx()
        self._live_collector(ctx)
        self.assertTrue(ctx.is_running)
        ctx.stop()
        self.assertFalse(ctx.is_running)
        self.assertTrue(ctx.collector.thread_alive)

    def test_status_exposes_fixed_booleans(self) -> None:
        ctx = self._ctx()
        self._live_collector(ctx)
        status = ctx.get_status()
        self.assertTrue(status["running"])
        self.assertTrue(status["thread_alive"])
        self.assertFalse(status["stop_pending"])
        ctx.stop()
        status = ctx.get_status()
        self.assertFalse(status["running"])
        self.assertTrue(status["thread_alive"])
        self.assertTrue(status["stop_pending"])


class MonitorContextStartAdmissionTest(unittest.TestCase):
    """MonitorContext.start の受付前検査を固定。

    - 既に実稼働中 (collector.is_running) は storage に触れず冪等に True
    - 旧 worker が生存 (thread_alive かつ stop_pending) の間は、storage を触らず・
      置き換えず・閉じずに False (reactivate しない)
    - 死を確認できた繰り返し stop で元の storage を閉じ、後の start が同じ storage を
      initialize し直して収集を再開できる
    実スレッド・実プロセス・実DBには触れず、フェイク collector とモック storage のみを使う。
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = str(Path(self._tmp.name) / "metrics.db")

    def _ctx(self, collector=None, storage=None) -> MonitorContext:
        with mock.patch.object(collector_mod, "HAS_PSUTIL", True):
            ctx = MonitorContext(db_path=self.db_path, process_details_enabled=False)
        if collector is not None:
            ctx.collector = collector
        if storage is not None:
            ctx.storage = storage
        return ctx

    class _FakeCollector:
        def __init__(
            self,
            *,
            is_running: bool = False,
            thread_alive: bool = False,
            stop_pending: bool = False,
            start_result: bool = True,
        ) -> None:
            self._is_running = is_running
            self._thread_alive = thread_alive
            self._stop_pending = stop_pending
            self._start_result = start_result
            self.starts = 0
            self.stops = 0

        @property
        def is_running(self) -> bool:
            return self._is_running

        @property
        def thread_alive(self) -> bool:
            return self._thread_alive

        @property
        def stop_pending(self) -> bool:
            return self._stop_pending

        def start(self, callback=None) -> bool:
            self.starts += 1
            if self._start_result:
                self._is_running = True
                self._thread_alive = True
                self._stop_pending = False
            return self._start_result

        def stop(self) -> None:
            self.stops += 1
            self._is_running = False
            if not self._thread_alive:
                self._stop_pending = False
            else:
                self._stop_pending = True

        def mark_dead(self) -> None:
            self._thread_alive = False

    def test_already_running_returns_true_without_touching_storage(self) -> None:
        collector = self._FakeCollector(is_running=True, thread_alive=True)
        storage = mock.Mock(spec=MetricsStorage)
        ctx = self._ctx(collector=collector, storage=storage)
        self.assertTrue(ctx.start())
        # storage にも collector にも触れない (冪等)。
        storage.initialize.assert_not_called()
        storage.close.assert_not_called()
        self.assertEqual(collector.starts, 0)
        self.assertTrue(ctx.is_running)
        self.assertIs(ctx.storage, storage)

    def test_stop_pending_live_worker_returns_false_without_touching_storage(self) -> None:
        collector = self._FakeCollector(thread_alive=True, stop_pending=True)
        storage = mock.Mock(spec=MetricsStorage)
        ctx = self._ctx(collector=collector, storage=storage)
        self.assertFalse(ctx.start())
        # storage を触らず・置き換えず・閉じず、collector も動かさない。
        storage.initialize.assert_not_called()
        storage.close.assert_not_called()
        self.assertEqual(collector.starts, 0)
        self.assertEqual(collector.stops, 0)
        self.assertFalse(ctx.is_running)
        self.assertIs(ctx.storage, storage)

    def test_restart_reinitializes_original_storage_after_death(self) -> None:
        collector = self._FakeCollector()
        storage = mock.Mock(spec=MetricsStorage)
        ctx = self._ctx(collector=collector, storage=storage)
        # 初回 start: initialize → collector.start
        self.assertTrue(ctx.start())
        storage.initialize.assert_called_once()
        self.assertEqual(collector.starts, 1)
        self.assertTrue(ctx.is_running)
        # stop: 生存 worker の間は storage を閉じない。
        ctx.stop()
        storage.close.assert_not_called()
        self.assertTrue(collector.stop_pending)
        self.assertTrue(collector.thread_alive)
        # 死を確認できた繰り返し stop で元の storage を閉じる (置き換えない)。
        collector.mark_dead()
        ctx.stop()
        storage.close.assert_called_once()
        self.assertIs(ctx.storage, storage)
        # 後の start で同じ storage を initialize し直し、収集を再開できる。
        self.assertTrue(ctx.start())
        self.assertEqual(storage.initialize.call_count, 2)
        self.assertEqual(collector.starts, 2)
        self.assertTrue(ctx.is_running)

    def test_start_call_order_initialize_before_collector_start(self) -> None:
        collector = self._FakeCollector()
        storage = mock.Mock(spec=MetricsStorage)
        ctx = self._ctx(collector=collector, storage=storage)
        order: list[str] = []
        storage.initialize.side_effect = lambda: order.append("storage.initialize")
        original_start = collector.start

        def recorded_start(callback=None) -> bool:
            order.append("collector.start")
            return original_start(callback=callback)

        collector.start = recorded_start
        self.assertTrue(ctx.start())
        self.assertEqual(order, ["storage.initialize", "collector.start"])


class MonitorContextRetryPolicyTest(unittest.TestCase):
    """MonitorContext._on_metrics の有限再試行と dropped-write カウンタを固定。

    - 同じ metrics オブジェクトを再試行し、成功したらその時点で止める
    - 全試行失敗 (exhaustion) のときだけ固定カウンタを増やし、型のみ・ASCII で通知
    - 再試行は write_attempts 回までで、間には短い有界バックオフを挟む
    - キュー・バッファは持たず、メモリ増加もない
    - get_status は安全なカウンタ (write_attempts / dropped_writes) のみを公開
    実DB・実スレッド・実sleepには触れず、モック storage と注入 sleep のみを使う。
    """

    CANARY_PATH = "C:/canary/secret/metrics.db"
    CANARY_MSG = "canary sqlite detail"

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = str(Path(self._tmp.name) / "metrics.db")

    def _make_ctx(
        self,
        storage: mock.Mock,
        *,
        attempts: int = 3,
        retry_delay: float = 0.25,
    ) -> tuple[MonitorContext, list[float]]:
        sleeps: list[float] = []

        def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        with mock.patch.object(collector_mod, "HAS_PSUTIL", True):
            ctx = MonitorContext(
                db_path=self.db_path,
                process_details_enabled=False,
                write_attempts=attempts,
                write_retry_delay=retry_delay,
                sleep_fn=fake_sleep,
            )
        ctx.storage = storage
        return ctx, sleeps

    def _metrics(self) -> SystemMetrics:
        return SystemMetrics(timestamp=time.time(), process_count=1)

    def test_fail_twice_then_success_retries_same_object_and_stops(self) -> None:
        storage = mock.Mock(spec=MetricsStorage)
        storage.store_metrics.side_effect = [
            RuntimeError("first"),
            RuntimeError("second"),
            None,
        ]
        ctx, sleeps = self._make_ctx(storage)
        ctx._on_metrics(self._metrics())
        # 同じ metrics を3回試行し、成功したら止まる。
        self.assertEqual(storage.store_metrics.call_count, 3)
        self.assertEqual(sleeps, [0.25, 0.25])  # 成功後の sleep はない
        self.assertEqual(ctx._dropped_write_count, 0)
        self.assertEqual(ctx._db_error_count, 0)

    def test_exhaustion_increments_dropped_counter_and_type_only_log(self) -> None:
        storage = mock.Mock(spec=MetricsStorage)
        storage.store_metrics.side_effect = sqlite3.OperationalError(self.CANARY_MSG)
        ctx, sleeps = self._make_ctx(storage, attempts=3, retry_delay=0.5)
        with redirect_stdout(io.StringIO()) as buf:
            ctx._on_metrics(self._metrics())
            ctx._on_metrics(self._metrics())
        out = buf.getvalue()
        # 全試行失敗時のみカウンタが増える。
        self.assertEqual(storage.store_metrics.call_count, 6)
        self.assertEqual(sleeps, [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(ctx._dropped_write_count, 2)
        self.assertEqual(ctx._db_error_count, 2)
        self.assertEqual(out.count("OperationalError"), 1)
        self.assertIn("dropped", out)
        self.assertIn("attempts=3", out)
        self.assertNotIn(self.CANARY_MSG, out)

    def test_no_sleep_after_success_on_first_attempt(self) -> None:
        storage = mock.Mock(spec=MetricsStorage)
        ctx, sleeps = self._make_ctx(storage)
        ctx._on_metrics(self._metrics())
        self.assertEqual(storage.store_metrics.call_count, 1)
        self.assertEqual(sleeps, [])
        self.assertEqual(ctx._dropped_write_count, 0)
        self.assertEqual(ctx._db_error_count, 0)

    def test_single_attempt_configuration_never_sleeps(self) -> None:
        storage = mock.Mock(spec=MetricsStorage)
        storage.store_metrics.side_effect = RuntimeError(self.CANARY_MSG)
        ctx, sleeps = self._make_ctx(storage, attempts=1, retry_delay=0.5)
        with redirect_stdout(io.StringIO()) as buf:
            ctx._on_metrics(self._metrics())
        self.assertEqual(storage.store_metrics.call_count, 1)
        self.assertEqual(sleeps, [])
        self.assertEqual(ctx._dropped_write_count, 1)
        self.assertIn("RuntimeError", buf.getvalue())

    def test_exhaustion_log_is_ascii_type_only_no_path_or_detail(self) -> None:
        storage = mock.Mock(spec=MetricsStorage)
        storage.store_metrics.side_effect = sqlite3.OperationalError(
            f"unable to open database file: {self.CANARY_PATH} ({self.CANARY_MSG})"
        )
        ctx, _ = self._make_ctx(storage)
        with redirect_stdout(io.StringIO()) as buf:
            ctx._on_metrics(self._metrics())
        out = buf.getvalue()
        self.assertIn("OperationalError", out)
        self.assertNotIn(self.CANARY_PATH, out)
        self.assertNotIn(self.CANARY_MSG, out)
        self.assertNotIn("unable to open", out)
        out.encode("cp932")  # ASCII / cp932 でエンコード可能
        self.assertEqual(ctx._dropped_write_count, 1)

    def test_status_exposes_only_safe_counters(self) -> None:
        storage = mock.Mock(spec=MetricsStorage)
        storage.store_metrics.side_effect = RuntimeError(self.CANARY_MSG)
        ctx, _ = self._make_ctx(storage, attempts=5)
        ctx._on_metrics(self._metrics())
        status = ctx.get_status()
        self.assertEqual(status["write_attempts"], 5)
        self.assertEqual(status["dropped_writes"], 1)
        for forbidden in ("path", "last_error", self.CANARY_MSG):
            self.assertNotIn(forbidden, status)


if __name__ == "__main__":
    unittest.main()
