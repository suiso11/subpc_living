from __future__ import annotations

import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from src.perception.activity import ActivitySample, VALID_APP_CATEGORIES
from src.perception.sources import (
    ActivitySource,
    ActivitySourceUnavailableError,
    LinuxActivitySource,
    ProcessNameClassifier,
    WindowsActivitySource,
    _linux_foreground_pid_reader,
    _linux_idle_millis_reader,
    _process_name_reader,
    _windows_foreground_pid_reader,
    _windows_idle_millis_reader,
    create_activity_source,
)
from src.perception import (
    LinuxActivitySource as PkgLinuxActivitySource,
    ProcessNameClassifier as PkgProcessNameClassifier,
    WindowsActivitySource as PkgWindowsActivitySource,
    create_activity_source as pkg_create_activity_source,
)


class ProcessNameClassifierTest(unittest.TestCase):
    def test_known_names_map_to_categories(self) -> None:
        c = ProcessNameClassifier()
        self.assertEqual(c.classify("code"), "work")
        self.assertEqual(c.classify("discord"), "communication")
        self.assertEqual(c.classify("spotify"), "media")
        self.assertEqual(c.classify("explorer"), "system")

    def test_case_insensitive_and_exe_stripped(self) -> None:
        c = ProcessNameClassifier()
        self.assertEqual(c.classify("Discord.EXE"), "communication")
        self.assertEqual(c.classify("CODE.exe"), "work")
        self.assertEqual(c.classify("  vlc  "), "media")

    def test_unmatched_returns_unknown(self) -> None:
        c = ProcessNameClassifier()
        self.assertEqual(c.classify("completely-unknown-app"), "unknown")
        self.assertEqual(c.classify("chrome"), "unknown")
        self.assertEqual(c.classify("firefox"), "unknown")

    def test_non_str_returns_unknown(self) -> None:
        self.assertEqual(ProcessNameClassifier().classify(None), "unknown")

    def test_custom_mapping(self) -> None:
        c = ProcessNameClassifier({"myapp": "work"})
        self.assertEqual(c.classify("myapp"), "work")
        self.assertEqual(c.classify("other"), "unknown")

    def test_invalid_category_in_mapping_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ProcessNameClassifier({"myapp": "gaming"})

    def test_builtin_categories_are_valid(self) -> None:
        c = ProcessNameClassifier()
        for key in c._table:
            self.assertIn(c._table[key], VALID_APP_CATEGORIES)


class _ReaderSink:
    def __init__(self, *, idle_ms: float = 1500.0, pid: int = 100, name: str = "code") -> None:
        self.idle_ms = idle_ms
        self.pid = pid
        self.name = name
        self.calls: list[str] = []

    def idle(self) -> float:
        self.calls.append("idle")
        return self.idle_ms

    def pid_of(self) -> int:
        self.calls.append("pid")
        return self.pid

    def name_of(self, pid: int) -> str:
        self.calls.append("name")
        return self.name


class IdleConversionTest(unittest.TestCase):
    def test_millis_converted_to_seconds(self) -> None:
        sink = _ReaderSink(idle_ms=1500.0, pid=7, name="code")
        src = WindowsActivitySource(
            idle_millis_reader=sink.idle,
            foreground_pid_reader=sink.pid_of,
            process_name_reader=sink.name_of,
            clock=lambda: 42.0,
        )
        sample = src.sample()
        self.assertEqual(sample.idle_seconds, 1.5)
        self.assertEqual(sample.timestamp, 42.0)
        self.assertEqual(sample.app_category, "work")

    def test_zero_idle(self) -> None:
        sink = _ReaderSink(idle_ms=0.0)
        src = LinuxActivitySource(
            idle_millis_reader=sink.idle,
            foreground_pid_reader=sink.pid_of,
            process_name_reader=sink.name_of,
            clock=lambda: 1.0,
        )
        self.assertEqual(src.sample().idle_seconds, 0.0)


class AdapterClassificationTest(unittest.TestCase):
    def test_process_name_mapped_to_category(self) -> None:
        for name, category in (
            ("discord.exe", "communication"),
            ("Code.exe", "work"),
            ("spotify", "media"),
            ("explorer.exe", "system"),
        ):
            sink = _ReaderSink(name=name)
            src = WindowsActivitySource(
                idle_millis_reader=sink.idle,
                foreground_pid_reader=sink.pid_of,
                process_name_reader=sink.name_of,
                clock=lambda: 1.0,
            )
            self.assertEqual(src.sample().app_category, category, name)

    def test_unmatched_process_name_is_unknown(self) -> None:
        sink = _ReaderSink(name="chrome")
        src = WindowsActivitySource(
            idle_millis_reader=sink.idle,
            foreground_pid_reader=sink.pid_of,
            process_name_reader=sink.name_of,
            clock=lambda: 1.0,
        )
        self.assertEqual(src.sample().app_category, "unknown")

    def test_custom_classifier_is_used(self) -> None:
        sink = _ReaderSink(name="anything")
        src = LinuxActivitySource(
            idle_millis_reader=sink.idle,
            foreground_pid_reader=sink.pid_of,
            process_name_reader=sink.name_of,
            classifier=ProcessNameClassifier({"anything": "media"}),
            clock=lambda: 1.0,
        )
        self.assertEqual(src.sample().app_category, "media")


class ConstructionDoesNotCollectTest(unittest.TestCase):
    def test_windows_construction_does_not_call_readers(self) -> None:
        sink = _ReaderSink()
        WindowsActivitySource(
            idle_millis_reader=sink.idle,
            foreground_pid_reader=sink.pid_of,
            process_name_reader=sink.name_of,
        )
        self.assertEqual(sink.calls, [])

    def test_linux_construction_does_not_call_readers(self) -> None:
        sink = _ReaderSink()
        LinuxActivitySource(
            idle_millis_reader=sink.idle,
            foreground_pid_reader=sink.pid_of,
            process_name_reader=sink.name_of,
        )
        self.assertEqual(sink.calls, [])

    def test_factory_construction_does_not_call_readers(self) -> None:
        sink = _ReaderSink()
        create_activity_source(
            "windows",
            idle_millis_reader=sink.idle,
            foreground_pid_reader=sink.pid_of,
            process_name_reader=sink.name_of,
        )
        self.assertEqual(sink.calls, [])

    def test_default_construction_on_host_platform_is_safe(self) -> None:
        WindowsActivitySource()
        LinuxActivitySource()
        create_activity_source("windows")
        create_activity_source("linux")


class PlatformFactoryDispatchTest(unittest.TestCase):
    def test_explicit_platforms(self) -> None:
        self.assertIsInstance(create_activity_source("windows"), WindowsActivitySource)
        self.assertIsInstance(create_activity_source("linux"), LinuxActivitySource)

    def test_platform_aliases(self) -> None:
        self.assertIsInstance(create_activity_source("win32"), WindowsActivitySource)
        self.assertIsInstance(create_activity_source("nt"), WindowsActivitySource)
        self.assertIsInstance(create_activity_source("linux2"), LinuxActivitySource)
        self.assertIsInstance(create_activity_source("WIN32"), WindowsActivitySource)

    def test_unsupported_explicit_platform_raises_value_error(self) -> None:
        for name in ("darwin", "macos", "freebsd", "ios"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    create_activity_source(name)

    def test_empty_platform_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            create_activity_source("   ")
        with self.assertRaises(ValueError):
            create_activity_source("")

    def test_auto_detect_windows(self) -> None:
        with mock.patch.object(sys, "platform", "win32"):
            self.assertIsInstance(create_activity_source(), WindowsActivitySource)

    def test_auto_detect_linux(self) -> None:
        with mock.patch.object(sys, "platform", "linux"):
            self.assertIsInstance(create_activity_source(), LinuxActivitySource)

    def test_auto_detect_unsupported_raises_unavailable(self) -> None:
        with mock.patch.object(sys, "platform", "darwin"):
            with self.assertRaises(ActivitySourceUnavailableError):
                create_activity_source()

    def test_custom_readers_flow_through_factory(self) -> None:
        sink = _ReaderSink()
        src = create_activity_source(
            "windows",
            idle_millis_reader=sink.idle,
            foreground_pid_reader=sink.pid_of,
            process_name_reader=sink.name_of,
        )
        sample = src.sample()
        self.assertEqual(sample.app_category, "work")
        self.assertEqual(sample.idle_seconds, 1.5)


class UnavailableFacilityTest(unittest.TestCase):
    def test_windows_readers_fail_without_windll(self) -> None:
        with mock.patch("src.perception.sources._windll", return_value=None):
            with self.assertRaises(ActivitySourceUnavailableError):
                _windows_idle_millis_reader()
            with self.assertRaises(ActivitySourceUnavailableError):
                _windows_foreground_pid_reader()

    def test_windows_idle_reader_fails_when_api_reports_failure(self) -> None:
        windll = _fake_windll(get_last_input_ok=0)
        with mock.patch("src.perception.sources._windll", return_value=windll):
            with self.assertRaises(ActivitySourceUnavailableError):
                _windows_idle_millis_reader()

    def test_windows_idle_reader_returns_tick_diff(self) -> None:
        windll = _fake_windll(last_dw_time=1000, tick_now=6000)
        with mock.patch("src.perception.sources._windll", return_value=windll):
            self.assertEqual(_windows_idle_millis_reader(), 5000.0)

    def test_windows_pid_reader_fails_without_foreground_window(self) -> None:
        windll = _fake_windll(hwnd=0)
        with mock.patch("src.perception.sources._windll", return_value=windll):
            with self.assertRaises(ActivitySourceUnavailableError):
                _windows_foreground_pid_reader()

    def test_windows_pid_reader_returns_pid(self) -> None:
        windll = _fake_windll(hwnd=0x1234, fg_pid=4242)
        with mock.patch("src.perception.sources._windll", return_value=windll):
            self.assertEqual(_windows_foreground_pid_reader(), 4242)

    def test_linux_idle_reader_parses_millis(self) -> None:
        with mock.patch("src.perception.sources.subprocess.run") as run:
            run.return_value = SimpleNamespace(returncode=0, stdout="1500")
            self.assertEqual(_linux_idle_millis_reader(), 1500.0)

    def test_linux_idle_reader_fails_on_nonzero_exit(self) -> None:
        with mock.patch("src.perception.sources.subprocess.run") as run:
            run.return_value = SimpleNamespace(returncode=1, stdout="")
            with self.assertRaises(ActivitySourceUnavailableError):
                _linux_idle_millis_reader()

    def test_linux_idle_reader_fails_on_missing_binary(self) -> None:
        with mock.patch("src.perception.sources.subprocess.run") as run:
            run.side_effect = FileNotFoundError()
            with self.assertRaises(ActivitySourceUnavailableError):
                _linux_idle_millis_reader()

    def test_linux_idle_reader_fails_on_timeout(self) -> None:
        with mock.patch("src.perception.sources.subprocess.run") as run:
            run.side_effect = subprocess.TimeoutExpired(cmd=["xprintidle"], timeout=2)
            with self.assertRaises(ActivitySourceUnavailableError):
                _linux_idle_millis_reader()

    def test_linux_idle_reader_fails_on_non_numeric_output(self) -> None:
        with mock.patch("src.perception.sources.subprocess.run") as run:
            run.return_value = SimpleNamespace(returncode=0, stdout="abc")
            with self.assertRaises(ActivitySourceUnavailableError):
                _linux_idle_millis_reader()

    def test_linux_pid_reader_parses_pid(self) -> None:
        with mock.patch("src.perception.sources.subprocess.run") as run:
            run.return_value = SimpleNamespace(returncode=0, stdout="4242")
            self.assertEqual(_linux_foreground_pid_reader(), 4242)

    def test_linux_pid_reader_fails_on_invalid_pid(self) -> None:
        with mock.patch("src.perception.sources.subprocess.run") as run:
            run.return_value = SimpleNamespace(returncode=0, stdout="0")
            with self.assertRaises(ActivitySourceUnavailableError):
                _linux_foreground_pid_reader()

    def test_process_name_reader_resolves_name(self) -> None:
        fake_psutil = _fake_psutil(process_name="Code")
        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
            self.assertEqual(_process_name_reader(123), "Code")

    def test_process_name_reader_fails_on_no_such_process(self) -> None:
        fake_psutil = _fake_psutil(process_name=None)
        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
            with self.assertRaises(ActivitySourceUnavailableError):
                _process_name_reader(999)

    def test_adapter_propagates_reader_unavailable_error(self) -> None:
        def boom() -> float:
            raise ActivitySourceUnavailableError("boom")

        src = WindowsActivitySource(
            idle_millis_reader=boom,
            foreground_pid_reader=lambda: 1,
            process_name_reader=lambda pid: "code",
        )
        with self.assertRaises(ActivitySourceUnavailableError):
            src.sample()


class NoRawFieldsTest(unittest.TestCase):
    def test_sample_exact_fields(self) -> None:
        self.assertEqual(
            set(ActivitySample.__dataclass_fields__),
            {"timestamp", "idle_seconds", "app_category"},
        )
        for forbidden in ("app_name", "window_title", "title", "text", "path", "pid", "raw"):
            self.assertNotIn(forbidden, ActivitySample.__dataclass_fields__)

    def test_sample_returns_only_activity_sample(self) -> None:
        sink = _ReaderSink(name="discord.exe")
        src = WindowsActivitySource(
            idle_millis_reader=sink.idle,
            foreground_pid_reader=sink.pid_of,
            process_name_reader=sink.name_of,
            clock=lambda: 1.0,
        )
        result = src.sample()
        self.assertIsInstance(result, ActivitySample)
        self.assertEqual(set(result.__dataclass_fields__), {"timestamp", "idle_seconds", "app_category"})

    def test_process_name_not_retained_on_source(self) -> None:
        sink = _ReaderSink(name="discord.exe")
        src = WindowsActivitySource(
            idle_millis_reader=sink.idle,
            foreground_pid_reader=sink.pid_of,
            process_name_reader=sink.name_of,
            clock=lambda: 1.0,
        )
        src.sample()
        for attr in vars(src).values():
            self.assertNotIn("discord", str(attr))
            self.assertNotIn("discord.exe", str(attr))


class ProtocolConformanceTest(unittest.TestCase):
    def test_dummy_object_satisfies_activity_source_protocol(self) -> None:
        class Dummy:
            def sample(self) -> ActivitySample:
                return ActivitySample(timestamp=1.0, idle_seconds=0, app_category="unknown")

        self.assertIsInstance(Dummy(), ActivitySource)

    def test_adapters_satisfy_activity_source_protocol(self) -> None:
        self.assertIsInstance(WindowsActivitySource(), ActivitySource)
        self.assertIsInstance(LinuxActivitySource(), ActivitySource)


class PackageExportTest(unittest.TestCase):
    def test_exports_from_package(self) -> None:
        self.assertIs(PkgWindowsActivitySource, WindowsActivitySource)
        self.assertIs(PkgLinuxActivitySource, LinuxActivitySource)
        self.assertIs(PkgProcessNameClassifier, ProcessNameClassifier)
        self.assertIs(pkg_create_activity_source, create_activity_source)


def _fake_psutil(*, process_name: str | None):
    errors = {
        "NoSuchProcess": type("NoSuchProcess", (Exception,), {}),
        "AccessDenied": type("AccessDenied", (Exception,), {}),
        "ZombieProcess": type("ZombieProcess", (Exception,), {}),
    }

    def process(pid: int):
        if process_name is None:
            raise errors["NoSuchProcess"](pid)

        class _Proc:
            def name(self) -> str:
                return process_name

        return _Proc()

    return SimpleNamespace(
        Process=process,
        NoSuchProcess=errors["NoSuchProcess"],
        AccessDenied=errors["AccessDenied"],
        ZombieProcess=errors["ZombieProcess"],
    )


def _fake_windll(*, get_last_input_ok=1, last_dw_time=0, tick_now=5000, hwnd=0x1234, fg_pid=0):
    import ctypes

    class FakeGetTickCount:
        restype = ctypes.c_uint32

        def __call__(self):
            return tick_now

    class FakeKernel32:
        GetTickCount = FakeGetTickCount()

    class FakeUser32:
        def GetLastInputInfo(self, info_ptr):
            if not get_last_input_ok:
                return 0
            arr = ctypes.cast(info_ptr, ctypes.POINTER(ctypes.c_uint))
            arr[1] = last_dw_time
            return 1

        def GetForegroundWindow(self):
            return hwnd

        def GetWindowThreadProcessId(self, hwnd, pid_ptr):
            ctypes.cast(pid_ptr, ctypes.POINTER(ctypes.c_ulong)).contents.value = fg_pid
            return 1

    return SimpleNamespace(user32=FakeUser32(), kernel32=FakeKernel32())


if __name__ == "__main__":
    unittest.main()