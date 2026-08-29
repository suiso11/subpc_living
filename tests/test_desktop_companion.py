from __future__ import annotations

import json
import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.desktop.api import DesktopApi

try:
    from PySide6.QtWidgets import QApplication

    from src.desktop.bridge import DesktopBridge
    from src.desktop.config import DesktopSettings
except ImportError:
    QApplication = None
    DesktopBridge = None
    DesktopSettings = None


class CompanionStatePayloadTest(unittest.TestCase):
    def test_disabled_returns_enabled_false(self) -> None:
        api = DesktopApi("http://test")
        try:
            self.assertEqual(api.companion_state(None), {"enabled": False})
        finally:
            api.close()

    def test_payload_is_privacy_safe(self) -> None:
        runtime = SimpleNamespace(
            status=SimpleNamespace(
                running=True,
                last_state=SimpleNamespace(
                    activity_mode="focused",
                    present=True,
                    focused_since=123.0,
                    interruptible=False,
                    display_state="focused",
                    updated_at=456.0,
                ),
                last_update_at=456.0,
                failure_count=2,
                consecutive_failures=1,
                last_error_type="ValueError",
            )
        )
        api = DesktopApi("http://test")
        try:
            payload = api.companion_state(runtime)
        finally:
            api.close()

        self.assertEqual(
            set(payload),
            {
                "enabled",
                "running",
                "last_update_at",
                "failure_count",
                "consecutive_failures",
                "last_error_type",
                "state",
            },
        )
        self.assertEqual(
            set(payload["state"]),
            {
                "activity_mode",
                "present",
                "focused_since",
                "interruptible",
                "display_state",
                "updated_at",
            },
        )
        self.assertTrue(payload["enabled"])
        self.assertTrue(payload["running"])
        self.assertEqual(payload["state"]["activity_mode"], "focused")

        blob = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "process",
            "pid",
            "window",
            "title",
            "app_category",
            "classifier",
            "error_text",
            "traceback",
            "sample",
            "event",
        ):
            self.assertNotIn(forbidden, blob)


@unittest.skipIf(DesktopBridge is None, "PySide6 is not installed")
class DesktopBridgeCompanionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _make_runtime(self) -> SimpleNamespace:
        runtime = SimpleNamespace(
            stopped=False,
            stop_count=0,
            status=SimpleNamespace(
                running=True,
                last_state=None,
                last_update_at=None,
                failure_count=0,
                consecutive_failures=0,
                last_error_type=None,
            ),
        )

        def stop() -> None:
            runtime.stopped = True
            runtime.stop_count += 1

        runtime.stop = stop
        return runtime

    def test_bridge_starts_runtime_and_stops_on_shutdown(self) -> None:
        runtime = self._make_runtime()
        with patch(
            "src.desktop.bridge.create_activity_runtime_from_env",
            return_value=runtime,
        ):
            bridge = DesktopBridge(DesktopSettings(), offline=True)

        self.assertIs(bridge._activity_runtime, runtime)
        payload = bridge.companionState
        self.assertTrue(payload["enabled"])
        self.assertTrue(payload["running"])
        self.assertIsNone(payload["state"])
        self.assertTrue(bridge.companion_timer.isActive())

        bridge.shutdown()
        self.assertTrue(runtime.stopped)

    def test_disabled_runtime_is_hidden_and_stops_nothing(self) -> None:
        with patch(
            "src.desktop.bridge.create_activity_runtime_from_env",
            return_value=None,
        ):
            bridge = DesktopBridge(DesktopSettings(), offline=True)

        self.assertIsNone(bridge._activity_runtime)
        self.assertEqual(bridge.companionState, {"enabled": False})
        self.assertFalse(bridge.companion_timer.isActive())
        bridge.shutdown()

    def test_stop_overlay_disables_overlay_stops_runtime_and_payload(self) -> None:
        runtime = self._make_runtime()
        with patch(
            "src.desktop.bridge.create_activity_runtime_from_env",
            return_value=runtime,
        ):
            bridge = DesktopBridge(DesktopSettings(), offline=True)
        bridge._overlay_enabled = True

        self.assertTrue(bridge.overlayEnabled)
        self.assertTrue(bridge.companion_timer.isActive())
        self.assertTrue(bridge.companionState["enabled"])

        bridge.stopOverlayFromOverlay()

        self.assertFalse(bridge.overlayEnabled)
        self.assertFalse(bridge.companion_timer.isActive())
        self.assertEqual(runtime.stop_count, 1)
        self.assertIsNone(bridge._activity_runtime)
        self.assertEqual(bridge.companionState, {"enabled": False})
        self.assertFalse(bridge.overlayShell["enabled"])

    def test_repeated_stop_does_not_double_stop(self) -> None:
        runtime = self._make_runtime()
        with patch(
            "src.desktop.bridge.create_activity_runtime_from_env",
            return_value=runtime,
        ):
            bridge = DesktopBridge(DesktopSettings(), offline=True)
        bridge._overlay_enabled = True

        bridge.stopOverlayFromOverlay()
        bridge.stopOverlayFromOverlay()
        bridge.stopOverlayFromOverlay()

        self.assertEqual(runtime.stop_count, 1)
        self.assertIsNone(bridge._activity_runtime)
        self.assertEqual(bridge.companionState, {"enabled": False})

    def test_stop_with_disabled_runtime_is_harmless(self) -> None:
        with patch(
            "src.desktop.bridge.create_activity_runtime_from_env",
            return_value=None,
        ):
            bridge = DesktopBridge(DesktopSettings(), offline=True)

        self.assertIsNone(bridge._activity_runtime)
        self.assertFalse(bridge.companion_timer.isActive())

        bridge.stopOverlayFromOverlay()

        self.assertIsNone(bridge._activity_runtime)
        self.assertEqual(bridge.companionState, {"enabled": False})

    def test_stop_emits_companion_and_overlay_state_signals(self) -> None:
        runtime = self._make_runtime()
        with patch(
            "src.desktop.bridge.create_activity_runtime_from_env",
            return_value=runtime,
        ):
            bridge = DesktopBridge(DesktopSettings(), offline=True)
        bridge._overlay_enabled = True
        companion_changes: list[bool] = []
        overlay_changes: list[bool] = []
        bridge.companionStateChanged.connect(lambda: companion_changes.append(True))
        bridge.overlayShellChanged.connect(lambda: overlay_changes.append(True))

        bridge.stopOverlayFromOverlay()

        self.assertEqual(len(companion_changes), 1)
        self.assertEqual(len(overlay_changes), 1)

    def test_shutdown_after_stop_does_not_double_stop(self) -> None:
        runtime = self._make_runtime()
        with patch(
            "src.desktop.bridge.create_activity_runtime_from_env",
            return_value=runtime,
        ):
            bridge = DesktopBridge(DesktopSettings(), offline=True)

        bridge.stopOverlayFromOverlay()
        bridge.shutdown()

        self.assertEqual(runtime.stop_count, 1)
        self.assertIsNone(bridge._activity_runtime)


@unittest.skipIf(DesktopBridge is None, "PySide6 is not installed")
class DesktopBridgeOverlayClickThroughTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _make_runtime(self) -> SimpleNamespace:
        runtime = SimpleNamespace(
            stopped=False,
            stop_count=0,
            status=SimpleNamespace(
                running=True,
                last_state=None,
                last_update_at=None,
                failure_count=0,
                consecutive_failures=0,
                last_error_type=None,
            ),
        )

        def stop() -> None:
            runtime.stopped = True
            runtime.stop_count += 1

        runtime.stop = stop
        return runtime

    def _make_bridge(self) -> DesktopBridge:
        with patch(
            "src.desktop.bridge.create_activity_runtime_from_env",
            return_value=self._make_runtime(),
        ):
            return DesktopBridge(DesktopSettings(), offline=True)

    def _track(self, bridge: DesktopBridge) -> tuple[list[bool], list[bool]]:
        state_changes: list[bool] = []
        requests: list[bool] = []
        bridge.overlayClickThroughChanged.connect(
            lambda: state_changes.append(bridge.overlayClickThrough)
        )
        bridge.overlayClickThroughRequested.connect(requests.append)
        return state_changes, requests

    def test_defaults_to_false(self) -> None:
        bridge = self._make_bridge()
        self.assertFalse(bridge.overlayClickThrough)
        self.assertFalse(bridge._overlay_click_through)
        bridge.shutdown()

    def test_enable_transition_while_overlay_enabled(self) -> None:
        bridge = self._make_bridge()
        bridge._overlay_enabled = True
        state_changes, requests = self._track(bridge)

        bridge.setOverlayClickThrough(True)

        self.assertTrue(bridge.overlayClickThrough)
        self.assertEqual(state_changes, [True])
        self.assertEqual(requests, [True])
        bridge.shutdown()

    def test_disable_transition_is_always_allowed(self) -> None:
        bridge = self._make_bridge()
        bridge._overlay_click_through = True
        state_changes, requests = self._track(bridge)

        bridge.setOverlayClickThrough(False)

        self.assertFalse(bridge.overlayClickThrough)
        self.assertEqual(state_changes, [False])
        self.assertEqual(requests, [False])
        bridge.shutdown()

    def test_enable_rejected_while_overlay_disabled(self) -> None:
        bridge = self._make_bridge()
        state_changes, requests = self._track(bridge)

        bridge.setOverlayClickThrough(True)

        self.assertFalse(bridge.overlayClickThrough)
        self.assertFalse(bridge._overlay_click_through)
        self.assertEqual(state_changes, [])
        self.assertEqual(requests, [])
        bridge.shutdown()

    def test_idempotent_calls_do_not_duplicate_signals(self) -> None:
        bridge = self._make_bridge()
        bridge._overlay_enabled = True
        state_changes, requests = self._track(bridge)

        bridge.setOverlayClickThrough(True)
        bridge.setOverlayClickThrough(True)
        bridge.setOverlayClickThrough(False)
        bridge.setOverlayClickThrough(False)

        self.assertEqual(state_changes, [True, False])
        self.assertEqual(requests, [True, False])
        bridge.shutdown()

    def test_signal_order_is_state_then_request(self) -> None:
        bridge = self._make_bridge()
        bridge._overlay_enabled = True
        order: list[str] = []
        bridge.overlayClickThroughChanged.connect(lambda: order.append("state"))
        bridge.overlayClickThroughRequested.connect(lambda _value: order.append("request"))

        bridge.setOverlayClickThrough(True)

        self.assertEqual(order, ["state", "request"])
        bridge.shutdown()

    def test_stop_overlay_forces_click_through_off_before_disable(self) -> None:
        bridge = self._make_bridge()
        bridge._overlay_enabled = True
        bridge.setOverlayClickThrough(True)
        state_changes, requests = self._track(bridge)

        bridge.stopOverlayFromOverlay()

        self.assertFalse(bridge.overlayClickThrough)
        self.assertFalse(bridge.overlayEnabled)
        self.assertEqual(state_changes, [False])
        self.assertEqual(requests, [False])
        bridge.shutdown()

    def test_repeated_stop_does_not_reemit_click_through(self) -> None:
        bridge = self._make_bridge()
        bridge._overlay_enabled = True
        bridge.setOverlayClickThrough(True)
        state_changes, requests = self._track(bridge)

        bridge.stopOverlayFromOverlay()
        bridge.stopOverlayFromOverlay()

        self.assertEqual(state_changes, [False])
        self.assertEqual(requests, [False])
        bridge.shutdown()

    def test_shutdown_forces_click_through_off(self) -> None:
        bridge = self._make_bridge()
        bridge._overlay_enabled = True
        bridge.setOverlayClickThrough(True)
        state_changes, requests = self._track(bridge)

        bridge.shutdown()

        self.assertFalse(bridge.overlayClickThrough)
        self.assertEqual(state_changes, [False])
        self.assertEqual(requests, [False])


class _FakeRecorder:
    instances: list["_FakeRecorder"] = []

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self.recording = False
        self.start_calls = 0
        self.stop_calls = 0
        type(self).instances.append(self)

    def start(self) -> None:
        self.start_calls += 1
        self.recording = True

    def stop(self) -> bytes:
        self.stop_calls += 1
        self.recording = False
        return b""


@unittest.skipIf(DesktopBridge is None, "PySide6 is not installed")
class DesktopBridgeMicrophonePolicyTest(unittest.TestCase):
    MIC = "SENSOR_MICROPHONE_ENABLED"

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @contextmanager
    def _bridge_scope(
        self,
        mic_value: str | None,
        *,
        fake_recorder_cls: type | None = None,
    ) -> Iterator[DesktopBridge]:
        recorder_cls = fake_recorder_cls or _FakeRecorder
        env = {key: value for key, value in os.environ.items() if key != self.MIC}
        if mic_value is not None:
            env[self.MIC] = mic_value
        with patch.dict(os.environ, env, clear=True):
            with patch(
                "src.desktop.bridge.create_activity_runtime_from_env",
                return_value=None,
            ):
                with patch("src.desktop.bridge.NativeAudioRecorder", recorder_cls):
                    yield DesktopBridge(DesktopSettings(), offline=True)

    def _assert_denied(self, mic_value: str | None) -> None:
        _FakeRecorder.instances = []
        with self._bridge_scope(mic_value) as bridge:
            toasts: list[str] = []
            bridge.toast.connect(lambda title, _message: toasts.append(title))

            self.assertIsNone(bridge.recorder)
            bridge.startRecording()
            bridge.startRecording()

            self.assertIsNone(bridge.recorder)
            self.assertFalse(bridge._recording)
            self.assertFalse(bridge.recording)
            self.assertEqual(_FakeRecorder.instances, [])
            self.assertEqual(len(toasts), 2)
            self.assertTrue(all("マイク" in title for title in toasts))
            bridge.shutdown()

    def test_default_missing_is_fail_closed(self) -> None:
        self._assert_denied(None)

    def test_canonical_false_is_fail_closed(self) -> None:
        self._assert_denied("false")

    def test_canonical_invalid_values_fail_closed(self) -> None:
        for value in ("1", "yes", "on", "0", "trueish", " FALSE "):
            self._assert_denied(value)

    def test_canonical_true_allows_existing_recorder(self) -> None:
        _FakeRecorder.instances = []
        with self._bridge_scope("true") as bridge:
            toasts: list[str] = []
            bridge.toast.connect(lambda title, _message: toasts.append(title))

            self.assertIsNone(bridge.recorder)
            bridge.startRecording()

            self.assertIsNotNone(bridge.recorder)
            self.assertTrue(bridge._recording)
            self.assertTrue(bridge.recording)
            self.assertEqual(bridge.recorder.start_calls, 1)
            self.assertEqual(len(_FakeRecorder.instances), 1)
            self.assertEqual(toasts, [])

            bridge.stopRecording()
            self.assertFalse(bridge.recording)
            bridge.shutdown()

    def test_enabled_whitespace_true_is_accepted(self) -> None:
        _FakeRecorder.instances = []
        with self._bridge_scope(" TRUE ") as bridge:
            bridge.startRecording()
            self.assertIsNotNone(bridge.recorder)
            self.assertTrue(bridge.recording)
            self.assertEqual(len(_FakeRecorder.instances), 1)
            bridge.shutdown()

    def test_enabled_reuses_recorder_across_starts(self) -> None:
        _FakeRecorder.instances = []
        with self._bridge_scope("true") as bridge:
            bridge.startRecording()
            bridge.stopRecording()
            bridge.startRecording()

            self.assertEqual(len(_FakeRecorder.instances), 1)
            self.assertEqual(bridge.recorder.start_calls, 2)
            bridge.shutdown()

    def test_disabled_never_starts_recorder_or_input_stream(self) -> None:
        class _ExplodingRecorder:
            def __init__(self, sample_rate: int = 16000) -> None:
                self.sample_rate = sample_rate
                self.recording = False

            def start(self) -> None:
                raise AssertionError(
                    "recorder.start must not be called when microphone is disabled"
                )

            def stop(self) -> bytes:
                return b""

        for value in (None, "false", "1"):
            with self._bridge_scope(
                value, fake_recorder_cls=_ExplodingRecorder
            ) as bridge:
                bridge.startRecording()
                self.assertIsNone(bridge.recorder)
                self.assertFalse(bridge.recording)
                bridge.shutdown()


@unittest.skipIf(DesktopBridge is None, "PySide6 is not installed")
class DesktopBridgeRecorderErrorSanitizationTest(unittest.TestCase):
    MIC = "SENSOR_MICROPHONE_ENABLED"

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @contextmanager
    def _bridge_scope(
        self,
        mic_value: str | None,
        recorder_cls: type,
    ) -> Iterator[DesktopBridge]:
        env = {key: value for key, value in os.environ.items() if key != self.MIC}
        if mic_value is not None:
            env[self.MIC] = mic_value
        with patch.dict(os.environ, env, clear=True):
            with patch(
                "src.desktop.bridge.create_activity_runtime_from_env",
                return_value=None,
            ):
                with patch("src.desktop.bridge.NativeAudioRecorder", recorder_cls):
                    yield DesktopBridge(DesktopSettings(), offline=True)

    def _assert_sanitized(
        self,
        toasts: list[tuple[str, str]],
        *,
        expected_title_needle: str,
        forbidden: tuple[str, ...],
    ) -> None:
        self.assertEqual(len(toasts), 1)
        title, message = toasts[0]
        self.assertIn(expected_title_needle, title)
        for token in forbidden:
            self.assertNotIn(token, title)
            self.assertNotIn(token, message)

    def test_disabled_mic_never_constructs_recorder(self) -> None:
        constructed: list[bool] = []

        class _CountingConstructor:
            def __init__(self, sample_rate: int = 16000) -> None:
                constructed.append(True)
                self.sample_rate = sample_rate
                self.recording = False

            def start(self) -> None:
                raise AssertionError("start must not be called")

            def stop(self) -> bytes:
                return b""

        with self._bridge_scope(None, _CountingConstructor) as bridge:
            bridge.startRecording()

        self.assertEqual(constructed, [])
        self.assertIsNone(bridge.recorder)
        self.assertFalse(bridge.recording)

    def test_construction_error_uses_fixed_message_and_state_stays_false(self) -> None:
        class _ExplodingConstructor:
            def __init__(self, sample_rate: int = 16000) -> None:
                raise RuntimeError(
                    "C:\\Program Files\\sounddevice\\microphone\\broken.ini"
                )

            def start(self) -> None:
                raise AssertionError("start must not be called")

            def stop(self) -> bytes:
                return b""

        toasts: list[tuple[str, str]] = []
        with self._bridge_scope("true", _ExplodingConstructor) as bridge:
            bridge.toast.connect(lambda title, message: toasts.append((title, message)))
            bridge.startRecording()

        self.assertIsNone(bridge.recorder)
        self.assertFalse(bridge.recording)
        self.assertFalse(bridge._recording)
        self._assert_sanitized(
            toasts,
            expected_title_needle="マイク",
            forbidden=("Program Files", "broken.ini", "sounddevice", "C:\\"),
        )

    def test_start_error_uses_fixed_message_and_no_raw_details(self) -> None:
        class _ExplodingStart:
            def __init__(self, sample_rate: int = 16000) -> None:
                self.sample_rate = sample_rate
                self.recording = False

            def start(self) -> None:
                raise RuntimeError(
                    "Device 'Realtek HD Audio' @C:/dev/mic failed: cannot open stream"
                )

            def stop(self) -> bytes:
                return b""

        toasts: list[tuple[str, str]] = []
        with self._bridge_scope("true", _ExplodingStart) as bridge:
            bridge.toast.connect(lambda title, message: toasts.append((title, message)))
            bridge.startRecording()

        self.assertFalse(bridge.recording)
        self.assertFalse(bridge._recording)
        self._assert_sanitized(
            toasts,
            expected_title_needle="マイク",
            forbidden=("Realtek", "C:/dev", "cannot open stream", "Device"),
        )

    def test_stop_error_uses_fixed_message_resets_state_and_sends_no_audio(self) -> None:
        class _ExplodingStop:
            def __init__(self, sample_rate: int = 16000) -> None:
                self.sample_rate = sample_rate
                self.recording = False

            def start(self) -> None:
                self.recording = True

            def stop(self) -> bytes:
                raise RuntimeError(
                    "Failed to finalize 44000-byte buffer at C:\\Users\\x\\rec.wav"
                )

        toasts: list[tuple[str, str]] = []
        with self._bridge_scope("true", _ExplodingStop) as bridge:
            bridge.toast.connect(lambda title, message: toasts.append((title, message)))
            bridge.startRecording()
            self.assertTrue(bridge.recording)
            bridge.stopRecording()

        self.assertFalse(bridge.recording)
        self.assertFalse(bridge._recording)
        self.assertEqual(bridge._pending_chat, [])
        self._assert_sanitized(
            toasts,
            expected_title_needle="録音",
            forbidden=("44000", "C:\\Users", "rec.wav", "buffer"),
        )


if __name__ == "__main__":
    unittest.main()
