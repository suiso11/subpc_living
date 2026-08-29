from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import httpx
    import PySide6
    from PySide6.QtCore import QEvent, QMetaObject, QObject, QUrl, Qt, Property, Q_ARG, Signal, Slot
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtWidgets import QApplication

    from src.desktop.api import DesktopApi
    from src.desktop.app import SingleInstanceGuard
    from src.desktop.bridge import DesktopBridge
    from src.desktop.config import DesktopSettings

    class FakeOverlayBridge(QObject):
        """Minimal overlay-facing bridge for offscreen Overlay.qml tests."""

        statusChanged = Signal()
        loadingChanged = Signal()
        messagesChanged = Signal()
        gameChanged = Signal()
        overlayShellChanged = Signal()
        growthChanged = Signal()
        tasksChanged = Signal()
        calendarChanged = Signal()
        overlayEnabledChanged = Signal()
        overlayClickThroughChanged = Signal()

        def __init__(
            self,
            *,
            connected: bool = True,
            loading: bool = False,
            shell_state: str = "idle",
            messages: list[dict] | None = None,
            starters: list[dict] | None = None,
            growth: dict | None = None,
            tasks: list[dict] | None = None,
            calendar_events: list[dict] | None = None,
            overlay_enabled: bool = True,
            click_through: bool = False,
        ) -> None:
            super().__init__()
            self._connected = connected
            self._loading = loading
            self._shell_state = shell_state
            self._messages = list(messages or [])
            self._starters = list(starters or [])
            self._growth = dict(growth or {})
            self._tasks = list(tasks or [])
            self._calendar_events = list(calendar_events or [])
            self._overlay_enabled = overlay_enabled
            self._overlay_click_through = click_through
            self.sent: list[str] = []
            self.opened = 0
            self.stopped = 0
            self.click_through_calls: list[bool] = []
            self.click_through_requests: list[bool] = []

        def set_growth(self, growth: dict) -> None:
            self._growth = dict(growth or {})
            self.growthChanged.emit()

        def set_tasks(self, tasks: list[dict]) -> None:
            self._tasks = list(tasks or [])
            self.tasksChanged.emit()

        def set_calendar_events(self, events: list[dict]) -> None:
            self._calendar_events = list(events or [])
            self.calendarChanged.emit()

        def _shell(self) -> dict:
            return {
                "enabled": True,
                "shell_state": self._shell_state,
                "visible": True,
                "shrink": False,
                "provenance": {"source_label": "activity", "fetched_at": 0},
                "companion": {"enabled": True},
            }

        overlayEnabled = Property(bool, lambda self: self._overlay_enabled, notify=overlayEnabledChanged)
        overlayClickThrough = Property(bool, lambda self: self._overlay_click_through, notify=overlayClickThroughChanged)
        avatarModel = Property("QVariantMap", lambda self: {"path": "", "exists": False}, constant=True)
        overlayShell = Property("QVariantMap", lambda self: self._shell(), notify=overlayShellChanged)
        messages = Property("QVariantList", lambda self: self._messages, notify=messagesChanged)
        game = Property("QVariantMap", lambda self: {"starters": self._starters}, notify=gameChanged)
        growth = Property("QVariantMap", lambda self: self._growth, notify=growthChanged)
        tasks = Property("QVariantList", lambda self: self._tasks, notify=tasksChanged)
        calendarEvents = Property("QVariantList", lambda self: self._calendar_events, notify=calendarChanged)
        connected = Property(bool, lambda self: self._connected, notify=statusChanged)
        loading = Property(bool, lambda self: self._loading, notify=loadingChanged)

        @Slot(str)
        def sendMessage(self, text: str) -> None:
            self.sent.append(text)

        @Slot()
        def openMainFromOverlay(self) -> None:
            self.opened += 1

        @Slot()
        def stopOverlayFromOverlay(self) -> None:
            self.stopped += 1

        @Slot(bool)
        def setOverlayClickThrough(self, enabled: bool) -> None:
            """Mirror DesktopBridge semantics: reject enabling while overlay is off."""
            self.click_through_calls.append(enabled)
            if enabled and not self._overlay_enabled:
                return
            if enabled == self._overlay_click_through:
                return
            self._overlay_click_through = enabled
            self.overlayClickThroughChanged.emit()
            self.click_through_requests.append(enabled)
except ImportError:
    QApplication = None


@unittest.skipIf(QApplication is None, "PySide6 is not installed in the server environment")
class DesktopQmlSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_main_qml_loads_offscreen(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payloads = {
                "/api/status": {"model": "test"},
                "/api/tasks": {"tasks": []},
                "/api/game": {"enabled": True, "rank": {"name": "相棒"}, "badges": [], "missions": []},
                "/api/chat/resume": {"session_id": "desktop_test", "messages": []},
                "/api/history/sessions": {"sessions": []},
            }
            return httpx.Response(200, json=payloads.get(request.url.path, {"ok": True}))

        with tempfile.TemporaryDirectory() as directory:
            settings = DesktopSettings(server_url="http://test")
            bridge = DesktopBridge(
                settings,
                parent=self.app,
                settings_path=Path(directory) / "desktop.json",
            )
            bridge.api.close()
            bridge.api = DesktopApi("http://test", transport=httpx.MockTransport(handler))
            engine = QQmlApplicationEngine()
            engine.rootContext().setContextProperty("bridge", bridge)
            qml = Path(__file__).resolve().parents[1] / "src" / "desktop" / "qml" / "Main.qml"
            engine.addImportPath(str(qml.parent))
            engine.load(QUrl.fromLocalFile(str(qml)))
            self.app.processEvents()
            self.assertEqual(len(engine.rootObjects()), 1)
            bridge.shutdown()

    def test_top_bar_adapts_without_overlapping_primary_navigation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = DesktopBridge(
                DesktopSettings(),
                parent=self.app,
                offline=True,
                settings_path=Path(directory) / "desktop.json",
            )
            engine = QQmlApplicationEngine()
            engine.rootContext().setContextProperty("bridge", bridge)
            qml = Path(__file__).resolve().parents[1] / "src" / "desktop" / "qml" / "Main.qml"
            engine.addImportPath(str(qml.parent))
            engine.load(QUrl.fromLocalFile(str(qml)))
            window = engine.rootObjects()[0]
            window.setWidth(820)
            window.setHeight(560)
            self.app.processEvents()

            brand = window.findChild(QObject, "brandPill")
            navigation = window.findChild(QObject, "navigationPill")
            command_hint = window.findChild(QObject, "commandHint")
            controls = window.findChild(QObject, "windowControls")
            self.assertIsNotNone(brand)
            self.assertIsNotNone(navigation)
            self.assertIsNotNone(command_hint)
            self.assertIsNotNone(controls)
            self.assertFalse(brand.property("visible"))
            self.assertFalse(command_hint.property("visible"))
            self.assertLessEqual(
                navigation.property("x") + navigation.property("width"),
                controls.property("x"),
            )

            window.setWidth(1220)
            self.app.processEvents()
            self.assertTrue(brand.property("visible"))
            self.assertTrue(command_hint.property("visible"))
            bridge.shutdown()

    def test_single_instance_guard_activates_existing_process(self) -> None:
        name = f"subpc-buddy-test-{uuid.uuid4().hex}"
        first = SingleInstanceGuard(name)
        second = SingleInstanceGuard(name)
        activated: list[bool] = []
        first.activateRequested.connect(lambda: activated.append(True))
        try:
            self.assertTrue(first.acquire())
            self.assertFalse(second.acquire())
            self.app.processEvents()
            self.assertEqual(activated, [True])
        finally:
            first.close()
            second.close()

    def test_qml_has_no_missing_properties_or_layout_conflicts(self) -> None:
        qml_dir = Path(__file__).resolve().parents[1] / "src" / "desktop" / "qml"
        executable = Path(PySide6.__file__).resolve().parent / (
            "qmllint.exe" if os.name == "nt" else "qmllint"
        )
        self.assertTrue(executable.is_file())
        result = subprocess.run(
            [str(executable), "-I", str(qml_dir), *map(str, sorted(qml_dir.glob("*.qml")))],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        self.assertNotIn("[missing-property]", output)
        self.assertNotIn("[Quick.layout-positioning]", output)


@unittest.skipIf(QApplication is None, "PySide6 is not installed in the server environment")
class DesktopOverlayQmlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _load(self, bridge: FakeOverlayBridge) -> tuple[QQmlApplicationEngine, QObject]:
        engine = QQmlApplicationEngine()
        engine.rootContext().setContextProperty("overlayBridge", bridge)
        qml_dir = Path(__file__).resolve().parents[1] / "src" / "desktop" / "qml"
        engine.addImportPath(str(qml_dir))
        engine.load(QUrl.fromLocalFile(str(qml_dir / "Overlay.qml")))
        self.app.processEvents()
        self.assertEqual(len(engine.rootObjects()), 1)
        return engine, engine.rootObjects()[0]

    def _expand(self, root: QObject) -> None:
        root.setProperty("expanded", True)
        self.app.processEvents()

    def test_expand_shows_hud_hides_avatar_and_focuses_composer(self) -> None:
        bridge = FakeOverlayBridge()
        engine, root = self._load(bridge)
        hud = root.findChild(QObject, "overlayHud")
        avatar = root.findChild(QObject, "overlayAvatar")
        composer = root.findChild(QObject, "overlayComposer")
        send = root.findChild(QObject, "overlaySendButton")
        self.assertIsNotNone(hud)
        self.assertIsNotNone(avatar)
        self.assertIsNotNone(composer)
        self.assertIsNotNone(send)
        self.assertFalse(hud.property("visible"))
        self.assertTrue(avatar.property("visible"))
        self.assertTrue(composer.property("enabled"))
        self._expand(root)
        self.assertTrue(hud.property("visible"))
        self.assertFalse(avatar.property("visible"))
        self.assertTrue(composer.property("activeFocus"))

    def test_send_trims_clears_and_calls_bridge_once(self) -> None:
        bridge = FakeOverlayBridge()
        engine, root = self._load(bridge)
        self._expand(root)
        composer = root.findChild(QObject, "overlayComposer")
        send = root.findChild(QObject, "overlaySendButton")
        composer.setProperty("text", "  こんにちは  ")
        self.app.processEvents()
        QMetaObject.invokeMethod(send, "clicked")
        self.app.processEvents()
        self.assertEqual(bridge.sent, ["こんにちは"])
        self.assertEqual(composer.property("text"), "")

    def test_send_empty_text_does_not_call_bridge(self) -> None:
        bridge = FakeOverlayBridge()
        engine, root = self._load(bridge)
        self._expand(root)
        composer = root.findChild(QObject, "overlayComposer")
        composer.setProperty("text", "   ")
        self.app.processEvents()
        QMetaObject.invokeMethod(root, "send")
        self.app.processEvents()
        self.assertEqual(bridge.sent, [])

    def test_composer_disabled_offline_and_loading(self) -> None:
        offline = FakeOverlayBridge(connected=False)
        engine, root = self._load(offline)
        self._expand(root)
        composer = root.findChild(QObject, "overlayComposer")
        send = root.findChild(QObject, "overlaySendButton")
        banner = root.findChild(QObject, "overlayStatusBanner")
        self.assertIsNotNone(banner)
        self.assertFalse(composer.property("enabled"))
        self.assertFalse(send.property("enabled"))
        composer.setProperty("text", "hello")
        self.app.processEvents()
        QMetaObject.invokeMethod(root, "send")
        self.app.processEvents()
        self.assertEqual(offline.sent, [])

        loading = FakeOverlayBridge(loading=True)
        engine2, root2 = self._load(loading)
        self._expand(root2)
        composer2 = root2.findChild(QObject, "overlayComposer")
        send2 = root2.findChild(QObject, "overlaySendButton")
        self.assertFalse(composer2.property("enabled"))
        self.assertFalse(send2.property("enabled"))

    def test_recent_messages_bounded_and_filtered(self) -> None:
        messages = [
            {"role": "user", "content": "m0"},
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "m2"},
            {"role": "user", "content": "m3"},
            {"role": "user", "content": "あ" * 400},
            {"role": "assistant", "content": "い" * 400},
        ]
        bridge = FakeOverlayBridge(messages=messages)
        engine, root = self._load(bridge)
        self._expand(root)
        repeater = root.findChild(QObject, "overlayMessageRepeater")
        recent = root.findChild(QObject, "overlayRecentMessages")
        self.assertEqual(repeater.property("count"), 4)
        self.assertTrue(recent.property("visible"))
        self.assertLessEqual(recent.property("implicitHeight"), 4 * 52 + 3 * 4)
        chat_messages = root.property("chatMessages").toVariant()
        self.assertEqual([m["role"] for m in chat_messages], ["assistant", "user", "user", "assistant"])

    def test_starter_chips_bounded_to_three_and_fill_composer(self) -> None:
        starters = [{"id": str(i), "label": f"L{i}", "prompt": f"P{i}"} for i in range(5)]
        bridge = FakeOverlayBridge(starters=starters)
        engine, root = self._load(bridge)
        self._expand(root)
        repeater = root.findChild(QObject, "overlayStarterRepeater")
        chips = root.findChild(QObject, "overlayStarterChips")
        self.assertEqual(repeater.property("count"), 3)
        self.assertTrue(chips.property("visible"))
        composer = root.findChild(QObject, "overlayComposer")
        QMetaObject.invokeMethod(root, "useStarter", Q_ARG("QVariant", "P1"))
        self.app.processEvents()
        self.assertEqual(composer.property("text"), "P1")
        self.assertTrue(composer.property("activeFocus"))

    def test_summary_renders_allowlisted_counts(self) -> None:
        import datetime

        today = datetime.date.today()
        tomorrow = today + datetime.timedelta(days=1)
        yesterday = today - datetime.timedelta(days=1)
        bridge = FakeOverlayBridge(
            growth={"today_points": 42, "streak_days": 3},
            tasks=[
                {"id": 1, "status": "open"},
                {"id": 2, "status": "open"},
                {"id": 3, "status": "done"},
                {"id": 4},
            ],
            calendar_events=[
                {"start": yesterday.isoformat()},
                {"start": today.isoformat()},
                {"start": tomorrow.isoformat() + "T10:00:00"},
            ],
        )
        engine, root = self._load(bridge)
        self._expand(root)
        growth_value = root.findChild(QObject, "overlaySummaryGrowthValue")
        tasks_value = root.findChild(QObject, "overlaySummaryTasksValue")
        calendar_value = root.findChild(QObject, "overlaySummaryCalendarValue")
        self.assertIsNotNone(growth_value)
        self.assertIsNotNone(tasks_value)
        self.assertIsNotNone(calendar_value)
        self.assertEqual(growth_value.property("text"), "+42pt · 3日")
        self.assertEqual(tasks_value.property("text"), "3件")
        self.assertEqual(calendar_value.property("text"), "2件")

    def test_summary_handles_null_and_malformed_data(self) -> None:
        bridge = FakeOverlayBridge(
            growth={"today_points": None, "streak_days": "bad"},
            tasks=[{"id": 1, "status": "dropped"}, {"id": 2, "status": "done"}, "not-a-dict"],
            calendar_events=[{"start": "not-a-date"}, None],
        )
        engine, root = self._load(bridge)
        self._expand(root)
        growth_value = root.findChild(QObject, "overlaySummaryGrowthValue")
        tasks_value = root.findChild(QObject, "overlaySummaryTasksValue")
        calendar_value = root.findChild(QObject, "overlaySummaryCalendarValue")
        self.assertEqual(growth_value.property("text"), "+0pt")
        self.assertEqual(tasks_value.property("text"), "0件")
        self.assertEqual(calendar_value.property("text"), "0件")

        bridge.set_growth(None)
        bridge.set_tasks(None)
        bridge.set_calendar_events(None)
        self.app.processEvents()
        self.assertEqual(growth_value.property("text"), "+0pt")
        self.assertEqual(tasks_value.property("text"), "0件")
        self.assertEqual(calendar_value.property("text"), "0件")

    def test_summary_updates_when_bridge_data_changes(self) -> None:
        import datetime

        today = datetime.date.today()
        bridge = FakeOverlayBridge(
            growth={"today_points": 1, "streak_days": 0},
            tasks=[{"id": 1, "status": "open"}],
            calendar_events=[{"start": today.isoformat()}],
        )
        engine, root = self._load(bridge)
        self._expand(root)
        growth_value = root.findChild(QObject, "overlaySummaryGrowthValue")
        tasks_value = root.findChild(QObject, "overlaySummaryTasksValue")
        calendar_value = root.findChild(QObject, "overlaySummaryCalendarValue")
        bridge.set_growth({"today_points": 99, "streak_days": 5})
        bridge.set_tasks([{"id": 1, "status": "open"}, {"id": 2, "status": "open"}])
        bridge.set_calendar_events([])
        self.app.processEvents()
        self.assertEqual(growth_value.property("text"), "+99pt · 5日")
        self.assertEqual(tasks_value.property("text"), "2件")
        self.assertEqual(calendar_value.property("text"), "0件")

    def test_escape_key_collapses_hud(self) -> None:
        bridge = FakeOverlayBridge()
        engine, root = self._load(bridge)
        self._expand(root)
        self.assertTrue(root.property("expanded"))
        root.requestActivate()
        self.app.processEvents()
        key = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(root, key)
        self.app.processEvents()
        self.assertFalse(root.property("expanded"))

    def test_click_through_defaults_off(self) -> None:
        bridge = FakeOverlayBridge()
        engine, root = self._load(bridge)
        control = root.findChild(QObject, "overlayClickThrough")
        hint = root.findChild(QObject, "overlayClickThroughHint")
        self.assertIsNotNone(control)
        self.assertIsNotNone(hint)
        self.assertFalse(bridge.overlayClickThrough)
        self.assertFalse(control.property("checked"))
        self.assertFalse(hint.property("visible"))
        self.assertEqual(bridge.click_through_calls, [])
        self.assertEqual(bridge.click_through_requests, [])

    def test_click_through_enable_requests_once_and_collapses(self) -> None:
        bridge = FakeOverlayBridge()
        engine, root = self._load(bridge)
        self._expand(root)
        control = root.findChild(QObject, "overlayClickThrough")
        QMetaObject.invokeMethod(control, "click")
        self.app.processEvents()
        self.assertTrue(bridge.overlayClickThrough)
        self.assertTrue(control.property("checked"))
        self.assertEqual(bridge.click_through_calls, [True])
        self.assertEqual(bridge.click_through_requests, [True])
        self.assertFalse(root.property("expanded"))

    def test_click_through_disable_keeps_hud_expanded(self) -> None:
        bridge = FakeOverlayBridge()
        engine, root = self._load(bridge)
        self._expand(root)
        control = root.findChild(QObject, "overlayClickThrough")
        QMetaObject.invokeMethod(control, "click")
        self.app.processEvents()
        self.assertFalse(root.property("expanded"))
        self._expand(root)
        QMetaObject.invokeMethod(control, "click")
        self.app.processEvents()
        self.assertFalse(bridge.overlayClickThrough)
        self.assertFalse(control.property("checked"))
        self.assertEqual(bridge.click_through_calls, [True, False])
        self.assertEqual(bridge.click_through_requests, [True, False])
        self.assertTrue(root.property("expanded"))

    def test_click_through_reset_reflection_on_failed_apply(self) -> None:
        bridge = FakeOverlayBridge()
        engine, root = self._load(bridge)
        self._expand(root)
        control = root.findChild(QObject, "overlayClickThrough")
        QMetaObject.invokeMethod(control, "click")
        self.app.processEvents()
        self.assertTrue(control.property("checked"))
        bridge.setOverlayClickThrough(False)
        self.app.processEvents()
        self.assertFalse(control.property("checked"))
        self.assertFalse(bridge.overlayClickThrough)

    def test_click_through_rejected_when_overlay_disabled(self) -> None:
        bridge = FakeOverlayBridge(overlay_enabled=False)
        engine, root = self._load(bridge)
        self._expand(root)
        control = root.findChild(QObject, "overlayClickThrough")
        QMetaObject.invokeMethod(control, "click")
        self.app.processEvents()
        self.assertFalse(bridge.overlayClickThrough)
        self.assertFalse(control.property("checked"))
        self.assertEqual(bridge.click_through_requests, [])
        self.assertTrue(root.property("expanded"))

    def test_click_through_recovery_hint_is_fixed_and_reacts(self) -> None:
        bridge = FakeOverlayBridge()
        engine, root = self._load(bridge)
        self._expand(root)
        hint = root.findChild(QObject, "overlayClickThroughHint")
        self.assertIsNotNone(hint)
        self.assertEqual(hint.property("text"), "Ctrl+Alt+Space で解除")
        self.assertFalse(hint.property("visible"))
        bridge.setOverlayClickThrough(True)
        self.app.processEvents()
        self.assertTrue(hint.property("visible"))
        bridge.setOverlayClickThrough(False)
        self.app.processEvents()
        self.assertFalse(hint.property("visible"))

    def test_click_through_control_preserves_chat_behavior(self) -> None:
        bridge = FakeOverlayBridge()
        engine, root = self._load(bridge)
        self._expand(root)
        composer = root.findChild(QObject, "overlayComposer")
        self.assertTrue(composer.property("activeFocus"))
        composer.setProperty("text", "こんにちは")
        self.app.processEvents()
        QMetaObject.invokeMethod(root.findChild(QObject, "overlaySendButton"), "clicked")
        self.app.processEvents()
        self.assertEqual(bridge.sent, ["こんにちは"])
        self.assertEqual(composer.property("text"), "")
        self.assertTrue(root.property("expanded"))


if __name__ == "__main__":
    unittest.main()
