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
    from PySide6.QtCore import QObject, QUrl
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtWidgets import QApplication

    from src.desktop.api import DesktopApi
    from src.desktop.app import SingleInstanceGuard
    from src.desktop.bridge import DesktopBridge
    from src.desktop.config import DesktopSettings
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


if __name__ == "__main__":
    unittest.main()
