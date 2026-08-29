"""Application bootstrap for the Windows-native QML client."""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path

from PySide6.QtCore import QAbstractNativeEventFilter, QObject, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from .bridge import DesktopBridge
from .config import DesktopSettings
from .windows import HOTKEY_ID, apply_click_through, apply_windows_backdrop


def resource_path(relative: str) -> Path:
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
    return root / relative


class WindowsHotkeyFilter(QObject, QAbstractNativeEventFilter):
    activated = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        QObject.__init__(self, parent)
        QAbstractNativeEventFilter.__init__(self)
        self.registered = False

    def register(self) -> bool:
        if os.name != "nt":
            return False
        # Ctrl+Alt+Space avoids the Windows-reserved Win+Space layout switch.
        MOD_ALT, MOD_CONTROL, MOD_NOREPEAT, VK_SPACE = 0x0001, 0x0002, 0x4000, 0x20
        self.registered = bool(
            ctypes.windll.user32.RegisterHotKey(
                None, HOTKEY_ID, MOD_ALT | MOD_CONTROL | MOD_NOREPEAT, VK_SPACE
            )
        )
        return self.registered

    def unregister(self) -> None:
        if self.registered and os.name == "nt":
            ctypes.windll.user32.UnregisterHotKey(None, HOTKEY_ID)
        self.registered = False

    def nativeEventFilter(self, event_type, message):  # noqa: N802, ANN001
        if os.name == "nt" and "windows" in bytes(event_type).decode(errors="ignore"):
            from ctypes import wintypes

            msg = wintypes.MSG.from_address(int(message))
            if msg.message == 0x0312 and msg.wParam == HOTKEY_ID:  # WM_HOTKEY
                self.activated.emit()
                return True, 0
        return False, 0


class SingleInstanceGuard(QObject):
    """Use a Windows named pipe (QLocalServer) to keep one resident client."""

    activateRequested = Signal()

    def __init__(self, name: str = "SUBPC-BUDDY-desktop-v1", parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.name = name
        self.server: QLocalServer | None = None

    def acquire(self) -> bool:
        probe = QLocalSocket()
        probe.connectToServer(self.name)
        if probe.waitForConnected(250):
            probe.write(b"show\n")
            probe.waitForBytesWritten(250)
            probe.disconnectFromServer()
            return False

        # A crashed process can leave a stale Unix socket. On Windows this is a
        # harmless no-op for an absent named pipe.
        QLocalServer.removeServer(self.name)
        self.server = QLocalServer(self)
        if not self.server.listen(self.name):
            self.server = None
            retry = QLocalSocket()
            retry.connectToServer(self.name)
            if retry.waitForConnected(500):
                retry.write(b"show\n")
                retry.waitForBytesWritten(250)
                retry.disconnectFromServer()
                return False
            return True
        self.server.newConnection.connect(self._accept_connections)
        return True

    def _accept_connections(self) -> None:
        if self.server is None:
            return
        while self.server.hasPendingConnections():
            connection = self.server.nextPendingConnection()
            if connection is not None:
                connection.disconnectFromServer()
                connection.deleteLater()
            self.activateRequested.emit()

    def close(self) -> None:
        if self.server is not None:
            self.server.close()
            self.server = None
        QLocalServer.removeServer(self.name)


class OverlayClickThroughController:
    """Wire DesktopBridge.overlayClickThroughRequested to the real overlay window.

    Holds a mutable overlay-window reference that is assigned after the overlay
    QML engine loads and stays reachable from the global hotkey path. Applying
    click-through True is authoritative: a failed native call, a missing window,
    or a zero hwnd immediately resets the bridge desired state to False without
    recursion. Disabling is best-effort and never triggers a reset. Call
    disconnect() before teardown so late requests are ignored safely.
    """

    def __init__(self, bridge: DesktopBridge) -> None:
        self._bridge = bridge
        self._overlay_window = None
        self._resetting = False
        bridge.overlayClickThroughRequested.connect(self._on_requested)

    def set_overlay_window(self, window) -> None:  # noqa: ANN001
        """Attach or replace the live overlay window reference."""
        self._overlay_window = window

    @property
    def overlay_window(self):  # noqa: ANN202
        return self._overlay_window

    def apply_default_click_through(self) -> None:
        """Best-effort apply of the default click-through (False) at creation."""
        hwnd = self._window_hwnd()
        if hwnd != 0:
            apply_click_through(hwnd, False)

    def restore_interaction(self) -> None:
        """Request click-through off so the overlay regains mouse input."""
        self._bridge.setOverlayClickThrough(False)

    def disconnect(self) -> None:
        """Detach from the bridge signal (safe on overlay failure/shutdown)."""
        try:
            self._bridge.overlayClickThroughRequested.disconnect(self._on_requested)
        except (RuntimeError, TypeError):
            pass

    def _window_hwnd(self) -> int:
        window = self._overlay_window
        if window is None:
            return 0
        try:
            return int(window.winId())
        except Exception:
            return 0

    def _on_requested(self, enabled: bool) -> None:
        hwnd = self._window_hwnd()
        applied = hwnd != 0 and apply_click_through(hwnd, enabled)
        if enabled and not applied:
            self._reset_desired_state()

    def _reset_desired_state(self) -> None:
        if self._resetting:
            return
        self._resetting = True
        try:
            self._bridge.setOverlayClickThrough(False)
        finally:
            self._resetting = False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SUBPC BUDDY Windows desktop client")
    parser.add_argument("--server", help="Backend URL, e.g. http://100.x.x.x:8000")
    parser.add_argument("--hidden", action="store_true", help="Start in the system tray")
    parser.add_argument("--no-tray", action="store_true", help="Disable tray integration")
    parser.add_argument("--no-overlay", action="store_true", help="Disable overlay even if env is set")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    QApplication.setApplicationName("SUBPC BUDDY")
    QApplication.setOrganizationName("SUBPC Living")
    app = QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)

    instance_guard = SingleInstanceGuard(parent=app)
    if not args.smoke_test and not instance_guard.acquire():
        return 0

    settings = DesktopSettings.load()
    if args.server:
        settings.server_url = args.server
        settings.session_id = ""
        try:
            settings.save()
        except OSError:
            pass

    # QML keeps a raw reference to this context object. Parenting it to the
    # application guarantees that it outlives every engine/root-object teardown.
    bridge = DesktopBridge(settings, parent=app, offline=args.smoke_test)
    click_through = OverlayClickThroughController(bridge)
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("bridge", bridge)
    qml_file = resource_path("src/desktop/qml/Main.qml")
    engine.addImportPath(str(qml_file.parent))
    engine.load(QUrl.fromLocalFile(str(qml_file)))
    if not engine.rootObjects():
        bridge.shutdown()
        return 1
    window = engine.rootObjects()[0]
    instance_guard.activateRequested.connect(lambda: _show_window(window))

    icon_path = resource_path("src/desktop/assets/app-icon.svg")
    icon = QIcon(str(icon_path))
    app.setWindowIcon(icon)
    window.setIcon(icon)

    tray: QSystemTrayIcon | None = None
    if not args.no_tray and QSystemTrayIcon.isSystemTrayAvailable():
        tray = QSystemTrayIcon(icon, app)
        tray.setToolTip("SUBPC BUDDY")
        menu = QMenu()
        show_action = QAction("SUBPC BUDDYを開く", menu)
        show_action.triggered.connect(lambda: _show_window(window))
        menu.addAction(show_action)
        for index, label in enumerate(("話す", "やること", "記録", "実績")):
            action = QAction(label, menu)
            action.triggered.connect(
                lambda _checked=False, page=index: (_set_page(window, page), _show_window(window))
            )
            menu.addAction(action)
        menu.addSeparator()
        quit_action = QAction("終了", menu)
        quit_action.triggered.connect(app.quit)
        menu.addAction(quit_action)
        tray.setContextMenu(menu)
        tray.activated.connect(
            lambda reason: _toggle_window(window)
            if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick)
            else None
        )
        bridge.nativeNotification.connect(
            lambda title, body: tray.showMessage(title, body, QSystemTrayIcon.MessageIcon.Information)
        )
        tray.show()

    # Without a tray there is nowhere to restore a hidden window from.
    app.setQuitOnLastWindowClosed(tray is None)

    window.setProperty("closeToTray", bool(tray and settings.close_to_tray))
    hotkey = WindowsHotkeyFilter(app)
    app.installNativeEventFilter(hotkey)
    hotkey_registered = hotkey.register()
    hotkey.activated.connect(lambda: _on_hotkey_activated(click_through, window))

    app.aboutToQuit.connect(hotkey.unregister)
    app.aboutToQuit.connect(instance_guard.close)
    app.aboutToQuit.connect(bridge.shutdown)
    app.aboutToQuit.connect(click_through.disconnect)
    QTimer.singleShot(0, bridge.initialize)
    QTimer.singleShot(250, lambda: apply_windows_backdrop(int(window.winId())))
    if os.name == "nt" and not hotkey_registered:
        QTimer.singleShot(
            500,
            lambda: bridge.toast.emit(
                "グローバルショートカットを登録できません",
                "Ctrl+Alt+Spaceを別のアプリが使用している可能性があります",
            ),
        )

    # --- Overlay window (Phase 6a) ---
    overlay_engine = None
    overlay_window = None
    if not args.no_overlay:
        overlay_on = os.environ.get("DESKTOP_OVERLAY_ENABLED", "").strip().lower() == "true"
        if overlay_on:
            try:
                overlay_engine = QQmlApplicationEngine()
                overlay_engine.rootContext().setContextProperty("overlayBridge", bridge)
                overlay_qml = resource_path("src/desktop/qml/Overlay.qml")
                overlay_engine.addImportPath(str(overlay_qml.parent))
                overlay_engine.load(QUrl.fromLocalFile(str(overlay_qml)))
                if overlay_engine.rootObjects():
                    overlay_window = overlay_engine.rootObjects()[0]
                    click_through.set_overlay_window(overlay_window)
                    bridge.mainWindowRequested.connect(lambda: _show_window(window))
                    QTimer.singleShot(250, click_through.apply_default_click_through)
            except Exception:
                overlay_engine = None
                overlay_window = None
                click_through.set_overlay_window(None)
                click_through.disconnect()

    if args.hidden or settings.start_hidden:
        window.hide()
    else:
        _show_window(window)
    if args.smoke_test:
        QTimer.singleShot(2_000, app.quit)
    return app.exec()


def _on_hotkey_activated(controller: OverlayClickThroughController, window) -> None:  # noqa: ANN001
    """Ctrl+Alt+Space: restore overlay interaction, then toggle the main window."""
    controller.restore_interaction()
    _toggle_window(window)


def _show_window(window) -> None:  # noqa: ANN001
    window.show()
    window.raise_()
    window.requestActivate()


def _toggle_window(window) -> None:  # noqa: ANN001
    if window.isVisible() and window.isActive():
        window.hide()
    else:
        _show_window(window)


def _set_page(window, page: int) -> None:  # noqa: ANN001
    window.setProperty("currentPage", page)


if __name__ == "__main__":
    raise SystemExit(main())
