"""Windows shell integration kept separate so non-Windows tests can import it."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

APP_RUN_NAME = "SUBPC BUDDY"
HOTKEY_ID = 0xBADD


def startup_command() -> str:
    if getattr(sys, "frozen", False):
        args = [str(Path(sys.executable).resolve()), "--hidden"]
    else:
        args = [str(Path(sys.executable).resolve()), "-m", "src.desktop", "--hidden"]
    return subprocess.list2cmdline(args)


def is_autostart_enabled() -> bool:
    if os.name != "nt":
        return False
    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
        ) as key:
            value, _ = winreg.QueryValueEx(key, APP_RUN_NAME)
        return bool(value)
    except OSError:
        return False


def set_autostart(enabled: bool) -> bool:
    if os.name != "nt":
        return False
    import winreg

    with winreg.CreateKey(
        winreg.HKEY_CURRENT_USER,
        r"Software\Microsoft\Windows\CurrentVersion\Run",
    ) as key:
        if enabled:
            winreg.SetValueEx(key, APP_RUN_NAME, 0, winreg.REG_SZ, startup_command())
        else:
            try:
                winreg.DeleteValue(key, APP_RUN_NAME)
            except FileNotFoundError:
                pass
    return is_autostart_enabled() == enabled


def apply_windows_backdrop(window_id: int) -> bool:
    """Enable Windows 11 dark Mica. Older Windows safely ignores this."""
    if os.name != "nt" or not window_id:
        return False
    import ctypes

    dark = ctypes.c_int(1)
    backdrop = ctypes.c_int(2)  # DWMSBT_MAINWINDOW
    dwm = ctypes.windll.dwmapi
    ok_dark = dwm.DwmSetWindowAttribute(
        ctypes.c_void_p(window_id), 20, ctypes.byref(dark), ctypes.sizeof(dark)
    ) == 0
    ok_backdrop = dwm.DwmSetWindowAttribute(
        ctypes.c_void_p(window_id), 38, ctypes.byref(backdrop), ctypes.sizeof(backdrop)
    ) == 0
    return bool(ok_dark or ok_backdrop)
