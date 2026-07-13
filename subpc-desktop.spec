# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.compat import is_win

datas = [
    ("src/desktop/qml", "src/desktop/qml"),
    ("src/web/static/favicon.svg", "src/web/static"),
    ("src/desktop/assets", "src/desktop/assets"),
]
datas += collect_data_files("PySide6", includes=["**/translations/*"])

a = Analysis(
    ["scripts/run_desktop.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=["PySide6.QtWebSockets", "PySide6.QtMultimedia"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="SUBPC-BUDDY",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="src/desktop/assets/app-icon.ico" if is_win else None,
    version="src/desktop/windows-version-info.txt" if is_win else None,
)
