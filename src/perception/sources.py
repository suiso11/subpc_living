"""プラットフォーム別 ActivitySource adapter。

オプトインで OS の「アクティブなアプリのプロセス名」と「idle ミリ秒」を読み、
プロセス名を直ちに AppCategory へ分類し、ActivitySample だけを返す。
生の window title・text・path・keystroke・screenshot・camera データは取得も
保持もせず、プロセス名は分類後はどこにも保存しない。

- Windows: ctypes の GetLastInputInfo / GetForegroundWindow で idle と
  foreground PID を取得し、プロセス名は psutil で解決する。
- Linux/X11: xprintidle / xdotool を短いタイムアウトで実行し、PID は psutil で解決する。
- 施設が利用できない場合は ActivitySourceUnavailableError を上げて明確に失敗する。

コンストラクタは読み取り器の束縛と検証だけを行う。OS へのアクセスは sample() を
呼ぶまで始まらない。読み取り器は注入・差し替え可能で、テストでは OS アクセスを
モックできる。
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from src.perception.activity import AppCategory, VALID_APP_CATEGORIES, ActivitySample

IdleMillisReader = Callable[[], float]
ForegroundPidReader = Callable[[], int]
ProcessNameReader = Callable[[int], str]


class ActivitySourceUnavailableError(RuntimeError):
    """OS の活動取得施設が利用できないときに上げる。"""


@runtime_checkable
class ActivitySource(Protocol):
    """ActivitySample を返す収集源の契約。

    生のアプリ名・window title・text・path・pid・raw input は公開せず、
    分類済みの最小値だけを返す。concrete 型に依存せず構造的適合で受け取れる。
    """

    def sample(self) -> ActivitySample: ...


@runtime_checkable
class AppClassifier(Protocol):
    """プロセス名を AppCategory へ分類する契約。"""

    def classify(self, process_name: str) -> AppCategory: ...


_DEFAULT_CATEGORY_MAP: dict[str, AppCategory] = {
    # work: 開発・編集・ターミナル
    "code": "work",
    "cursor": "work",
    "vscodium": "work",
    "pycharm": "work",
    "pycharm64": "work",
    "idea": "work",
    "idea64": "work",
    "webstorm": "work",
    "goland": "work",
    "vim": "work",
    "nvim": "work",
    "emacs": "work",
    "sublime_text": "work",
    "gnome-terminal": "work",
    "konsole": "work",
    "terminator": "work",
    "x-terminal-emulator": "work",
    "kitty": "work",
    "alacritty": "work",
    "wezterm": "work",
    "windows_terminal": "work",
    "powershell": "work",
    "pwsh": "work",
    "cmd": "work",
    "python": "work",
    "python3": "work",
    "node": "work",
    # communication
    "discord": "communication",
    "slack": "communication",
    "teams": "communication",
    "zoom": "communication",
    "line": "communication",
    "skype": "communication",
    "telegram": "communication",
    "whatsapp": "communication",
    "signal": "communication",
    # media
    "spotify": "media",
    "vlc": "media",
    "mpv": "media",
    "rhythmbox": "media",
    "audacious": "media",
    "cmus": "media",
    # system
    "explorer": "system",
    "taskmgr": "system",
    "gnome-shell": "system",
    "kwin_x11": "system",
    "plasmashell": "system",
    "xfce4-session": "system",
    "systemsettings": "system",
}


class ProcessNameClassifier:
    """控えめな組み込み分類器。

    既知のプロセス名だけをカテゴリへ分類し、一致しないものは unknown を返す。
    名前は小文字化して ".exe" を取り除いてから照合する (Windows/Linux 共通)。
    """

    def __init__(self, mapping: dict[str, AppCategory] | None = None) -> None:
        table = _DEFAULT_CATEGORY_MAP if mapping is None else mapping
        for name, category in table.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"process name must be a non-empty str, got {name!r}")
            if category not in VALID_APP_CATEGORIES:
                raise ValueError(f"invalid category {category!r} for process {name!r}")
        self._table = dict(table)

    def classify(self, process_name: str) -> AppCategory:
        if not isinstance(process_name, str):
            return "unknown"
        key = process_name.strip().lower()
        if key.endswith(".exe"):
            key = key[:-4]
        return self._table.get(key, "unknown")


def _windll():
    import ctypes

    return getattr(ctypes, "windll", None)


def _windows_idle_millis_reader() -> float:
    """GetLastInputInfo の tick 差 (ミリ秒) を返す。ラップアラウンドは符号なし32bitで扱う。"""
    import ctypes

    windll = _windll()
    if windll is None:
        raise ActivitySourceUnavailableError("ctypes.windll unavailable (requires Windows)")
    user32 = windll.user32
    kernel32 = windll.kernel32
    kernel32.GetTickCount.restype = ctypes.c_uint32

    class LastInputInfo(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

    info = LastInputInfo()
    info.cbSize = ctypes.sizeof(LastInputInfo)
    if not user32.GetLastInputInfo(ctypes.byref(info)):
        raise ActivitySourceUnavailableError("GetLastInputInfo failed")
    now = int(kernel32.GetTickCount())
    last = int(info.dwTime)
    return float((now - last) & 0xFFFFFFFF)


def _windows_foreground_pid_reader() -> int:
    """GetForegroundWindow / GetWindowThreadProcessId で foreground PID を返す。"""
    import ctypes

    windll = _windll()
    if windll is None:
        raise ActivitySourceUnavailableError("ctypes.windll unavailable (requires Windows)")
    user32 = windll.user32
    hwnd = int(user32.GetForegroundWindow())
    if not hwnd:
        raise ActivitySourceUnavailableError("no foreground window")
    pid = ctypes.c_ulong()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if not pid.value:
        raise ActivitySourceUnavailableError("no pid for foreground window")
    return int(pid.value)


def _run_capture_or_unavailable(label: str, cmd: list[str], timeout: float) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise ActivitySourceUnavailableError(f"{label}: executable not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise ActivitySourceUnavailableError(f"{label}: timed out") from exc
    if proc.returncode != 0:
        raise ActivitySourceUnavailableError(f"{label}: exit code {proc.returncode}")
    stdout = (proc.stdout or "").strip()
    if not stdout:
        raise ActivitySourceUnavailableError(f"{label}: empty output")
    return stdout


def _linux_idle_millis_reader(timeout: float = 2.0) -> float:
    """xprintidle の出力 (ミリ秒) を返す。"""
    stdout = _run_capture_or_unavailable("xprintidle", ["xprintidle"], timeout)
    try:
        value = float(stdout)
    except ValueError as exc:
        raise ActivitySourceUnavailableError("xprintidle: non-numeric output") from exc
    if value < 0:
        raise ActivitySourceUnavailableError("xprintidle: negative idle value")
    return value


def _linux_foreground_pid_reader(timeout: float = 2.0) -> int:
    """xdotool getactivewindow getwindowpid の出力 (PID) を返す。"""
    stdout = _run_capture_or_unavailable(
        "xdotool", ["xdotool", "getactivewindow", "getwindowpid"], timeout
    )
    try:
        pid = int(stdout)
    except ValueError as exc:
        raise ActivitySourceUnavailableError("xdotool: non-numeric output") from exc
    if pid <= 0:
        raise ActivitySourceUnavailableError("xdotool: invalid pid")
    return pid


def _process_name_reader(pid: int) -> str:
    """psutil で PID からプロセス名だけを返す。path や実行ファイルは取得しない。"""
    try:
        import psutil
    except ImportError as exc:
        raise ActivitySourceUnavailableError("psutil not available") from exc
    try:
        return psutil.Process(pid).name()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as exc:
        raise ActivitySourceUnavailableError(
            f"cannot resolve process name: {type(exc).__name__}"
        ) from exc


class _ProcessActivitySource:
    """プロセス名を分類し idle 秒と合わせて ActivitySample を返す共通実装。

    コンストラクタは読み取り器の束縛と検証だけを行い、OS アクセスはしない。
    sample() を呼ぶまで収集は始まらない。生のプロセス名は保存しない。
    """

    def __init__(
        self,
        *,
        idle_millis_reader: IdleMillisReader,
        foreground_pid_reader: ForegroundPidReader,
        process_name_reader: ProcessNameReader,
        classifier: AppClassifier | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._idle_millis_reader = idle_millis_reader
        self._foreground_pid_reader = foreground_pid_reader
        self._process_name_reader = process_name_reader
        self._classifier = classifier if classifier is not None else ProcessNameClassifier()
        self._clock = clock if clock is not None else time.time

    def sample(self) -> ActivitySample:
        idle_ms = self._idle_millis_reader()
        pid = self._foreground_pid_reader()
        process_name = self._process_name_reader(pid)
        try:
            category = self._classifier.classify(process_name)
        except Exception as exc:
            raise ActivitySourceUnavailableError(
                f"classifier failed: {type(exc).__name__}"
            ) from exc
        return ActivitySample(
            timestamp=self._clock(),
            idle_seconds=idle_ms / 1000.0,
            app_category=category,
        )


class WindowsActivitySource(_ProcessActivitySource):
    """Windows 用 ActivitySource。ctypes + psutil を使う。"""

    def __init__(
        self,
        *,
        idle_millis_reader: IdleMillisReader = _windows_idle_millis_reader,
        foreground_pid_reader: ForegroundPidReader = _windows_foreground_pid_reader,
        process_name_reader: ProcessNameReader = _process_name_reader,
        classifier: AppClassifier | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        super().__init__(
            idle_millis_reader=idle_millis_reader,
            foreground_pid_reader=foreground_pid_reader,
            process_name_reader=process_name_reader,
            classifier=classifier,
            clock=clock,
        )


class LinuxActivitySource(_ProcessActivitySource):
    """Linux/X11 用 ActivitySource。xprintidle / xdotool + psutil を使う。"""

    def __init__(
        self,
        *,
        idle_millis_reader: IdleMillisReader = _linux_idle_millis_reader,
        foreground_pid_reader: ForegroundPidReader = _linux_foreground_pid_reader,
        process_name_reader: ProcessNameReader = _process_name_reader,
        classifier: AppClassifier | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        super().__init__(
            idle_millis_reader=idle_millis_reader,
            foreground_pid_reader=foreground_pid_reader,
            process_name_reader=process_name_reader,
            classifier=classifier,
            clock=clock,
        )


_PLATFORM_ALIASES: dict[str, str] = {
    "windows": "windows",
    "win32": "windows",
    "nt": "windows",
    "linux": "linux",
    "linux2": "linux",
}


def _detect_platform() -> str:
    alias = _PLATFORM_ALIASES.get(sys.platform)
    if alias is None:
        raise ActivitySourceUnavailableError(
            f"unsupported platform: {sys.platform!r} (supported: windows, linux)"
        )
    return alias


def create_activity_source(
    platform: str | None = None,
    *,
    classifier: AppClassifier | None = None,
    idle_millis_reader: IdleMillisReader | None = None,
    foreground_pid_reader: ForegroundPidReader | None = None,
    process_name_reader: ProcessNameReader | None = None,
    clock: Callable[[], float] | None = None,
) -> ActivitySource:
    """プラットフォームに応じた ActivitySource を返す factory。

    platform は "windows" / "linux" (別名 win32 / nt / linux2 も可)。None なら
    sys.platform から自動判定する。明示指定が未対応なら ValueError、
    自動判定が未対応なら ActivitySourceUnavailableError を上げる。
    コンストラクタは OS アクセスを一切しない。sample() を呼ぶまで収集は始まらない。
    """
    if platform is None:
        selected = _detect_platform()
    else:
        if not isinstance(platform, str) or not platform.strip():
            raise ValueError(
                f"platform must be a non-empty str, got {platform!r} "
                "(supported: windows, linux)"
            )
        alias = _PLATFORM_ALIASES.get(platform.strip().lower())
        if alias is None:
            raise ValueError(
                f"unsupported platform: {platform!r} (supported: windows, linux)"
            )
        selected = alias

    if selected == "windows":
        return WindowsActivitySource(
            idle_millis_reader=idle_millis_reader or _windows_idle_millis_reader,
            foreground_pid_reader=foreground_pid_reader or _windows_foreground_pid_reader,
            process_name_reader=process_name_reader or _process_name_reader,
            classifier=classifier,
            clock=clock,
        )
    return LinuxActivitySource(
        idle_millis_reader=idle_millis_reader or _linux_idle_millis_reader,
        foreground_pid_reader=foreground_pid_reader or _linux_foreground_pid_reader,
        process_name_reader=process_name_reader or _process_name_reader,
        classifier=classifier,
        clock=clock,
    )