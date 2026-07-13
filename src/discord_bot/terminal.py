"""tmux-backed Discord terminal bridge."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
import time


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class TerminalResult:
    ok: bool
    text: str


class DiscordTerminal:
    """Small wrapper around one tmux session per Discord channel."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        workdir: str | Path,
        session_prefix: str = "subpc_discord",
        capture_lines: int = 80,
        command_prefix: str = "!t",
    ):
        self.enabled = enabled
        self.workdir = Path(workdir).expanduser()
        self.session_prefix = session_prefix
        self.capture_lines = max(10, min(capture_lines, 300))
        self.command_prefix = command_prefix

    def is_available(self) -> bool:
        return shutil.which("tmux") is not None

    def handle_message(self, channel_id: int, content: str) -> TerminalResult | None:
        content = content.strip()
        if not content:
            return None
        if not content.startswith(self.command_prefix):
            return None
        payload = content[len(self.command_prefix) :].strip()
        if not payload:
            return TerminalResult(True, self.help_text())
        return self.handle_command(channel_id, payload)

    def handle_command(self, channel_id: int, payload: str) -> TerminalResult:
        if not self.enabled:
            return TerminalResult(False, "terminal is disabled. Set DISCORD_ALLOW_TERMINAL=true.")
        if not self.is_available():
            return TerminalResult(False, "tmux が見つかりません。`sudo apt install tmux` が必要です。")

        command, _, arg = payload.partition(" ")
        command_lower = command.lower()
        if command_lower in {"help", "h", "?"}:
            return TerminalResult(True, self.help_text())
        if command_lower in {"capture", "cap", "show"}:
            lines = self._parse_lines(arg) if arg else self.capture_lines
            return TerminalResult(True, self.capture(channel_id, lines=lines))
        if command_lower in {"enter"}:
            self.send_keys(channel_id, "Enter")
            time.sleep(0.2)
            return TerminalResult(True, self.capture(channel_id, lines=30))
        if command_lower in {"submit"}:
            self.send_keys(channel_id, "C-j")
            time.sleep(0.2)
            return TerminalResult(True, self.capture(channel_id, lines=30))
        if command_lower in {"key", "keys"}:
            if not arg.strip():
                return TerminalResult(False, "usage: !t key <tmux-key> [tmux-key ...]")
            for key in arg.split():
                if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
                    return TerminalResult(False, f"invalid key: {key}")
                self.send_keys(channel_id, key)
            time.sleep(0.2)
            return TerminalResult(True, self.capture(channel_id, lines=30))
        if command_lower in {"type", "paste"}:
            if not arg:
                return TerminalResult(False, f"usage: {self.command_prefix} {command_lower} <text>")
            self.paste_text(channel_id, arg)
            time.sleep(0.1)
            return TerminalResult(True, self.capture(channel_id, lines=30))
        if command_lower in {"reset", "new"}:
            self.reset(channel_id)
            return TerminalResult(True, self.capture(channel_id, lines=20))
        if command_lower in {"interrupt", "ctrl-c", "ctrlc"}:
            self.send_keys(channel_id, "C-c")
            time.sleep(0.2)
            return TerminalResult(True, self.capture(channel_id, lines=30))
        if command_lower in {"clear"}:
            self.send_keys(channel_id, "C-l")
            time.sleep(0.1)
            return TerminalResult(True, self.capture(channel_id, lines=20))

        self.send_command(channel_id, payload)
        time.sleep(0.5)
        return TerminalResult(True, self.capture(channel_id, lines=self.capture_lines))

    def help_text(self) -> str:
        return (
            "terminal commands:\n"
            f"{self.command_prefix} <command>    shellへ送信\n"
            f"{self.command_prefix} capture [n]  最新n行を表示\n"
            f"{self.command_prefix} submit       Ctrl-Jで送信\n"
            f"{self.command_prefix} enter        Enter\n"
            f"{self.command_prefix} key <key>    tmux keyを送信\n"
            f"{self.command_prefix} paste <text> 文字だけ貼り付け\n"
            f"{self.command_prefix} interrupt    Ctrl-C\n"
            f"{self.command_prefix} reset        tmux sessionを作り直す\n"
            f"{self.command_prefix} clear        画面クリア"
        )

    def send_command(self, channel_id: int, command: str) -> None:
        self._ensure_session(channel_id)
        submit_key = self._default_submit_key(channel_id)
        if submit_key == "C-j":
            self.send_keys(channel_id, "C-u")
            time.sleep(0.05)
        self.paste_text(channel_id, command)
        self.send_keys(channel_id, submit_key)

    def send_keys(self, channel_id: int, key: str) -> None:
        self._ensure_session(channel_id)
        self._run(["tmux", "send-keys", "-t", self._session_name(channel_id), key])

    def paste_text(self, channel_id: int, text: str) -> None:
        self._ensure_session(channel_id)
        buffer_name = f"{self.session_prefix}_{channel_id}_input"
        self._run(["tmux", "set-buffer", "-b", buffer_name, text])
        self._run(["tmux", "paste-buffer", "-b", buffer_name, "-t", self._session_name(channel_id)])
        self._run(["tmux", "delete-buffer", "-b", buffer_name], check=False)

    def capture(self, channel_id: int, *, lines: int | None = None) -> str:
        self._ensure_session(channel_id)
        lines = lines or self.capture_lines
        start = max(0, lines)
        base_cmd = [
            "tmux",
            "capture-pane",
            "-p",
            "-J",
            "-t",
            self._session_name(channel_id),
            "-S",
            f"-{start}",
        ]
        proc = self._run(["tmux", "capture-pane", "-a", *base_cmd[2:]], check=False)
        if proc.returncode != 0:
            proc = self._run(base_cmd, check=False)
        return clean_terminal_output(proc.stdout)

    def pane_command(self, channel_id: int) -> str:
        self._ensure_session(channel_id)
        proc = self._run(
            [
                "tmux",
                "display-message",
                "-p",
                "-t",
                self._session_name(channel_id),
                "#{pane_current_command}",
            ],
            check=False,
        )
        return proc.stdout.strip()

    def reset(self, channel_id: int) -> None:
        name = self._session_name(channel_id)
        self._run(["tmux", "kill-session", "-t", name], check=False)
        self._ensure_session(channel_id)

    def _ensure_session(self, channel_id: int) -> None:
        name = self._session_name(channel_id)
        exists = self._run(["tmux", "has-session", "-t", name], check=False)
        if exists.returncode == 0:
            return
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._run(
            [
                "tmux",
                "new-session",
                "-d",
                "-x",
                "160",
                "-y",
                "40",
                "-s",
                name,
                "-c",
                str(self.workdir),
                "bash",
            ]
        )

    def _session_name(self, channel_id: int) -> str:
        return f"{self.session_prefix}_{channel_id}"

    def _default_submit_key(self, channel_id: int) -> str:
        command = self.pane_command(channel_id)
        if command in {"codex", "node"}:
            return "C-j"
        return "Enter"

    def _run(self, cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            cwd=self.workdir if self.workdir.exists() else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=15,
            check=check,
        )

    @staticmethod
    def _parse_lines(value: str) -> int:
        try:
            return max(10, min(int(value.strip()), 300))
        except ValueError:
            return 80


def clean_terminal_output(text: str) -> str:
    text = ANSI_RE.sub("", text)
    text = CONTROL_RE.sub("", text)
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) or "(no output)"
