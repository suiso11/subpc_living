"""Minimal stdio MCP client for one-shot tool calls.

The project only needs a small MCP client surface right now: initialize a
server process, call one tool, then shut it down. Keeping this local avoids
pulling in another runtime dependency for the Discord service.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Sequence


class MCPStdioError(RuntimeError):
    """Raised when a stdio MCP server cannot be called."""


class MCPStdioClient:
    """Tiny JSON-RPC-over-stdio MCP client."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        timeout_sec: float = 45.0,
    ):
        if not command:
            raise ValueError("command is required")
        self.command = list(command)
        self.env = env or {}
        self.cwd = str(cwd) if cwd is not None else None
        self.timeout_sec = timeout_sec

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Start the server, call a tool, and return the raw MCP result."""

        proc_env = os.environ.copy()
        proc_env.update(self.env)
        proc = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=proc_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdin is not None
        assert proc.stdout is not None
        assert proc.stderr is not None

        messages: queue.Queue[dict[str, Any] | Exception | None] = queue.Queue()
        stderr_chunks: list[bytes] = []

        stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(proc.stdout, messages),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(proc.stderr, stderr_chunks),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        try:
            self._send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "subpc_living",
                            "version": "0.1",
                        },
                    },
                },
            )
            self._wait_for_response(proc, messages, 1, stderr_chunks)
            self._send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                },
            )
            self._send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": name,
                        "arguments": arguments,
                    },
                },
            )
            return self._wait_for_response(proc, messages, 2, stderr_chunks)
        finally:
            self._terminate(proc)

    @staticmethod
    def _send(proc: subprocess.Popen, message: dict[str, Any]) -> None:
        if proc.stdin is None:
            raise MCPStdioError("MCP server stdin is closed")
        # MCP stdio transport は改行区切りJSON (LSP風の Content-Length フレーミングではない)
        payload = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        proc.stdin.write(payload + b"\n")
        proc.stdin.flush()

    def _wait_for_response(
        self,
        proc: subprocess.Popen,
        messages: "queue.Queue[dict[str, Any] | Exception | None]",
        expected_id: int,
        stderr_chunks: list[bytes],
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_sec
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MCPStdioError(self._format_error("MCP call timed out", stderr_chunks))

            try:
                item = messages.get(timeout=min(remaining, 0.25))
            except queue.Empty:
                if proc.poll() is not None:
                    raise MCPStdioError(
                        self._format_error(
                            f"MCP server exited with code {proc.returncode}",
                            stderr_chunks,
                        )
                    )
                continue

            if item is None:
                raise MCPStdioError(self._format_error("MCP server stdout closed", stderr_chunks))
            if isinstance(item, Exception):
                raise MCPStdioError(self._format_error(str(item), stderr_chunks))
            if item.get("id") != expected_id:
                continue
            if "error" in item:
                raise MCPStdioError(self._format_error(json.dumps(item["error"], ensure_ascii=False), stderr_chunks))
            result = item.get("result", {})
            if not isinstance(result, dict):
                raise MCPStdioError(f"Unexpected MCP response: {result!r}")
            if result.get("isError"):
                raise MCPStdioError(self._tool_error_message(result))
            return result

    @staticmethod
    def _tool_error_message(result: dict[str, Any]) -> str:
        """MCP tool が content で返したエラー文を取り出す。"""
        messages: list[str] = []
        content = result.get("content", [])
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "text":
                    continue
                value = item.get("text")
                if isinstance(value, str) and value.strip():
                    messages.append(value.strip())
        return "\n".join(messages) or "MCP tool returned an error"

    @staticmethod
    def _read_stdout(stream, messages: "queue.Queue[dict[str, Any] | Exception | None]") -> None:
        try:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    messages.put(json.loads(line.decode("utf-8")))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # サーバーがstdoutにログ等を混ぜても壊れないよう非JSON行は無視
                    continue
            messages.put(None)
        except Exception as exc:
            messages.put(exc)

    @staticmethod
    def _read_stderr(stream, chunks: list[bytes]) -> None:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                return
            chunks.append(chunk)

    @staticmethod
    def _terminate(proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)

    @staticmethod
    def _format_error(message: str, stderr_chunks: list[bytes]) -> str:
        stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace").strip()
        if stderr:
            return f"{message}\nstderr:\n{stderr[-4000:]}"
        return message
