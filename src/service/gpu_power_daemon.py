"""
root 側 GPU 電力制御デーモン

ユーザー権限で動く Web UI / 音声パイプラインからの要求を
ローカル UNIX ソケットで受け取り、nvidia-smi の権限操作を代理実行する。
"""
from __future__ import annotations

import argparse
import grp
import json
import os
import signal
import socket
import stat
from pathlib import Path

from src.service.power import DEFAULT_POWERD_SOCKET, create_all_managers


class GpuPowerDaemon:
    """GPU 電力制御デーモン"""

    def __init__(
        self,
        socket_path: str = DEFAULT_POWERD_SOCKET,
        socket_mode: int = 0o660,
        socket_group: str | None = None,
        enable_persistence: bool = False,
        initial_mode: str | None = None,
    ):
        self.socket_path = Path(socket_path)
        self.socket_mode = socket_mode
        self.socket_group = socket_group
        self.enable_persistence = enable_persistence
        self.initial_mode = initial_mode
        self._server: socket.socket | None = None
        self._running = False
        self._managers = create_all_managers()

    def _prepare_socket(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)

        if self.socket_path.exists():
            st = self.socket_path.stat()
            if not stat.S_ISSOCK(st.st_mode):
                raise RuntimeError(f"{self.socket_path} はソケットではありません")
            self.socket_path.unlink()

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.socket_path))
        server.listen(16)
        server.settimeout(1.0)
        os.chmod(self.socket_path, self.socket_mode)

        if self.socket_group:
            gid = grp.getgrnam(self.socket_group).gr_gid
            os.chown(self.socket_path, 0, gid)

        self._server = server

    def _cleanup_socket(self) -> None:
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None

        try:
            if self.socket_path.exists() and stat.S_ISSOCK(self.socket_path.stat().st_mode):
                self.socket_path.unlink()
        except OSError:
            pass

    def _get_manager(self, gpu_id: int):
        for mgr in self._managers:
            if mgr.gpu_id == gpu_id:
                return mgr
        return None

    def _set_initial_state(self) -> None:
        if self.enable_persistence:
            for mgr in self._managers:
                ok, msg = mgr.set_persistence_mode(True)
                print(f"[powerd] GPU{mgr.gpu_id} persistence: {'ok' if ok else 'ng'} ({msg})")

        if self.initial_mode == "idle":
            for mgr in self._managers:
                ok, msg = mgr.set_idle_mode()
                print(f"[powerd] GPU{mgr.gpu_id} idle: {'ok' if ok else 'ng'} ({msg})")
        elif self.initial_mode == "active":
            for mgr in self._managers:
                ok, msg = mgr.set_active_mode()
                print(f"[powerd] GPU{mgr.gpu_id} active: {'ok' if ok else 'ng'} ({msg})")

    def _recv_json(self, conn: socket.socket) -> dict:
        chunks: list[bytes] = []
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break

        if not chunks:
            raise ValueError("empty request")

        raw = b"".join(chunks).decode("utf-8").strip()
        if len(raw) > 8192:
            raise ValueError("request too large")
        return json.loads(raw)

    def _send_json(self, conn: socket.socket, payload: dict) -> None:
        conn.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))

    def _handle_request(self, payload: dict) -> dict:
        cmd = payload.get("command")

        if cmd == "ping":
            return {"ok": True, "message": "pong"}

        if cmd == "status":
            return {
                "ok": True,
                "message": "ok",
                "gpu_count": len(self._managers),
                "gpus": [mgr.get_status() for mgr in self._managers],
            }

        if not self._managers:
            return {"ok": False, "message": "GPU が見つかりません"}

        if cmd == "set_persistence_mode":
            enable = bool(payload.get("enable", True))
            results = []
            all_ok = True
            for mgr in self._managers:
                ok, msg = mgr.set_persistence_mode(enable)
                all_ok = all_ok and ok
                results.append(f"GPU{mgr.gpu_id}: {msg}")
            return {"ok": all_ok, "message": "; ".join(results)}

        gpu_id = int(payload.get("gpu_id", 0))
        mgr = self._get_manager(gpu_id)
        if mgr is None:
            return {"ok": False, "message": f"GPU{gpu_id} が見つかりません"}

        if cmd == "set_power_limit":
            watts = int(payload.get("watts"))
            if watts < 50 or watts > 500:
                return {"ok": False, "message": f"不正な power limit: {watts}W"}
            ok, msg = mgr.set_power_limit(watts)
            return {"ok": ok, "message": msg}

        return {"ok": False, "message": f"unknown command: {cmd}"}

    def serve_forever(self) -> None:
        if not self._managers:
            print("[powerd] nvidia-smi が見つからないか、GPU が検出されません")
            return

        self._prepare_socket()
        self._set_initial_state()
        self._running = True
        print(f"[powerd] listening on {self.socket_path}")

        while self._running:
            try:
                conn, _addr = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            with conn:
                try:
                    payload = self._recv_json(conn)
                    response = self._handle_request(payload)
                except Exception as e:
                    response = {"ok": False, "message": str(e)}
                self._send_json(conn, response)

        self._cleanup_socket()

    def stop(self) -> None:
        self._running = False
        self._cleanup_socket()


def main() -> None:
    parser = argparse.ArgumentParser(description="subpc_living GPU power daemon")
    parser.add_argument("--socket-path", default=DEFAULT_POWERD_SOCKET,
                        help=f"UNIX socket path (default: {DEFAULT_POWERD_SOCKET})")
    parser.add_argument("--socket-mode", default="0660",
                        help="UNIX socket mode in octal (default: 0660)")
    parser.add_argument("--socket-group", help="UNIX socket group name")
    parser.add_argument("--enable-persistence", action="store_true",
                        help="起動時に Persistence Mode を有効化")
    parser.add_argument("--initial-mode", choices=["idle", "active"],
                        help="起動時に適用する GPU モード")
    args = parser.parse_args()

    daemon = GpuPowerDaemon(
        socket_path=args.socket_path,
        socket_mode=int(args.socket_mode, 8),
        socket_group=args.socket_group,
        enable_persistence=args.enable_persistence,
        initial_mode=args.initial_mode,
    )

    def _handle_signal(_signum, _frame):
        daemon.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    daemon.serve_forever()


if __name__ == "__main__":
    main()
