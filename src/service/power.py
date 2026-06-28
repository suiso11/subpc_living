"""
GPU 省電力制御モジュール
nvidia-smi を使って GPU の電力制限を動的に変更する。
常時稼働時のアイドル消費電力を抑え、推論時にフルパワーへ復帰させる。
Phase 9: GPU名を検出して適切なデフォルト値を自動設定。
Phase 10: マルチGPU対応 (P40 + RTX 2070 Super)
"""
import json
import os
import socket
import stat
import subprocess
import shutil
from typing import Optional


# GPU別の電力プリセット (idle_watts, active_watts)
GPU_POWER_PRESETS: dict[str, tuple[int, int]] = {
    "P40":             (125, 250),   # TDP 250W, min 125W
    "P100":            (125, 250),   # TDP 250W
    "V100":            (125, 300),   # TDP 300W
    "GTX 1060":        (80, 120),    # TDP 120W
    "GTX 1070":        (80, 150),    # TDP 150W
    "GTX 1080":        (80, 180),    # TDP 180W
    "RTX 2070":        (125, 215),   # TDP 215W, min 125W
    "RTX 2070 SUPER":  (125, 215),   # TDP 215W, min 125W
    "RTX 2080":        (125, 215),   # TDP 215W
    "RTX 3080":        (100, 320),   # TDP 320W
    "RTX 4090":        (100, 450),   # TDP 450W
}

DEFAULT_POWERD_SOCKET = os.environ.get("SUBPC_GPU_POWER_SOCKET", "/run/subpc-gpu-powerd.sock")


def _is_root() -> bool:
    """root 権限で動作中か"""
    return hasattr(os, "geteuid") and os.geteuid() == 0


def _socket_exists(path: str) -> bool:
    """UNIX ソケットが存在するか"""
    try:
        return stat.S_ISSOCK(os.stat(path).st_mode)
    except OSError:
        return False


def _detect_gpu_preset(gpu_id: int = 0) -> tuple[int, int]:
    """nvidia-smi からGPU名を取得し、適切なプリセットを返す"""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return (100, 250)  # デフォルト

    try:
        result = subprocess.run(
            [nvidia_smi, f"--id={gpu_id}",
             "--query-gpu=gpu_name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            gpu_name = result.stdout.strip()
            for key, preset in GPU_POWER_PRESETS.items():
                if key.lower() in gpu_name.lower():
                    return preset
    except Exception:
        pass

    return (100, 250)  # デフォルト


def _detect_gpu_count() -> int:
    """nvidia-smi で搭載GPU数を取得"""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return 0
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return len(result.stdout.strip().splitlines())
    except Exception:
        pass
    return 0


class GpuPowerManager:
    """nvidia-smi を使った GPU 電力制限管理

    Phase 10: マルチGPU対応。各GPUごとに個別管理可能。

    Args:
        idle_watts: アイドル時の電力制限 (W)。None で自動検出。
        active_watts: アクティブ時の電力制限 (W)。None で自動検出。
        gpu_id: 対象GPU ID。デフォルト 0。
    """

    def __init__(
        self,
        idle_watts: Optional[int] = None,
        active_watts: Optional[int] = None,
        gpu_id: int = 0,
    ):
        # GPU名からデフォルト値を決定
        default_idle, default_active = _detect_gpu_preset(gpu_id)
        self.idle_watts = idle_watts if idle_watts is not None else default_idle
        self.active_watts = active_watts if active_watts is not None else default_active
        self.gpu_id = gpu_id
        self._nvidia_smi = shutil.which("nvidia-smi")
        self._power_socket = DEFAULT_POWERD_SOCKET
        self._power_control_blocked = False
        self._power_control_message = "ok"

    @property
    def available(self) -> bool:
        """nvidia-smi が利用可能か"""
        return self._nvidia_smi is not None

    @property
    def power_control_available(self) -> bool:
        """GPU の電力制御を実行できるか"""
        if not self.available or self._power_control_blocked:
            return False
        return _is_root() or _socket_exists(self._power_socket)

    @property
    def power_control_message(self) -> str:
        """電力制御不可時の理由"""
        if not self.available:
            return "nvidia-smi not found"
        if self._power_control_blocked:
            return self._power_control_message
        if _is_root():
            return "ok"
        if _socket_exists(self._power_socket):
            return f"GPU power daemon 経由 ({self._power_socket})"
        return f"root/sudo 権限が必要です (または {self._power_socket} を起動)"

    def _run_smi(self, args: list[str]) -> tuple[bool, str]:
        """nvidia-smi コマンドを実行する

        Returns:
            (成功したか, 出力テキスト)
        """
        if not self.available:
            return False, "nvidia-smi not found"
        try:
            result = subprocess.run(
                [self._nvidia_smi] + args,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return True, result.stdout.strip()
            return False, result.stderr.strip() or result.stdout.strip()
        except subprocess.TimeoutExpired:
            return False, "nvidia-smi timeout"
        except Exception as e:
            return False, str(e)

    def _run_powerd(self, payload: dict) -> tuple[bool, str]:
        """root 側 GPU power daemon に制御要求を送る

        一時的なソケットエラー (daemon 再起動中など) では恒久無効化しない。
        次回呼び出しで再試行可能にする。
        """
        if not _socket_exists(self._power_socket):
            return False, f"GPU power daemon 未起動 ({self._power_socket})"

        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(5)
                sock.connect(self._power_socket)
                sock.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))

                chunks: list[bytes] = []
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    if b"\n" in chunk:
                        break

            if not chunks:
                return False, "GPU power daemon から応答がありません"

            raw = b"".join(chunks).decode("utf-8").strip()
            data = json.loads(raw)
            ok = bool(data.get("ok"))
            message = str(data.get("message", ""))
            # daemon 側の明示的な権限エラーのみ恒久ブロック扱いにする
            if not ok and ("Insufficient Permissions" in message or "権限" in message):
                self._power_control_blocked = True
                self._power_control_message = message
            return ok, message
        except Exception as e:
            # 一時エラーとして扱う。恒久無効化せず次回再試行可能に
            return False, f"GPU power daemon 接続失敗: {e}"

    def probe_power_control(self) -> tuple[bool, str]:
        """GPU 電力制御が利用可能かを確認"""
        if not self.available:
            return False, "nvidia-smi not found"
        if _is_root():
            return True, "ok"
        if not _socket_exists(self._power_socket):
            return False, f"GPU power daemon 未起動 ({self._power_socket})"
        return self._run_powerd({"command": "ping"})

    def get_gpu_info(self) -> dict:
        """GPU の現在の情報を取得する

        Returns:
            {"name", "power_draw_w", "power_limit_w", "temperature_c",
             "utilization_percent", "memory_used_mb", "memory_total_mb"}
        """
        if not self.available:
            return {"status": "unavailable", "message": "nvidia-smi not found"}

        query = "gpu_name,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used,memory.total"
        ok, output = self._run_smi([
            f"--id={self.gpu_id}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ])

        if not ok:
            return {"status": "error", "message": output}

        try:
            parts = [p.strip() for p in output.split(",")]
            return {
                "status": "ok",
                "name": parts[0],
                "power_draw_w": float(parts[1]),
                "power_limit_w": float(parts[2]),
                "temperature_c": int(parts[3]),
                "utilization_percent": int(parts[4]),
                "memory_used_mb": int(float(parts[5])),
                "memory_total_mb": int(float(parts[6])),
            }
        except (IndexError, ValueError) as e:
            return {"status": "error", "message": f"parse error: {e}", "raw": output}

    def set_power_limit(self, watts: int) -> tuple[bool, str]:
        """GPU の電力制限を設定する (root/sudo 権限が必要)

        Args:
            watts: 電力制限値 (W)

        Returns:
            (成功したか, メッセージ)
        """
        if not self.available:
            return False, "nvidia-smi not found"
        if not self.power_control_available:
            return False, self.power_control_message

        if not _is_root():
            return self._run_powerd({
                "command": "set_power_limit",
                "gpu_id": self.gpu_id,
                "watts": int(watts),
            })

        ok, output = self._run_smi([
            f"--id={self.gpu_id}",
            f"--power-limit={watts}",
        ])

        if ok:
            return True, f"GPU power limit set to {watts}W"
        if "Insufficient Permissions" in output or "permission" in output.lower():
            self._power_control_message = "root/sudo 権限が必要です"
            self._power_control_blocked = True
        return False, output

    def set_persistence_mode(self, enable: bool = True) -> tuple[bool, str]:
        """GPU の Persistence Mode を設定する (root/sudo 権限が必要)

        Persistence Mode を有効にすると GPU ドライバが常駐し、
        プロセス起動時のロード時間を短縮できる。
        """
        if not self.available:
            return False, "nvidia-smi not found"
        if not self.power_control_available:
            return False, self.power_control_message

        if not _is_root():
            return self._run_powerd({
                "command": "set_persistence_mode",
                "enable": bool(enable),
            })

        mode = "1" if enable else "0"
        ok, output = self._run_smi([f"--persistence-mode={mode}"])

        if ok:
            state = "enabled" if enable else "disabled"
            return True, f"Persistence mode {state}"
        return False, output

    def set_idle_mode(self) -> tuple[bool, str]:
        """アイドルモード: 電力制限を低く設定"""
        return self.set_power_limit(self.idle_watts)

    def set_active_mode(self) -> tuple[bool, str]:
        """アクティブモード: 電力制限をフルに設定"""
        return self.set_power_limit(self.active_watts)

    def get_status(self) -> dict:
        """現在の状態を返す"""
        info = self.get_gpu_info()
        return {
            "available": self.available,
            "power_control_available": self.power_control_available,
            "power_control_message": self.power_control_message,
            "power_socket": self._power_socket,
            "idle_watts": self.idle_watts,
            "active_watts": self.active_watts,
            "gpu_info": info,
        }


def create_all_managers() -> list['GpuPowerManager']:
    """全GPUの電力マネージャーをまとめて生成する"""
    count = _detect_gpu_count()
    if count == 0:
        return []
    return [GpuPowerManager(gpu_id=i) for i in range(count)]


def main():
    """CLI エントリポイント。全GPU情報を表示する。"""
    import json

    managers = create_all_managers()
    if not managers:
        print("nvidia-smi が見つかりません。GPU 省電力制御は利用できません。")
        return

    for mgr in managers:
        print(f"--- GPU {mgr.gpu_id} ---")
        info = mgr.get_gpu_info()
        print(json.dumps(info, ensure_ascii=False, indent=2))
        print()


if __name__ == "__main__":
    main()
