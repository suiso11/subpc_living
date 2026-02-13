"""
GPU 省電力制御モジュール
nvidia-smi を使って GPU の電力制限を動的に変更する。
常時稼働時のアイドル消費電力を抑え、推論時にフルパワーへ復帰させる。
Phase 9: GPU名を検出して適切なデフォルト値を自動設定。
"""
import subprocess
import shutil
from typing import Optional


# GPU別の電力プリセット (idle_watts, active_watts)
GPU_POWER_PRESETS: dict[str, tuple[int, int]] = {
    "P40":       (100, 250),   # TDP 250W
    "P100":      (100, 250),   # TDP 250W
    "V100":      (100, 300),   # TDP 300W
    "GTX 1060":  (80, 120),    # TDP 120W
    "GTX 1070":  (80, 150),    # TDP 150W
    "GTX 1080":  (80, 180),    # TDP 180W
    "RTX 2080":  (80, 215),    # TDP 215W
    "RTX 3080":  (100, 320),   # TDP 320W
    "RTX 4090":  (100, 450),   # TDP 450W
}


def _detect_gpu_preset() -> tuple[int, int]:
    """nvidia-smi からGPU名を取得し、適切なプリセットを返す"""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return (100, 250)  # デフォルト

    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=gpu_name", "--format=csv,noheader"],
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


class GpuPowerManager:
    """nvidia-smi を使った GPU 電力制限管理

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
        default_idle, default_active = _detect_gpu_preset()
        self.idle_watts = idle_watts if idle_watts is not None else default_idle
        self.active_watts = active_watts if active_watts is not None else default_active
        self.gpu_id = gpu_id
        self._nvidia_smi = shutil.which("nvidia-smi")

    @property
    def available(self) -> bool:
        """nvidia-smi が利用可能か"""
        return self._nvidia_smi is not None

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

        ok, output = self._run_smi([
            f"--id={self.gpu_id}",
            f"--power-limit={watts}",
        ])

        if ok:
            return True, f"GPU power limit set to {watts}W"
        return False, output

    def set_persistence_mode(self, enable: bool = True) -> tuple[bool, str]:
        """GPU の Persistence Mode を設定する (root/sudo 権限が必要)

        Persistence Mode を有効にすると GPU ドライバが常駐し、
        プロセス起動時のロード時間を短縮できる。
        """
        if not self.available:
            return False, "nvidia-smi not found"

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
            "idle_watts": self.idle_watts,
            "active_watts": self.active_watts,
            "gpu_info": info,
        }


def main():
    """CLI エントリポイント。GPU 情報を表示する。"""
    import json

    mgr = GpuPowerManager()

    if not mgr.available:
        print("nvidia-smi が見つかりません。GPU 省電力制御は利用できません。")
        return

    info = mgr.get_gpu_info()
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
