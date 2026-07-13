"""
システムメトリクス収集
psutil を使用して CPU / メモリ / ディスク / ネットワーク / GPU 等のメトリクスを収集
Phase 10: マルチGPUメトリクス収集対応 (P40 + RTX 2070 Super)
"""
import time
import threading
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class SystemMetrics:
    """1回分のシステムメトリクススナップショット"""
    timestamp: float = 0.0

    # CPU
    cpu_percent: float = 0.0             # 全体 CPU 使用率 (%)
    cpu_per_core: list[float] = field(default_factory=list)  # コアごとの使用率
    cpu_freq_mhz: float = 0.0           # 現在のクロック (MHz)
    load_avg_1m: float = 0.0            # ロードアベレージ 1分

    # メモリ
    mem_total_gb: float = 0.0
    mem_used_gb: float = 0.0
    mem_percent: float = 0.0
    swap_percent: float = 0.0

    # ディスク
    disk_total_gb: float = 0.0
    disk_used_gb: float = 0.0
    disk_percent: float = 0.0
    disk_read_mb_s: float = 0.0
    disk_write_mb_s: float = 0.0

    # ネットワーク
    net_sent_mb_s: float = 0.0
    net_recv_mb_s: float = 0.0

    # 温度 (利用可能な場合)
    cpu_temp_c: Optional[float] = None
    gpu_temp_c: Optional[float] = None

    # GPU (nvidia-smi 経由、利用可能な場合)
    gpu_util_percent: Optional[float] = None
    gpu_mem_used_mb: Optional[float] = None
    gpu_mem_total_mb: Optional[float] = None
    gpu_power_w: Optional[float] = None

    # マルチGPUメトリクス (Phase 10)
    gpus: list[dict] = field(default_factory=list)  # [{index, name, util, mem_used, mem_total, temp, power}]

    # プロセス
    process_count: int = 0
    top_cpu_processes: list[dict] = field(default_factory=list)  # [{name, pid, cpu_percent}]

    def to_dict(self) -> dict:
        return asdict(self)


class SystemCollector:
    """
    システムメトリクスを定期的に収集するコレクター

    バックグラウンドスレッドで動作し、指定間隔でメトリクスを収集。
    """

    def __init__(self, interval: float = 30.0):
        """
        Args:
            interval: 収集間隔 (秒)。デフォルト30秒。
        """
        if not HAS_PSUTIL:
            raise RuntimeError("psutil がインストールされていません: pip install psutil")

        self.interval = interval
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._latest: Optional[SystemMetrics] = None
        self._lock = threading.Lock()
        self._callback = None  # メトリクス収集時のコールバック

        # ディスク I/O の差分計算用
        self._prev_disk_io = None
        self._prev_net_io = None
        self._prev_time = None

    def start(self, callback=None) -> None:
        """
        バックグラウンド収集を開始

        Args:
            callback: 収集のたびに呼ばれる関数 callback(metrics: SystemMetrics)
        """
        self._callback = callback
        self._running = True

        # 初回の I/O カウンタを取得
        try:
            self._prev_disk_io = psutil.disk_io_counters()
            self._prev_net_io = psutil.net_io_counters()
            self._prev_time = time.time()
        except Exception:
            pass

        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """収集を停止"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    def get_latest(self) -> Optional[SystemMetrics]:
        """最新のメトリクスを取得"""
        with self._lock:
            return self._latest

    def collect_once(self) -> SystemMetrics:
        """メトリクスを1回収集して返す"""
        metrics = SystemMetrics(timestamp=time.time())

        # --- CPU ---
        try:
            metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics.cpu_per_core = psutil.cpu_percent(percpu=True)
            freq = psutil.cpu_freq()
            if freq:
                metrics.cpu_freq_mhz = freq.current
            load = psutil.getloadavg()
            metrics.load_avg_1m = load[0]
        except Exception:
            pass

        # --- メモリ ---
        try:
            vm = psutil.virtual_memory()
            metrics.mem_total_gb = round(vm.total / (1024 ** 3), 2)
            metrics.mem_used_gb = round(vm.used / (1024 ** 3), 2)
            metrics.mem_percent = vm.percent
            swap = psutil.swap_memory()
            metrics.swap_percent = swap.percent
        except Exception:
            pass

        # --- ディスク ---
        try:
            disk = psutil.disk_usage("/")
            metrics.disk_total_gb = round(disk.total / (1024 ** 3), 2)
            metrics.disk_used_gb = round(disk.used / (1024 ** 3), 2)
            metrics.disk_percent = disk.percent

            # I/O レート計算
            now = time.time()
            cur_disk = psutil.disk_io_counters()
            if self._prev_disk_io and self._prev_time:
                dt = now - self._prev_time
                if dt > 0:
                    metrics.disk_read_mb_s = round(
                        (cur_disk.read_bytes - self._prev_disk_io.read_bytes) / dt / (1024 ** 2), 2
                    )
                    metrics.disk_write_mb_s = round(
                        (cur_disk.write_bytes - self._prev_disk_io.write_bytes) / dt / (1024 ** 2), 2
                    )
            self._prev_disk_io = cur_disk
        except Exception:
            pass

        # --- ネットワーク ---
        try:
            now = time.time()
            cur_net = psutil.net_io_counters()
            if self._prev_net_io and self._prev_time:
                dt = now - self._prev_time
                if dt > 0:
                    metrics.net_sent_mb_s = round(
                        (cur_net.bytes_sent - self._prev_net_io.bytes_sent) / dt / (1024 ** 2), 2
                    )
                    metrics.net_recv_mb_s = round(
                        (cur_net.bytes_recv - self._prev_net_io.bytes_recv) / dt / (1024 ** 2), 2
                    )
            self._prev_net_io = cur_net
            self._prev_time = now
        except Exception:
            pass

        # --- 温度 ---
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # CPU温度: coretemp or k10temp
                for key in ("coretemp", "k10temp", "acpitz", "cpu_thermal"):
                    if key in temps and temps[key]:
                        metrics.cpu_temp_c = temps[key][0].current
                        break
        except Exception:
            pass

        # --- GPU (nvidia-smi) ---
        metrics = self._collect_gpu(metrics)

        # --- プロセス ---
        try:
            metrics.process_count = len(psutil.pids())
            procs = []
            for proc in psutil.process_iter(["pid", "name", "cpu_percent"]):
                try:
                    info = proc.info
                    if info["cpu_percent"] and info["cpu_percent"] > 0:
                        procs.append({
                            "name": info["name"],
                            "pid": info["pid"],
                            "cpu_percent": round(info["cpu_percent"], 1),
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            # CPU使用率トップ5
            procs.sort(key=lambda x: x["cpu_percent"], reverse=True)
            metrics.top_cpu_processes = procs[:5]
        except Exception:
            pass

        return metrics

    def _collect_gpu(self, metrics: SystemMetrics) -> SystemMetrics:
        """全GPUの情報をnvidia-smiで収集 (Phase 10: マルチGPU対応)"""
        try:
            import subprocess
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                gpu_list = []
                for line in result.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 7:
                        gpu_data = {
                            "index": int(parts[0]),
                            "name": parts[1],
                            "util_percent": float(parts[2]),
                            "mem_used_mb": float(parts[3]),
                            "mem_total_mb": float(parts[4]),
                            "temp_c": float(parts[6]),
                        }
                        try:
                            gpu_data["power_w"] = float(parts[5])
                        except ValueError:
                            gpu_data["power_w"] = None
                        gpu_list.append(gpu_data)

                metrics.gpus = gpu_list

                # 後方互換: 最初のGPUデータを単一フィールドにもセット
                if gpu_list:
                    g0 = gpu_list[0]
                    metrics.gpu_util_percent = g0["util_percent"]
                    metrics.gpu_mem_used_mb = g0["mem_used_mb"]
                    metrics.gpu_mem_total_mb = g0["mem_total_mb"]
                    metrics.gpu_power_w = g0.get("power_w")
                    metrics.gpu_temp_c = g0["temp_c"]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # nvidia-smi なし
        except Exception:
            pass
        return metrics

    def _collect_loop(self) -> None:
        """バックグラウンド収集ループ"""
        # 起動直後に1回収集（psutil のcpu_percentは初回呼び出しが不正確なため）
        psutil.cpu_percent(interval=0.1)
        time.sleep(0.5)

        while self._running:
            try:
                metrics = self.collect_once()
                with self._lock:
                    self._latest = metrics

                if self._callback:
                    try:
                        self._callback(metrics)
                    except Exception:
                        pass

            except Exception:
                pass

            time.sleep(self.interval)
