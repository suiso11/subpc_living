"""
システムメトリクス収集
psutil を使用して CPU / メモリ / ディスク / ネットワーク / GPU 等のメトリクスを収集
Phase 10: マルチGPUメトリクス収集対応 (P40 + RTX 2070 Super)
"""
import time
import threading
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

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

    def __init__(
        self,
        interval: float = 30.0,
        process_details_enabled: bool = False,
        *,
        join_timeout: float = 5.0,
        thread_factory: Optional[Callable[[], threading.Thread]] = None,
        collect_fn: Optional[Callable[[], SystemMetrics]] = None,
    ):
        """
        Args:
            interval: 収集間隔 (秒)。デフォルト30秒。
            process_details_enabled: プロセス名・PID・トップリスト収集の opt-in。
                既定オフ (fail closed)。SENSOR_PROCESS_DETAILS_ENABLED=true で有効化。
            join_timeout: stop() がスレッド終了を待つ最大秒数。
            thread_factory: 収集スレッド生成用の注入フック (テスト用)。
                None のとき既定の threading.Thread を使う。start() ごとに1回呼ばれる。
            collect_fn: 1回分の収集処理の注入フック (テスト用)。
                None のとき self.collect_once を使う。
        """
        if not HAS_PSUTIL:
            raise RuntimeError("psutil がインストールされていません: pip install psutil")

        self.interval = interval
        self._process_details_enabled = process_details_enabled
        self._join_timeout = join_timeout
        self._thread_factory = thread_factory
        self._collect_fn = collect_fn or self.collect_once
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_pending = False
        self._latest: Optional[SystemMetrics] = None
        self._lock = threading.Lock()
        self._callback = None  # メトリクス収集時のコールバック

        # ディスク I/O の差分計算用
        self._prev_disk_io = None
        self._prev_net_io = None
        self._prev_time = None

    def start(self, callback=None) -> bool:
        """
        バックグラウンド収集を開始

        戻り値は真偽のみ (生例外は外部へ露出しない)。
        - True: ライブな収集 worker が存在する場合。今回開始した場合、既に稼働中、
          あるいは前回 stop で停止しきらず生存している旧スレッドの所有権が
          残っている場合を含む。また thread.start() が例外を投げてもスレッドが
          生存していれば保持して True とする。
        - False: ライブな worker が残っていない場合のみ。thread_factory の失敗、
          起動失敗で死亡・未起動、起動直後に即死した場合。さらに、stop 要求
          (stop_pending) 中に旧 worker が生存し続けている場合は、その worker を
          reactivate せず _running を変更しないまま False を返す。

        不変条件: start() が False を返すのは、ライブな worker が残っていない場合
        だけである。呼び出し側は False を見たらコレクターを破棄してよい。
        (stop_pending 中の旧 worker は「停止中」であってライブ worker ではない)

        Args:
            callback: 収集のたびに呼ばれる関数 callback(metrics: SystemMetrics)

        Returns:
            ライブな worker が存在するとき True、それ以外は False。
        """
        with self._lock:
            if self._running:
                return True

            # 前回の stop タイムアウト等で生存スレッドの所有権が残っている場合。
            if self._thread is not None:
                try:
                    alive = self._thread.is_alive()
                except Exception:
                    alive = False
                if alive:
                    if self._stop_pending:
                        # stop 保留中の旧 worker は reactivate しない。
                        # _running を変更せず False を返す (停止を継続させる)。
                        return False
                    self._running = True
                    self._callback = callback
                    return True

            self._stop_pending = False
            self._running = True

            # 初回の I/O カウンタを取得
            try:
                self._prev_disk_io = psutil.disk_io_counters()
                self._prev_net_io = psutil.net_io_counters()
                self._prev_time = time.time()
            except Exception:
                pass

            try:
                thread = (
                    self._thread_factory()
                    if self._thread_factory is not None
                    else threading.Thread(target=self._collect_loop, daemon=True)
                )
            except Exception:
                self._running = False
                self._callback = None
                self._thread = None
                self._stop_pending = False
                return False

            try:
                thread.start()
            except Exception:
                # start() が例外を投げても、スレッドが生存していれば所有権を保持し
                # ライブ worker として扱う。死亡・未起動なら破棄して False。
                try:
                    alive = thread.is_alive()
                except Exception:
                    alive = False
                if alive:
                    self._thread = thread
                    self._running = True
                    self._stop_pending = False
                    self._callback = callback
                    return True
                self._running = False
                self._callback = None
                self._thread = None
                self._stop_pending = False
                return False

            # 起動直後に即死していないか確認。即死ならライブ worker は残らない。
            try:
                alive = thread.is_alive()
            except Exception:
                alive = True
            if not alive:
                self._running = False
                self._callback = None
                self._thread = None
                self._stop_pending = False
                return False

            self._thread = thread
            self._callback = callback
            return True

    def stop(self) -> None:
        """収集を停止 (冪等)

        スレッド終了は join_timeout 秒まで待つ。タイムアウトで生存している場合は
        参照を保持して回収しない (stop_pending を立て、後続の stop() で再度 join
        できる)。死亡を確認できたときだけ参照を破棄し、stop_pending を解除する。
        join / is_alive の失敗でも生例外は露出しない。
        """
        with self._lock:
            self._running = False
            thread = self._thread

        if thread is None:
            return

        try:
            thread.join(timeout=self._join_timeout)
        except Exception:
            with self._lock:
                if self._thread is thread:
                    self._stop_pending = True
            return

        with self._lock:
            try:
                alive = thread.is_alive()
            except Exception:
                alive = True
            if self._thread is thread and not alive:
                self._thread = None
                self._stop_pending = False
            elif self._thread is thread:
                self._stop_pending = True

    @property
    def is_running(self) -> bool:
        """実際に稼働中 (収集スレッドが生存 かつ 停止要求なし) のときだけ True。

        予期せぬスレッド死亡や停止後 (join タイムアウト / stop_pending) の状態を
        「稼働中」として報告しない。停止保留中に保持される生存スレッドの
        所有権は _thread に残り、再起動をブロックし続ける。
        """
        if not self._running or self._stop_pending:
            return False
        thread = self._thread
        if thread is None:
            return False
        try:
            return thread.is_alive()
        except Exception:
            return False

    @property
    def thread_alive(self) -> bool:
        """所有する収集スレッドが実際に生存しているか (生の生存判定)。

        stop 要求 (stop_pending) の有無に関係なく報告する。
        """
        thread = self._thread
        if thread is None:
            return False
        try:
            return thread.is_alive()
        except Exception:
            return False

    @property
    def stop_pending(self) -> bool:
        """停止要求済みだが収集スレッドの死亡が未確認か。"""
        return self._stop_pending

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
        # プロセス詳細 (名前・PID・トップリスト) は opt-in (SENSOR_PROCESS_DETAILS_ENABLED)。
        # 既定オフ。オフのときは集計 (プロセス件数) のみを残し、詳細は収集しない。
        try:
            metrics.process_count = len(psutil.pids())
        except Exception:
            pass

        if self._process_details_enabled:
            try:
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
        try:
            psutil.cpu_percent(interval=0.1)
        except Exception:
            pass
        time.sleep(0.5)

        while self._running:
            try:
                metrics = self._collect_fn()
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
