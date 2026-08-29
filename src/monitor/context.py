"""
システムモニターコンテキスト
メトリクス収集 + ストレージ を統合し、LLMプロンプトに注入するコンテキストを生成

Phase 10: マルチGPUメトリクス表示対応

バックグラウンドでメトリクスを定期収集し、SQLiteに蓄積。
会話時にサブPCの現在の状況をシステムプロンプトに追加する。
"""
import time
import threading
from pathlib import Path
from typing import Callable, Optional

from src.monitor.collector import SystemCollector, SystemMetrics
from src.monitor.storage import MetricsStorage
from src.perception.policy import resolve_sensor_policy


class MonitorContext:
    """
    システムモニターコンテキストマネージャー

    - バックグラウンドでメトリクスを定期収集 (psutil)
    - SQLite に蓄積
    - LLMプロンプト用のコンテキストテキストを生成
    """

    def __init__(
        self,
        db_path: str = "data/metrics/system_metrics.db",
        collect_interval: float = 30.0,
        process_details_enabled: Optional[bool] = None,
        *,
        write_attempts: int = 3,
        write_retry_delay: float = 0.25,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ):
        """
        Args:
            db_path: SQLite DBのパス
            collect_interval: 収集間隔 (秒)
            process_details_enabled: プロセス詳細収集の opt-in。None のとき
                SENSOR_PROCESS_DETAILS_ENABLED から解決する (既定オフ / fail closed)。
            write_attempts: 書込失敗時の最大再試行回数 (デフォルト 3)。
                1 未満は 1 に切り上げる。
            write_retry_delay: 再試行間の短い有界バックオフ (秒)。
                0 未満は 0 に切り上げる。
            sleep_fn: バックオフ sleep の注入フック (テスト用)。None のとき time.sleep。
        """
        if process_details_enabled is None:
            process_details_enabled = resolve_sensor_policy().process_details
        self.process_details_enabled = process_details_enabled
        self.write_attempts = max(1, int(write_attempts))
        self.write_retry_delay = max(0.0, float(write_retry_delay))
        self._sleep = sleep_fn or time.sleep
        self._db_error_count = 0
        self._dropped_write_count = 0
        self.collector = SystemCollector(
            interval=collect_interval, process_details_enabled=process_details_enabled
        )
        self.storage = MetricsStorage(
            db_path=db_path, process_details_enabled=process_details_enabled
        )
        self.collect_interval = collect_interval
        self._running = False

    def start(self) -> bool:
        """DB初期化 + バックグラウンド収集を開始

        storage に触れる前に collector の実状態を検査する (受付前検査)。
        - 既に実稼働中 (``collector.is_running``) なら storage を触らず冪等に True
        - 前回 stop が完了せず旧 worker が生存 (``thread_alive`` かつ ``stop_pending``)
          の間は、storage を触らず・置き換えず・閉じずに False (reactivate しない)
        どちらでもないときのみ storage の initialize と collector の start を試みる。

        Returns:
            ストレージ初期化と収集 worker 開始の両方が成功したとき True。
            どちらかが失敗した場合は収集 worker を停止し DB を閉じてから False。
        """
        # 既に実稼働中: storage に触れず冪等に True (単一飛行維持)。
        if self.collector.is_running:
            self._running = True
            return True
        # 旧 worker が生存 (stop_pending): storage を触らず・置き換えず・閉じず False。
        if self.collector.thread_alive and self.collector.stop_pending:
            return False
        try:
            self.storage.initialize()
        except Exception as e:
            self._running = False
            self._cleanup_after_failed_start()
            # DBパス・SQLiteメッセージなどの生情報は出力しない。型のみで診断する。
            print(f"monitor start failed: {type(e).__name__}")
            return False
        if not self.collector.start(callback=self._on_metrics):
            self._running = False
            self._cleanup_after_failed_start()
            print("monitor start failed: collector unavailable")
            return False
        self._running = True
        return True

    def _cleanup_after_failed_start(self) -> None:
        """start() 失敗時の部分後始末。失敗していない側のリソースも冪等に閉じる。

        コレクター worker が生存し続けている間 (stop_pending) は storage を閉じない
        (生存 worker のコールバックが storage を書き続けるため)。worker の死を確認
        できたときのみ storage を閉じる。
        """
        try:
            self.collector.stop()
        except Exception:
            pass
        if not self.collector.thread_alive:
            try:
                self.storage.close()
            except Exception:
                pass

    def stop(self) -> None:
        """収集を停止する。

        collector を先に停止 (join) する。join タイムアウトでコレクター worker が
        生存し続ける間は storage を開いたまま所有する (生存 worker のコールバックが
        storage を書くため、閉じると書込を壊す)。worker の死を確認できたとき、
        または繰り返し stop で死を確認したときにのみ storage を閉じる。
        """
        self._running = False
        self.collector.stop()
        if not self.collector.thread_alive:
            self.storage.close()

    @property
    def is_running(self) -> bool:
        """stop 要求後は即座に False (worker が生存し続けていても)。"""
        return self._running and self.collector.is_running

    def _on_metrics(self, metrics: SystemMetrics) -> None:
        """メトリクス収集時のコールバック — DBに保存 (有限回の再試行つき)

        一時的な書込失敗に対して同じ metrics オブジェクトを write_attempts 回まで
        再試行し、成功したらその時点で止める。試行間には write_retry_delay 秒の
        短い有界バックオフを挟む。全試行失敗 (exhaustion) のときだけ固定の
        dropped-write カウンタを増やし、型のみ・ASCII の診断を出す (パス・
        SQLiteメッセージ・プロセス詳細は出さない)。キュー・バッファは持たず、
        メモリは増加しない。
        """
        last_error: Optional[Exception] = None
        attempt = 0
        while attempt < self.write_attempts:
            try:
                self.storage.store_metrics(metrics)
                return
            except Exception as e:
                attempt += 1
                last_error = e
                if attempt < self.write_attempts:
                    self._sleep(self.write_retry_delay)
        # 全試行失敗: 固定カウンタを増やし、型のみ・ASCII で通知 (パス・プロセス詳細なし)。
        self._dropped_write_count += 1
        self._db_error_count += 1
        if self._db_error_count == 1 or self._db_error_count % 100 == 0:
            print(
                f"metrics db write dropped ({self._db_error_count}th, "
                f"attempts={self.write_attempts}, {type(last_error).__name__})"
            )

    def get_context_text(self) -> str:
        """
        LLMのシステムプロンプトに注入するPCモニターコンテキストを生成

        Returns:
            PC状態の自然言語テキスト。非稼働時は空文字。
        """
        if not self.is_running:
            return ""

        metrics = self.collector.get_latest()
        if metrics is None:
            return ""

        lines = []
        lines.append("\n--- サブPCの現在の状態 ---")

        # CPU
        cpu_status = "低負荷" if metrics.cpu_percent < 30 else "中負荷" if metrics.cpu_percent < 70 else "高負荷"
        lines.append(f"- CPU: {metrics.cpu_percent:.0f}% ({cpu_status})")
        if metrics.cpu_temp_c is not None:
            lines.append(f"  温度: {metrics.cpu_temp_c:.0f}°C")

        # メモリ
        mem_status = "余裕あり" if metrics.mem_percent < 60 else "少し使用中" if metrics.mem_percent < 80 else "逼迫"
        lines.append(f"- メモリ: {metrics.mem_used_gb:.1f}GB / {metrics.mem_total_gb:.1f}GB ({metrics.mem_percent:.0f}%, {mem_status})")

        # GPU (マルチGPU対応 Phase 10)
        if metrics.gpus:
            # 役割ラベル
            gpu_roles = {}
            try:
                from src.service.gpu_config import get_cached_config
                cfg = get_cached_config()
                if cfg.profile == "dual_gpu":
                    gpu_roles[cfg.llm_gpu_index] = "LLM専用"
                    gpu_roles[cfg.inference_gpu_index] = "推論専用"
            except Exception:
                pass

            for gpu in metrics.gpus:
                idx = gpu["index"]
                name = gpu["name"]
                role = gpu_roles.get(idx, "")
                role_str = f" [{role}]" if role else ""
                gpu_status = "アイドル" if gpu["util_percent"] < 10 else "稼働中" if gpu["util_percent"] < 80 else "フル稼働"
                lines.append(f"- GPU{idx} ({name}){role_str}: {gpu['util_percent']:.0f}% ({gpu_status})")
                lines.append(f"  VRAM: {gpu['mem_used_mb']:.0f}MB / {gpu['mem_total_mb']:.0f}MB")
                lines.append(f"  温度: {gpu['temp_c']:.0f}°C")
        elif metrics.gpu_util_percent is not None:
            # 後方互換: 単一GPU表示
            gpu_status = "アイドル" if metrics.gpu_util_percent < 10 else "稼働中" if metrics.gpu_util_percent < 80 else "フル稼働"
            lines.append(f"- GPU: {metrics.gpu_util_percent:.0f}% ({gpu_status})")
            if metrics.gpu_mem_used_mb is not None and metrics.gpu_mem_total_mb:
                lines.append(f"  VRAM: {metrics.gpu_mem_used_mb:.0f}MB / {metrics.gpu_mem_total_mb:.0f}MB")
            if metrics.gpu_temp_c is not None:
                lines.append(f"  温度: {metrics.gpu_temp_c:.0f}°C")

        # ディスク
        if metrics.disk_percent > 85:
            lines.append(f"- ディスク: {metrics.disk_percent:.0f}% 使用 ⚠️ 残り少ない")
        elif metrics.disk_percent > 70:
            lines.append(f"- ディスク: {metrics.disk_percent:.0f}% 使用")

        # 負荷異常の警告
        warnings = []
        if metrics.cpu_percent > 90:
            warnings.append("CPU使用率が非常に高い")
        if metrics.mem_percent > 90:
            warnings.append("メモリが逼迫しています")
        if metrics.cpu_temp_c and metrics.cpu_temp_c > 85:
            warnings.append("CPU温度が高い")
        # マルチGPU温度チェック
        for gpu in metrics.gpus:
            if gpu.get("temp_c", 0) > 85:
                warnings.append(f"GPU{gpu['index']}({gpu['name']})の温度が高い({gpu['temp_c']:.0f}°C)")
        if not metrics.gpus and metrics.gpu_temp_c and metrics.gpu_temp_c > 85:
            warnings.append("GPU温度が高い")

        if warnings:
            lines.append(f"- ⚠️ 注意: {', '.join(warnings)}")

        return "\n".join(lines)

    def get_status(self) -> dict:
        """APIレスポンス用の状態辞書

        状態メタデータは固定キー (bool / 安全なカウンタ) のみ。例外内容・パス・
        プロセス詳細などの可変情報は含めない。外部公開は Web 側の allowlist で
        さらに絞られる。
        """
        metrics = self.collector.get_latest()
        result = {
            "running": self.is_running,
            "thread_alive": self.collector.thread_alive,
            "stop_pending": self.collector.stop_pending,
            "collect_interval": self.collect_interval,
            "write_attempts": self.write_attempts,
            "dropped_writes": self._dropped_write_count,
            "record_count": 0,
        }
        try:
            result["record_count"] = self.storage.get_record_count()
        except Exception:
            pass

        if metrics:
            result.update({
                "cpu_percent": metrics.cpu_percent,
                "mem_percent": metrics.mem_percent,
                "mem_used_gb": metrics.mem_used_gb,
                "gpu_util_percent": metrics.gpu_util_percent,
                "gpu_mem_used_mb": metrics.gpu_mem_used_mb,
                "gpu_temp_c": metrics.gpu_temp_c,
                "cpu_temp_c": metrics.cpu_temp_c,
                "disk_percent": metrics.disk_percent,
                "process_count": metrics.process_count,
                "last_collected": metrics.timestamp,
            })
        return result

    def get_recent_summary(self, minutes: int = 60) -> dict:
        """直近N分のサマリーを返す"""
        rows = self.storage.get_recent(minutes=minutes)
        if not rows:
            return {"period_minutes": minutes, "sample_count": 0}

        cpu_vals = [r["cpu_percent"] for r in rows if r["cpu_percent"] is not None]
        mem_vals = [r["mem_percent"] for r in rows if r["mem_percent"] is not None]
        gpu_vals = [r["gpu_util_percent"] for r in rows if r["gpu_util_percent"] is not None]

        summary = {
            "period_minutes": minutes,
            "sample_count": len(rows),
        }

        if cpu_vals:
            summary["cpu_avg"] = round(sum(cpu_vals) / len(cpu_vals), 1)
            summary["cpu_max"] = round(max(cpu_vals), 1)
        if mem_vals:
            summary["mem_avg"] = round(sum(mem_vals) / len(mem_vals), 1)
            summary["mem_max"] = round(max(mem_vals), 1)
        if gpu_vals:
            summary["gpu_avg"] = round(sum(gpu_vals) / len(gpu_vals), 1)
            summary["gpu_max"] = round(max(gpu_vals), 1)

        return summary
