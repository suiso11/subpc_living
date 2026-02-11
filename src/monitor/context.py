"""
システムモニターコンテキスト
メトリクス収集 + ストレージ を統合し、LLMプロンプトに注入するコンテキストを生成

バックグラウンドでメトリクスを定期収集し、SQLiteに蓄積。
会話時にサブPCの現在の状況をシステムプロンプトに追加する。
"""
import time
import threading
from pathlib import Path
from typing import Optional

from src.monitor.collector import SystemCollector, SystemMetrics
from src.monitor.storage import MetricsStorage


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
    ):
        """
        Args:
            db_path: SQLite DBのパス
            collect_interval: 収集間隔 (秒)
        """
        self.collector = SystemCollector(interval=collect_interval)
        self.storage = MetricsStorage(db_path=db_path)
        self.collect_interval = collect_interval
        self._running = False

    def start(self) -> bool:
        """DB初期化 + バックグラウンド収集を開始"""
        try:
            self.storage.initialize()
            self.collector.start(callback=self._on_metrics)
            self._running = True
            return True
        except Exception as e:
            print(f"⚠️  MonitorContext 起動失敗: {e}")
            return False

    def stop(self) -> None:
        """収集を停止しDBを閉じる"""
        self._running = False
        self.collector.stop()
        self.storage.close()

    @property
    def is_running(self) -> bool:
        return self._running and self.collector.is_running

    def _on_metrics(self, metrics: SystemMetrics) -> None:
        """メトリクス収集時のコールバック — DBに保存"""
        try:
            self.storage.store_metrics(metrics)
        except Exception:
            pass  # DB書込失敗は無視（ログ欠損は許容）

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

        # GPU
        if metrics.gpu_util_percent is not None:
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
        if metrics.gpu_temp_c and metrics.gpu_temp_c > 85:
            warnings.append("GPU温度が高い")
        if metrics.cpu_temp_c and metrics.cpu_temp_c > 85:
            warnings.append("CPU温度が高い")

        if warnings:
            lines.append(f"- ⚠️ 注意: {', '.join(warnings)}")

        return "\n".join(lines)

    def get_status(self) -> dict:
        """APIレスポンス用の状態辞書"""
        metrics = self.collector.get_latest()
        result = {
            "running": self.is_running,
            "collect_interval": self.collect_interval,
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
