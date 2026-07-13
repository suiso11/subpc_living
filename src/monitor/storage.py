"""
システムメトリクスの永続化ストレージ
SQLite で時系列データを蓄積
"""
import sqlite3
import json
import time
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from src.monitor.collector import SystemMetrics


class MetricsStorage:
    """
    SQLite ベースのメトリクスストレージ

    テーブル:
      - metrics: 時系列のシステムメトリクス
      - hourly_summary: 1時間ごとの集計サマリー
    """

    def __init__(self, db_path: str = "data/metrics/system_metrics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        """DB接続を開いてテーブルを作成"""
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")  # 並行アクセス対応
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()

    def close(self) -> None:
        """DB接続を閉じる"""
        if self._conn:
            self._conn.close()
            self._conn = None

    @contextmanager
    def _cursor(self):
        """スレッドセーフなカーソルのコンテキストマネージャー"""
        if self._conn is None:
            raise RuntimeError("DB未初期化。initialize() を先に呼んでください。")
        cursor = self._conn.cursor()
        try:
            yield cursor
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    def _create_tables(self) -> None:
        """テーブル作成"""
        with self._cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    cpu_percent REAL,
                    cpu_freq_mhz REAL,
                    load_avg_1m REAL,
                    mem_percent REAL,
                    mem_used_gb REAL,
                    swap_percent REAL,
                    disk_percent REAL,
                    disk_read_mb_s REAL,
                    disk_write_mb_s REAL,
                    net_sent_mb_s REAL,
                    net_recv_mb_s REAL,
                    cpu_temp_c REAL,
                    gpu_util_percent REAL,
                    gpu_mem_used_mb REAL,
                    gpu_temp_c REAL,
                    gpu_power_w REAL,
                    process_count INTEGER,
                    top_processes TEXT,
                    extra_json TEXT
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp
                ON metrics (timestamp)
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS hourly_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hour_start REAL NOT NULL,
                    hour_label TEXT NOT NULL,
                    sample_count INTEGER,
                    cpu_avg REAL,
                    cpu_max REAL,
                    mem_avg REAL,
                    mem_max REAL,
                    gpu_util_avg REAL,
                    gpu_mem_avg REAL,
                    disk_percent REAL,
                    summary_text TEXT
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_hourly_hour_start
                ON hourly_summary (hour_start)
            """)

    def store_metrics(self, m: SystemMetrics) -> None:
        """メトリクスを1レコード保存"""
        top_procs_json = json.dumps(m.top_cpu_processes, ensure_ascii=False)
        extra = json.dumps({
            "cpu_per_core": m.cpu_per_core,
            "mem_total_gb": m.mem_total_gb,
            "disk_total_gb": m.disk_total_gb,
            "disk_used_gb": m.disk_used_gb,
            "gpu_mem_total_mb": m.gpu_mem_total_mb,
        }, ensure_ascii=False)

        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO metrics (
                    timestamp, cpu_percent, cpu_freq_mhz, load_avg_1m,
                    mem_percent, mem_used_gb, swap_percent,
                    disk_percent, disk_read_mb_s, disk_write_mb_s,
                    net_sent_mb_s, net_recv_mb_s,
                    cpu_temp_c, gpu_util_percent, gpu_mem_used_mb,
                    gpu_temp_c, gpu_power_w,
                    process_count, top_processes, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                m.timestamp, m.cpu_percent, m.cpu_freq_mhz, m.load_avg_1m,
                m.mem_percent, m.mem_used_gb, m.swap_percent,
                m.disk_percent, m.disk_read_mb_s, m.disk_write_mb_s,
                m.net_sent_mb_s, m.net_recv_mb_s,
                m.cpu_temp_c, m.gpu_util_percent, m.gpu_mem_used_mb,
                m.gpu_temp_c, m.gpu_power_w,
                m.process_count, top_procs_json, extra,
            ))

    def get_recent(self, minutes: int = 60) -> list[dict]:
        """直近N分のメトリクスを取得"""
        cutoff = time.time() - (minutes * 60)
        with self._cursor() as cur:
            cur.execute("""
                SELECT timestamp, cpu_percent, mem_percent, mem_used_gb,
                       gpu_util_percent, gpu_mem_used_mb, gpu_temp_c,
                       cpu_temp_c, disk_percent, process_count,
                       load_avg_1m, net_sent_mb_s, net_recv_mb_s
                FROM metrics
                WHERE timestamp > ?
                ORDER BY timestamp ASC
            """, (cutoff,))
            rows = cur.fetchall()

        return [
            {
                "timestamp": r[0], "cpu_percent": r[1],
                "mem_percent": r[2], "mem_used_gb": r[3],
                "gpu_util_percent": r[4], "gpu_mem_used_mb": r[5],
                "gpu_temp_c": r[6], "cpu_temp_c": r[7],
                "disk_percent": r[8], "process_count": r[9],
                "load_avg_1m": r[10], "net_sent_mb_s": r[11],
                "net_recv_mb_s": r[12],
            }
            for r in rows
        ]

    def get_latest_row(self) -> Optional[dict]:
        """最新の1レコードを取得"""
        with self._cursor() as cur:
            cur.execute("""
                SELECT timestamp, cpu_percent, cpu_freq_mhz, load_avg_1m,
                       mem_percent, mem_used_gb, swap_percent,
                       disk_percent, gpu_util_percent, gpu_mem_used_mb,
                       gpu_temp_c, gpu_power_w, cpu_temp_c,
                       process_count, top_processes
                FROM metrics
                ORDER BY timestamp DESC LIMIT 1
            """)
            row = cur.fetchone()

        if not row:
            return None

        return {
            "timestamp": row[0], "cpu_percent": row[1],
            "cpu_freq_mhz": row[2], "load_avg_1m": row[3],
            "mem_percent": row[4], "mem_used_gb": row[5],
            "swap_percent": row[6], "disk_percent": row[7],
            "gpu_util_percent": row[8], "gpu_mem_used_mb": row[9],
            "gpu_temp_c": row[10], "gpu_power_w": row[11],
            "cpu_temp_c": row[12], "process_count": row[13],
            "top_processes": json.loads(row[14]) if row[14] else [],
        }

    def compute_hourly_summary(self, hour_start: float) -> Optional[dict]:
        """
        指定した hour_start (Unix timestamp) から1時間分の集計を計算

        Returns:
            集計結果の辞書。データなしの場合 None。
        """
        hour_end = hour_start + 3600
        with self._cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as cnt,
                    AVG(cpu_percent) as cpu_avg,
                    MAX(cpu_percent) as cpu_max,
                    AVG(mem_percent) as mem_avg,
                    MAX(mem_percent) as mem_max,
                    AVG(gpu_util_percent) as gpu_avg,
                    AVG(gpu_mem_used_mb) as gpu_mem_avg,
                    AVG(disk_percent) as disk_avg
                FROM metrics
                WHERE timestamp >= ? AND timestamp < ?
            """, (hour_start, hour_end))
            row = cur.fetchone()

        if not row or row[0] == 0:
            return None

        import datetime
        hour_label = datetime.datetime.fromtimestamp(hour_start).strftime("%Y-%m-%d %H:00")

        return {
            "hour_start": hour_start,
            "hour_label": hour_label,
            "sample_count": row[0],
            "cpu_avg": round(row[1] or 0, 1),
            "cpu_max": round(row[2] or 0, 1),
            "mem_avg": round(row[3] or 0, 1),
            "mem_max": round(row[4] or 0, 1),
            "gpu_util_avg": round(row[5] or 0, 1) if row[5] is not None else None,
            "gpu_mem_avg": round(row[6] or 0, 1) if row[6] is not None else None,
            "disk_percent": round(row[7] or 0, 1),
        }

    def store_hourly_summary(self, summary: dict) -> None:
        """時間別サマリーを保存"""
        with self._cursor() as cur:
            cur.execute("""
                INSERT OR REPLACE INTO hourly_summary (
                    hour_start, hour_label, sample_count,
                    cpu_avg, cpu_max, mem_avg, mem_max,
                    gpu_util_avg, gpu_mem_avg, disk_percent,
                    summary_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                summary["hour_start"], summary["hour_label"],
                summary["sample_count"],
                summary["cpu_avg"], summary["cpu_max"],
                summary["mem_avg"], summary["mem_max"],
                summary.get("gpu_util_avg"), summary.get("gpu_mem_avg"),
                summary["disk_percent"],
                summary.get("summary_text", ""),
            ))

    def get_daily_summaries(self, days: int = 1) -> list[dict]:
        """直近N日分の時間別サマリーを取得"""
        cutoff = time.time() - (days * 86400)
        with self._cursor() as cur:
            cur.execute("""
                SELECT hour_start, hour_label, sample_count,
                       cpu_avg, cpu_max, mem_avg, mem_max,
                       gpu_util_avg, gpu_mem_avg, disk_percent,
                       summary_text
                FROM hourly_summary
                WHERE hour_start > ?
                ORDER BY hour_start ASC
            """, (cutoff,))
            rows = cur.fetchall()

        return [
            {
                "hour_start": r[0], "hour_label": r[1], "sample_count": r[2],
                "cpu_avg": r[3], "cpu_max": r[4],
                "mem_avg": r[5], "mem_max": r[6],
                "gpu_util_avg": r[7], "gpu_mem_avg": r[8],
                "disk_percent": r[9], "summary_text": r[10],
            }
            for r in rows
        ]

    def get_record_count(self) -> int:
        """メトリクスの総レコード数"""
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM metrics")
            return cur.fetchone()[0]

    def cleanup_old(self, keep_days: int = 30) -> int:
        """古いレコードを削除 (keep_days 日より古いものを削除)"""
        cutoff = time.time() - (keep_days * 86400)
        with self._cursor() as cur:
            cur.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff,))
            deleted = cur.rowcount
        return deleted
