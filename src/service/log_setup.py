"""サービス共通のログ初期化。

- stdout (journald 向け) と logs/<service>.log (ローテーション付き) の両方に出す
- レベルは環境変数 LOG_LEVEL (DEBUG/INFO/WARNING/ERROR、default: INFO)
- ファイルは 5MB x 3世代でローテーション
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def resolve_level(value: str | None = None) -> int:
    name = (value or os.environ.get("LOG_LEVEL") or "INFO").strip().upper()
    return getattr(logging, name, logging.INFO)


def setup_logging(
    service: str,
    *,
    level: str | None = None,
    log_dir: str | Path | None = None,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> logging.Logger:
    """ルートロガーを設定し、サービス名のロガーを返す。

    多重呼び出しは安全 (既存ハンドラを付け替える)。
    """
    root = logging.getLogger()
    root.setLevel(resolve_level(level))

    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    directory = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    try:
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            directory / f"{service}.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as e:
        root.warning("ログファイルを開けません (stdoutのみで続行): %s", e)

    # httpx / uvicorn.access などの多弁なロガーを抑える
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger(service)
