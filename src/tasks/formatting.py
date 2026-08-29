from __future__ import annotations

from datetime import datetime


def format_short_due(local: datetime, *, with_time: bool = False) -> str:
    """M/D (ゼロ埋めなし) を返す。with_time=True ならゼロ埋め HH:MM を付ける。

    strftime("%-m/%-d") は Windows 非対応のため datetime 属性で組み立てる。
    """
    if with_time:
        return f"{local.month}/{local.day} {local.hour:02d}:{local.minute:02d}"
    return f"{local.month}/{local.day}"