"""決定論的 CalendarSource。

UserProfile のスケジュールから、次の予定 (next_event_at / next_event_title) を
Policy が使える形で計算する。時刻の解釈はローカルのみ、best-effort で
クラッシュしない。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NextEvent:
    """次の予定。日時が確定できない場合は None。"""

    start_at: float | None
    title: str | None


class CalendarSource:
    """UserProfile のスケジュールから次の予定を決定的に求める。"""

    def __init__(self, profile) -> None:
        self._profile = profile

    def next_event(self, now: float | None = None) -> NextEvent:
        if now is None:
            now = time.time()

        try:
            schedule = self._profile.get_upcoming_schedule()
        except Exception:
            logger.exception("CalendarSource: get_upcoming_schedule failed")
            return NextEvent(None, None)

        best: tuple[float, str] | None = None
        for entry in schedule:
            try:
                start_at = self._entry_start_at(entry)
            except Exception:
                logger.exception("CalendarSource: entry conversion failed")
                continue
            if start_at is None or start_at < now:
                continue
            title = entry.get("title")
            if best is None or start_at < best[0]:
                best = (start_at, title)

        if best is None:
            return NextEvent(None, None)
        return NextEvent(start_at=best[0], title=best[1])

    def _entry_start_at(self, entry: dict) -> float | None:
        date_str = entry.get("date")
        time_str = entry.get("time")
        if not date_str:
            return None
        if time_str:
            dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        else:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.timestamp()
