"""Manual daily diary generation command."""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chat.client import OllamaClient
from src.chat.config import ChatConfig
from src.diary.collector import DiaryCollector
from src.diary.service import DailyDiaryService
from src.integrations.google_calendar import GoogleCalendarMCPClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a daily diary markdown file.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Target date: YYYY-MM-DD")
    parser.add_argument("--timezone", default="Asia/Tokyo")
    parser.add_argument("--calendar-id", default="primary")
    parser.add_argument("--no-calendar", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_date = date.fromisoformat(args.date)
    config = ChatConfig.load(PROJECT_ROOT / "config" / "chat_config.json")
    llm = OllamaClient(base_url=config.ollama_base_url, model=config.model)
    calendar_client = None if args.no_calendar else GoogleCalendarMCPClient.from_env()
    collector = DiaryCollector(
        PROJECT_ROOT,
        calendar_client=calendar_client,
        timezone=args.timezone,
    )
    service = DailyDiaryService(
        project_root=PROJECT_ROOT,
        llm=llm,
        collector=collector,
        timezone=args.timezone,
        temperature=0.4,
        num_ctx=config.num_ctx,
    )
    try:
        result = service.generate(
            target_date,
            save=not args.no_save,
            overwrite=args.overwrite,
            include_calendar=not args.no_calendar,
            calendar_id=args.calendar_id,
        )
        print(result.markdown)
        if not args.no_save:
            print(f"\nSaved: {result.markdown_path}", file=sys.stderr)
    finally:
        llm.close()


if __name__ == "__main__":
    main()

