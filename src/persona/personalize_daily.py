"""Manual daily personalization command."""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chat.config import ChatConfig
from src.llm.providers.ollama import OllamaProvider
from src.persona.daily_personalizer import DailyPersonalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine user_profile.json from a saved diary.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Target date: YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="Write audit JSON but do not update profile")
    parser.add_argument("--min-confidence", type=float, default=0.72)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_date = date.fromisoformat(args.date)
    config = ChatConfig.load(PROJECT_ROOT / "config" / "chat_config.json")
    llm = OllamaProvider(base_url=config.ollama_base_url, model=config.model)
    personalizer = DailyPersonalizer(
        project_root=PROJECT_ROOT,
        llm=llm,
        min_confidence=args.min_confidence,
        num_ctx=config.num_ctx,
    )
    try:
        result = personalizer.run(target_date, dry_run=args.dry_run)
        print(f"target_date: {result.target_date}")
        print(f"dry_run: {result.dry_run}")
        print(f"applied_count: {result.applied_count}")
        print(f"audit_path: {result.audit_path}")
        if result.skipped:
            print(f"skipped: {len(result.skipped)}")
    finally:
        llm.close()


if __name__ == "__main__":
    main()

