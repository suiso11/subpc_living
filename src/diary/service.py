"""Daily diary generation service."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from src.diary.collector import DiaryCollector, DiarySources

if TYPE_CHECKING:
    from src.chat.client import OllamaClient


DIARY_SYSTEM_PROMPT = """\
あなたは subpc_living の日次ライフログ作成係です。
入力されたローカルログだけを根拠に、日本語で短い日記を作成してください。

厳守:
- 事実と推測を混ぜない。推測は「たぶん」「ログ上は」と明記する。
- 会話ログや予定名を長く引用しない。必要な内容だけ短く要約する。
- ユーザー本人が書いた一人称日記のふりをしない。
- 説教や大げさな励ましを避ける。
- 出力はMarkdown本文だけにする。
"""


@dataclass(frozen=True)
class DailyDiaryResult:
    target_date: str
    markdown: str
    markdown_path: str
    metadata_path: str
    generated: bool
    calendar_error: str = ""


class DailyDiaryService:
    """Collect sources, generate a diary, and persist it under data/diary."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        llm: "OllamaClient",
        collector: DiaryCollector,
        data_dir: str | Path = "data/diary",
        timezone: str = "Asia/Tokyo",
        temperature: float = 0.4,
        num_ctx: int = 8192,
    ):
        self.project_root = Path(project_root)
        self.llm = llm
        self.collector = collector
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_absolute():
            self.data_dir = self.project_root / self.data_dir
        self.timezone = timezone
        self.temperature = temperature
        self.num_ctx = num_ctx

    def diary_path(self, target_date: date) -> Path:
        return self.data_dir / f"{target_date.isoformat()}.md"

    def metadata_path(self, target_date: date) -> Path:
        return self.data_dir / f"{target_date.isoformat()}.json"

    def posted_state_path(self) -> Path:
        return self.data_dir / "posted.json"

    def diary_exists(self, target_date: date) -> bool:
        return self.diary_path(target_date).exists()

    def was_posted(self, target_date: date) -> bool:
        state = self._load_posted_state()
        return target_date.isoformat() in state

    def mark_posted(self, target_date: date, *, channel_id: int) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        state = self._load_posted_state()
        state[target_date.isoformat()] = {
            "posted_at": datetime.now(ZoneInfo(self.timezone)).isoformat(timespec="seconds"),
            "channel_id": channel_id,
        }
        self.posted_state_path().write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def generate(
        self,
        target_date: date,
        *,
        save: bool = True,
        overwrite: bool = False,
        include_calendar: bool = True,
        calendar_id: str | list[str] = "primary",
        calendar_account: str | list[str] | None = None,
    ) -> DailyDiaryResult:
        md_path = self.diary_path(target_date)
        meta_path = self.metadata_path(target_date)
        if save and md_path.exists() and not overwrite:
            return DailyDiaryResult(
                target_date=target_date.isoformat(),
                markdown=md_path.read_text(encoding="utf-8"),
                markdown_path=str(md_path),
                metadata_path=str(meta_path),
                generated=False,
            )

        sources = self.collector.collect(
            target_date,
            include_calendar=include_calendar,
            calendar_id=calendar_id,
            calendar_account=calendar_account,
        )
        markdown = self._generate_markdown(sources)
        calendar_error = str(sources.calendar.get("error") or "")

        if save:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            md_path.write_text(markdown, encoding="utf-8")
            metadata = {
                "target_date": target_date.isoformat(),
                "generated_at": datetime.now(ZoneInfo(self.timezone)).isoformat(timespec="seconds"),
                "markdown_path": str(md_path),
                "calendar_error": calendar_error,
                "source_counts": {
                    "calendar_events": len(sources.calendar.get("events", [])),
                    "manual_schedule": len(sources.manual_schedule),
                    "discord_turns": len(sources.discord_turns),
                    "voice_transcripts": len(sources.voice_transcripts),
                    "recent_summaries": len(sources.recent_summaries),
                    "metrics_samples": sources.metrics_summary.get("sample_count"),
                },
            }
            meta_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return DailyDiaryResult(
            target_date=target_date.isoformat(),
            markdown=markdown,
            markdown_path=str(md_path),
            metadata_path=str(meta_path),
            generated=True,
            calendar_error=calendar_error,
        )

    def _generate_markdown(self, sources: DiarySources) -> str:
        payload = json.dumps(asdict(sources), ensure_ascii=False, indent=2)
        prompt = (
            "次のローカルログから、その日の短い日記をMarkdownで作成してください。\n"
            "構成は「# YYYY-MM-DD の日記」「今日の流れ」「残ったこと」程度で十分です。\n\n"
            f"入力JSON:\n{payload}"
        )
        try:
            text = self.llm.generate(
                [
                    {"role": "system", "content": DIARY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                num_ctx=self.num_ctx,
            ).strip()
        except Exception as exc:
            text = self._fallback_markdown(sources, error=str(exc))

        if not text.startswith("#"):
            text = f"# {sources.target_date} の日記\n\n{text}"
        return text.rstrip() + "\n"

    @staticmethod
    def _fallback_markdown(sources: DiarySources, *, error: str) -> str:
        lines = [
            f"# {sources.target_date} の日記",
            "",
            "LLMでの日記生成に失敗したため、ログから機械的にまとめています。",
            "",
            "## 今日の材料",
        ]
        events = sources.calendar.get("events", [])
        if events:
            lines.append(f"- カレンダー予定: {len(events)}件")
            for event in events[:12]:
                lines.append(f"  - {event.get('start', '')} {event.get('title', '')}")
        elif sources.calendar.get("error"):
            lines.append(f"- カレンダー取得エラー: {sources.calendar.get('error')}")
        else:
            lines.append("- カレンダー予定: なし")

        if sources.manual_schedule:
            lines.append(f"- 手動スケジュール: {len(sources.manual_schedule)}件")
        if sources.discord_turns:
            lines.append(f"- Discord会話: {len(sources.discord_turns)}ターン")
        if sources.voice_transcripts:
            lines.append(f"- Discord通話文字起こし: {len(sources.voice_transcripts)}件")
        sample_count = sources.metrics_summary.get("sample_count")
        if sample_count:
            lines.append(f"- PCメトリクス: {sample_count}サンプル")

        lines.extend(["", "## 生成エラー", f"```text\n{error[:1500]}\n```"])
        return "\n".join(lines)

    def _load_posted_state(self) -> dict[str, dict]:
        path = self.posted_state_path()
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
