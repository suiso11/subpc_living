"""Daily profile refinement from diary entries.

This module deliberately stores a small, structured profile instead of feeding
raw diary history into every conversation. Each run writes an audit JSON so the
automatic updates can be inspected later.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.persona.profile import UserProfile

if TYPE_CHECKING:
    from src.chat.client import OllamaClient


PERSONALIZER_SYSTEM_PROMPT = """\
あなたは subpc_living のプロフィール更新係です。
日記から、翌日以降の応答に役立つ「安定した情報」だけを抽出してください。

厳守:
- その日だけの出来事、気分、体調、予定を恒久的な事実として保存しない。
- 本人が明示していない性格・病名・能力・価値観を推測して保存しない。
- センシティブな情報は、本人の明示があり、今後の支援に必要な場合だけ短く残す。
- 既存プロフィールと重複する内容は出さない。
- JSONだけを返す。説明文やMarkdownを混ぜない。

出力形式:
{
  "preferences": [{"key": "coffee", "value": "浅煎りが好き", "confidence": 0.82, "reason": "日記に好みとして出ている"}],
  "habits": [{"key": "sleep_pattern", "value": "夜更かし気味", "confidence": 0.76, "reason": "複数ログで示唆"}],
  "notes": [{"text": "大学の授業文脈を優先して解釈する", "confidence": 0.9, "reason": "既存文脈と整合"}],
  "facts": [{"text": "制作や探索を生活の中心に置く傾向がある", "confidence": 0.8, "reason": "日記と既存プロフィールに整合"}]
}
"""


@dataclass(frozen=True)
class PersonalizationResult:
    target_date: str
    dry_run: bool
    audit_path: str
    candidates: dict[str, Any]
    applied: dict[str, list[Any]]
    skipped: list[str]

    @property
    def applied_count(self) -> int:
        return sum(len(v) for v in self.applied.values())


class DailyPersonalizer:
    """Extract and apply stable profile updates from a daily diary."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        llm: "OllamaClient",
        profile_path: str | Path = "data/profile/user_profile.json",
        diary_dir: str | Path = "data/diary",
        audit_dir: str | Path = "data/profile/personalization",
        min_confidence: float = 0.72,
        temperature: float = 0.2,
        num_ctx: int = 8192,
    ):
        self.project_root = Path(project_root)
        self.llm = llm
        self.profile_path = self._resolve(profile_path)
        self.diary_dir = self._resolve(diary_dir)
        self.audit_dir = self._resolve(audit_dir)
        self.min_confidence = min_confidence
        self.temperature = temperature
        self.num_ctx = num_ctx

    def run(
        self,
        target_date: date,
        *,
        diary_markdown: str | None = None,
        dry_run: bool = False,
    ) -> PersonalizationResult:
        diary_text = diary_markdown if diary_markdown is not None else self._load_diary(target_date)
        profile = UserProfile(str(self.profile_path))
        profile.load()

        candidates = self._extract_candidates(profile.data, diary_text, target_date)
        applied, skipped = self._apply_candidates(profile, candidates, dry_run=dry_run)
        audit = {
            "target_date": target_date.isoformat(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dry_run": dry_run,
            "min_confidence": self.min_confidence,
            "candidates": candidates,
            "applied": applied,
            "skipped": skipped,
        }
        audit_path = self.audit_dir / f"{target_date.isoformat()}.json"
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

        return PersonalizationResult(
            target_date=target_date.isoformat(),
            dry_run=dry_run,
            audit_path=str(audit_path),
            candidates=candidates,
            applied=applied,
            skipped=skipped,
        )

    def _extract_candidates(
        self,
        profile_data: dict[str, Any],
        diary_text: str,
        target_date: date,
    ) -> dict[str, Any]:
        prompt = {
            "target_date": target_date.isoformat(),
            "current_profile_digest": self._profile_digest(profile_data),
            "diary_markdown": diary_text[:9000],
        }
        response = self.llm.generate(
            [
                {"role": "system", "content": PERSONALIZER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "次のJSONを読んで、プロフィール更新候補をJSONだけで返してください。\n"
                        + json.dumps(prompt, ensure_ascii=False, indent=2)
                    ),
                },
            ],
            temperature=self.temperature,
            num_ctx=self.num_ctx,
        )
        return self._parse_candidates(response)

    def _apply_candidates(
        self,
        profile: UserProfile,
        candidates: dict[str, Any],
        *,
        dry_run: bool,
    ) -> tuple[dict[str, list[Any]], list[str]]:
        data = profile.data
        data.setdefault("preferences", {})
        data.setdefault("habits", {})
        data.setdefault("notes", [])
        data.setdefault("extracted_facts", [])

        applied: dict[str, list[Any]] = {
            "preferences": [],
            "habits": [],
            "notes": [],
            "facts": [],
        }
        skipped: list[str] = []

        for item in self._candidate_list(candidates, "preferences"):
            key = self._clean_key(item.get("key"))
            value = self._clean_text(item.get("value"), limit=120)
            if not self._passes(item) or not key or not value:
                skipped.append(f"preference skipped: {item}")
                continue
            if data["preferences"].get(key) == value:
                skipped.append(f"preference duplicate: {key}")
                continue
            applied["preferences"].append({"key": key, "value": value})
            if not dry_run:
                data["preferences"][key] = value

        for item in self._candidate_list(candidates, "habits"):
            key = self._clean_key(item.get("key"))
            value = self._clean_text(item.get("value"), limit=120)
            if not self._passes(item) or not key or not value:
                skipped.append(f"habit skipped: {item}")
                continue
            if data["habits"].get(key) == value:
                skipped.append(f"habit duplicate: {key}")
                continue
            applied["habits"].append({"key": key, "value": value})
            if not dry_run:
                data["habits"][key] = value

        existing_notes = {str(note).strip() for note in data["notes"]}
        for item in self._candidate_list(candidates, "notes"):
            text = self._clean_text(item.get("text"), limit=180)
            if not self._passes(item) or not text:
                skipped.append(f"note skipped: {item}")
                continue
            if text in existing_notes:
                skipped.append(f"note duplicate: {text}")
                continue
            applied["notes"].append(text)
            existing_notes.add(text)
            if not dry_run:
                data["notes"].append(text)

        existing_facts = {str(fact).strip() for fact in data["extracted_facts"]}
        for item in self._candidate_list(candidates, "facts"):
            text = self._clean_text(item.get("text"), limit=220)
            if not self._passes(item) or not text:
                skipped.append(f"fact skipped: {item}")
                continue
            if text in existing_facts:
                skipped.append(f"fact duplicate: {text}")
                continue
            applied["facts"].append(text)
            existing_facts.add(text)
            if not dry_run:
                data["extracted_facts"].append(text)

        if not dry_run:
            data["notes"] = data["notes"][-80:]
            data["extracted_facts"] = data["extracted_facts"][-100:]
            profile.save()

        return applied, skipped

    def _load_diary(self, target_date: date) -> str:
        path = self.diary_dir / f"{target_date.isoformat()}.md"
        if not path.exists():
            raise FileNotFoundError(f"diary not found: {path}")
        return path.read_text(encoding="utf-8")

    def _resolve(self, path: str | Path) -> Path:
        result = Path(path)
        if result.is_absolute():
            return result
        return self.project_root / result

    @staticmethod
    def _profile_digest(profile_data: dict[str, Any]) -> dict[str, Any]:
        return {
            "preferences": profile_data.get("preferences", {}),
            "habits": profile_data.get("habits", {}),
            "notes": profile_data.get("notes", [])[-20:]
            if isinstance(profile_data.get("notes"), list)
            else [],
            "extracted_facts": profile_data.get("extracted_facts", [])[-30:]
            if isinstance(profile_data.get("extracted_facts"), list)
            else [],
        }

    @staticmethod
    def _parse_candidates(response: str) -> dict[str, Any]:
        text = response.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise
            data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("personalizer response must be a JSON object")
        for key in ("preferences", "habits", "notes", "facts"):
            if key not in data or not isinstance(data[key], list):
                data[key] = []
        return data

    @staticmethod
    def _candidate_list(candidates: dict[str, Any], key: str) -> list[dict[str, Any]]:
        values = candidates.get(key, [])
        if not isinstance(values, list):
            return []
        return [item for item in values if isinstance(item, dict)]

    def _passes(self, item: dict[str, Any]) -> bool:
        try:
            confidence = float(item.get("confidence", 0))
        except (TypeError, ValueError):
            return False
        return confidence >= self.min_confidence

    @staticmethod
    def _clean_key(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        key = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip().lower()).strip("_")
        return key[:48]

    @staticmethod
    def _clean_text(value: Any, *, limit: int) -> str:
        if not isinstance(value, str):
            return ""
        text = re.sub(r"\s+", " ", value).strip()
        if len(text) > limit:
            text = text[: limit - 1].rstrip() + "…"
        return text
