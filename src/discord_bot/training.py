"""Discord feedback logging for future fine-tuning datasets."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import threading
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.growth.tracker import GrowthTracker


GOOD_REACTION = "\N{THUMBS UP SIGN}"
BAD_REACTION = "\N{THUMBS DOWN SIGN}"

CORRECTION_PREFIXES = ("修正", "訂正", "直し", "correct")


@dataclass(frozen=True)
class CorrectionResult:
    ok: bool
    reason: str
    turn_id: str | None = None


class DiscordTrainingLog:
    """Append-only JSONL store for Discord conversations and feedback."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        enabled: bool = True,
        system_prompt: str = "",
        growth_tracker: "GrowthTracker | None" = None,
    ):
        self.root_dir = Path(root_dir)
        self.enabled = enabled
        self.system_prompt = system_prompt
        self.growth_tracker = growth_tracker
        self.conversations_path = self.root_dir / "conversations.jsonl"
        self.feedback_path = self.root_dir / "feedback.jsonl"
        self.candidates_path = self.root_dir / "training_candidates.jsonl"
        self._lock = threading.Lock()
        self._turns_by_assistant_message_id: dict[int, dict] = {}
        self._last_turn_by_channel_id: dict[int, dict] = {}
        self._conversation_count = 0
        self._feedback_count = 0
        self._candidate_count = 0
        if self.enabled:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self._load_index()

    def _load_index(self) -> None:
        self._conversation_count = self._count_jsonl(self.conversations_path)
        self._feedback_count = self._count_jsonl(self.feedback_path)
        self._candidate_count = self._count_jsonl(self.candidates_path)

        if not self.conversations_path.exists():
            return
        with self.conversations_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    turn = json.loads(line)
                except json.JSONDecodeError:
                    continue
                channel_id = turn.get("channel_id")
                if isinstance(channel_id, int):
                    self._last_turn_by_channel_id[channel_id] = turn
                for message_id in turn.get("assistant_message_ids", []):
                    if isinstance(message_id, int):
                        self._turns_by_assistant_message_id[message_id] = turn

    @staticmethod
    def _count_jsonl(path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def _append_jsonl(path: Path, record: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, separators=(",", ":"))
            f.write("\n")

    def record_turn(
        self,
        *,
        guild_id: int | None,
        channel_id: int,
        user_id: int,
        user_message_id: int,
        assistant_message_ids: list[int],
        user_text: str,
        assistant_text: str,
        model: str,
        num_ctx: int,
        source: str = "discord_auto_reply",
        profile: str = "default",
        num_predict: int | None = None,
        temperature: float | None = None,
    ) -> str | None:
        if not self.enabled:
            return None

        turn = {
            "turn_id": uuid.uuid4().hex,
            "created_at": self._now(),
            "source": source,
            "profile": profile,
            "guild_id": guild_id,
            "channel_id": channel_id,
            "user_id": user_id,
            "user_message_id": user_message_id,
            "assistant_message_ids": assistant_message_ids,
            "model": model,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
            "user": user_text,
            "assistant": assistant_text,
        }
        with self._lock:
            self._append_jsonl(self.conversations_path, turn)
            self._conversation_count += 1
            self._last_turn_by_channel_id[channel_id] = turn
            for message_id in assistant_message_ids:
                self._turns_by_assistant_message_id[message_id] = turn
        if self.growth_tracker is not None:
            try:
                self.growth_tracker.record_signal(
                    kind="training_turn",
                    source=source,
                    event_key=f"training:{turn['turn_id']}",
                )
            except Exception:
                pass
        return turn["turn_id"]

    def get_turn_by_assistant_message_id(self, assistant_message_id: int) -> dict | None:
        """bot返答メッセージIDから会話ターンを引く (見つからなければ None)。"""
        with self._lock:
            return self._turns_by_assistant_message_id.get(assistant_message_id)

    def record_feedback(
        self,
        *,
        assistant_message_id: int,
        guild_id: int | None,
        channel_id: int,
        user_id: int,
        emoji: str,
    ) -> bool:
        if not self.enabled:
            return False
        value = reaction_value(emoji)
        if value is None:
            return False
        with self._lock:
            turn = self._turns_by_assistant_message_id.get(assistant_message_id)
            if turn is None:
                return False
            record = {
                "feedback_id": uuid.uuid4().hex,
                "created_at": self._now(),
                "source": "discord_reaction",
                "turn_id": turn["turn_id"],
                "assistant_message_id": assistant_message_id,
                "guild_id": guild_id,
                "channel_id": channel_id,
                "user_id": user_id,
                "emoji": emoji,
                "value": value,
            }
            self._append_jsonl(self.feedback_path, record)
            self._feedback_count += 1
        if self.growth_tracker is not None:
            try:
                self.growth_tracker.record_signal(
                    kind="feedback",
                    source="discord",
                    event_key=f"feedback:{record['feedback_id']}",
                    metadata={"value": value},
                )
            except Exception:
                pass
        return True

    def record_correction(
        self,
        *,
        assistant_message_id: int | None,
        channel_id: int,
        guild_id: int | None,
        user_id: int,
        correction_message_id: int,
        corrected_text: str,
    ) -> CorrectionResult:
        if not self.enabled:
            return CorrectionResult(False, "training log is disabled")
        corrected_text = corrected_text.strip()
        if not corrected_text:
            return CorrectionResult(False, "修正文が空です")

        with self._lock:
            turn = None
            if assistant_message_id is not None:
                turn = self._turns_by_assistant_message_id.get(assistant_message_id)
                if turn is None:
                    return CorrectionResult(False, "返信先のbot返答ログが見つかりません")
            else:
                turn = self._last_turn_by_channel_id.get(channel_id)
            if turn is None:
                return CorrectionResult(False, "対応するbot返答ログが見つかりません")

            candidate = {
                "candidate_id": uuid.uuid4().hex,
                "created_at": self._now(),
                "source": "discord_correction",
                "turn_id": turn["turn_id"],
                "guild_id": guild_id,
                "channel_id": channel_id,
                "user_id": user_id,
                "correction_message_id": correction_message_id,
                "user_message_id": turn.get("user_message_id"),
                "assistant_message_ids": turn.get("assistant_message_ids", []),
                "model": turn.get("model"),
                "num_ctx": turn.get("num_ctx"),
                "num_predict": turn.get("num_predict"),
                "temperature": turn.get("temperature"),
                "turn_source": turn.get("source"),
                "profile": turn.get("profile", "default"),
                "messages": self._build_sft_messages(turn["user"], corrected_text),
                "input": turn["user"],
                "preferred_output": corrected_text,
                "rejected_output": turn["assistant"],
            }
            self._append_jsonl(self.candidates_path, candidate)
            self._candidate_count += 1
        if self.growth_tracker is not None:
            try:
                self.growth_tracker.record_signal(
                    kind="correction",
                    source="discord",
                    event_key=f"correction:{candidate['candidate_id']}",
                )
            except Exception:
                pass
        return CorrectionResult(True, "saved", turn_id=turn["turn_id"])

    def _build_sft_messages(self, user_text: str, assistant_text: str) -> list[dict]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})
        return messages

    def summary_text(self) -> str:
        if not self.enabled:
            return "training_log: disabled"
        return (
            "training_log: enabled\n"
            f"training_dir: {self.root_dir}\n"
            f"logged_turns: {self._conversation_count}\n"
            f"feedback: {self._feedback_count}\n"
            f"candidates: {self._candidate_count}"
        )


def parse_correction(text: str) -> str | None:
    stripped = text.strip()
    for prefix in CORRECTION_PREFIXES:
        for separator in (":", "："):
            marker = f"{prefix}{separator}"
            if stripped.lower().startswith(marker.lower()):
                return stripped[len(marker) :].strip()
    return None


def reaction_value(emoji: str) -> int | None:
    if emoji == GOOD_REACTION:
        return 1
    if emoji == BAD_REACTION:
        return -1
    return None
