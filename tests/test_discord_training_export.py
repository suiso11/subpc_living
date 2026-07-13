from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from src.discord_bot.training import DiscordTrainingLog
from src.growth.tracker import GrowthTracker
from scripts.export_discord_training import (
    export_preference,
    export_sft,
    feedback_scores,
    turn_index,
)


class DiscordTrainingExportTest(unittest.TestCase):
    def test_training_feedback_and_correction_emit_growth_signals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tracker = GrowthTracker(root / "growth.db")
            log = DiscordTrainingLog(root / "training", growth_tracker=tracker)
            log.record_turn(
                guild_id=1, channel_id=10, user_id=20, user_message_id=30,
                assistant_message_ids=[40], user_text="質問", assistant_text="返答",
                model="test", num_ctx=1024,
            )
            self.assertTrue(log.record_feedback(
                assistant_message_id=40, guild_id=1, channel_id=10,
                user_id=20, emoji="👍",
            ))
            self.assertTrue(log.record_correction(
                assistant_message_id=40, channel_id=10, guild_id=1,
                user_id=20, correction_message_id=50, corrected_text="より良い返答",
            ).ok)

            summary = tracker.summary()
            self.assertEqual(summary["tracked_points"], 55)
            self.assertEqual(summary["signals"]["training_turn"], 1)
            self.assertEqual(summary["signals"]["feedback"], 1)
            self.assertEqual(summary["signals"]["correction"], 1)

    def test_training_log_preserves_profile_metadata_in_correction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = DiscordTrainingLog(Path(tmp), enabled=True, system_prompt="")
            turn_id = log.record_turn(
                guild_id=1,
                channel_id=10,
                user_id=20,
                user_message_id=30,
                assistant_message_ids=[40],
                user_text="眠い",
                assistant_text="寝なさい。",
                model="gemma4:26b",
                num_ctx=8192,
                source="discord_voice_transcript",
                profile="voice_short",
                num_predict=96,
                temperature=0.35,
            )

            result = log.record_correction(
                assistant_message_id=40,
                channel_id=10,
                guild_id=1,
                user_id=20,
                correction_message_id=50,
                corrected_text="それなら10分だけ休みましょ。",
            )

            self.assertTrue(result.ok)
            self.assertEqual(result.turn_id, turn_id)
            candidates = log.candidates_path.read_text(encoding="utf-8")
            self.assertIn('"profile":"voice_short"', candidates)
            self.assertIn('"turn_source":"discord_voice_transcript"', candidates)
            self.assertIn('"num_predict":96', candidates)

    def test_preference_export_filters_voice_short_corrections(self) -> None:
        turns = [
            {
                "turn_id": "voice-1",
                "source": "discord_voice_transcript",
                "profile": "voice_short",
                "channel_id": 10,
                "model": "gemma4:26b",
                "num_ctx": 8192,
                "num_predict": 96,
                "temperature": 0.35,
                "user": "眠い",
                "assistant": "寝なさい。",
            },
            {
                "turn_id": "normal-1",
                "source": "discord_auto_reply",
                "profile": "default",
                "channel_id": 20,
                "user": "眠い",
                "assistant": "少し休みましょう。",
            },
        ]
        candidates = [
            {
                "turn_id": "voice-1",
                "input": "眠い",
                "preferred_output": "それなら10分だけ休みましょ。",
                "rejected_output": "寝なさい。",
            },
            {
                "turn_id": "normal-1",
                "input": "眠い",
                "preferred_output": "少し休みましょう。",
                "rejected_output": "寝なさい。",
            },
        ]

        rows = export_preference(
            candidates=candidates,
            turns_by_id=turn_index(turns),
            profile="voice_short",
            source="discord_voice_transcript",
            channel_id=None,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["prompt"], "眠い")
        self.assertEqual(rows[0]["chosen"], "それなら10分だけ休みましょ。")
        self.assertEqual(rows[0]["metadata"]["profile"], "voice_short")
        self.assertEqual(rows[0]["metadata"]["source"], "discord_voice_transcript")

    def test_sft_export_requires_explicit_correction_by_default(self) -> None:
        turns = [
            {
                "turn_id": "voice-1",
                "source": "discord_voice_transcript",
                "profile": "voice_short",
                "user": "どうする",
                "assistant": "やりなさい。",
            },
            {
                "turn_id": "voice-2",
                "source": "discord_voice_transcript",
                "profile": "voice_short",
                "user": "進捗どう",
                "assistant": "少し進んでます。",
            },
        ]
        candidates = [
            {
                "turn_id": "voice-1",
                "input": "どうする",
                "preferred_output": "まず一個だけ置いてみましょ。",
                "rejected_output": "やりなさい。",
            }
        ]
        feedback = [{"turn_id": "voice-2", "value": 1}]

        default_rows = export_sft(
            candidates=candidates,
            turns=turns,
            turns_by_id=turn_index(turns),
            scores=feedback_scores(feedback),
            profile="voice_short",
            source="discord_voice_transcript",
            channel_id=None,
            include_positive_feedback=False,
            min_score=1,
        )
        expanded_rows = export_sft(
            candidates=candidates,
            turns=turns,
            turns_by_id=turn_index(turns),
            scores=feedback_scores(feedback),
            profile="voice_short",
            source="discord_voice_transcript",
            channel_id=None,
            include_positive_feedback=True,
            min_score=1,
        )

        self.assertEqual(len(default_rows), 1)
        self.assertEqual(default_rows[0]["messages"][1]["content"], "まず一個だけ置いてみましょ。")
        self.assertEqual(len(expanded_rows), 2)


if __name__ == "__main__":
    unittest.main()
