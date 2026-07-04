from __future__ import annotations

import unittest

from src.chat.emotion import (
    EmotionTagStreamFilter,
    VALID_EMOTIONS,
    emotion_to_sbv2_style,
    parse_emotion_tag,
)


class ParseEmotionTagTest(unittest.TestCase):
    def test_basic_tag(self) -> None:
        emotion, clean = parse_emotion_tag("[emo:happy]こんにちは")
        self.assertEqual(emotion, "happy")
        self.assertEqual(clean, "こんにちは")

    def test_no_tag_returns_neutral_and_original(self) -> None:
        emotion, clean = parse_emotion_tag("こんにちは")
        self.assertEqual(emotion, "neutral")
        self.assertEqual(clean, "こんにちは")

    def test_empty_text(self) -> None:
        self.assertEqual(parse_emotion_tag(""), ("neutral", ""))

    def test_invalid_emotion_is_neutral_but_tag_removed(self) -> None:
        emotion, clean = parse_emotion_tag("[emo:xyz]本文")
        self.assertEqual(emotion, "neutral")
        self.assertEqual(clean, "本文")

    def test_case_insensitive(self) -> None:
        emotion, clean = parse_emotion_tag("[emo:Happy]や")
        self.assertEqual(emotion, "happy")
        self.assertEqual(clean, "や")

    def test_leading_whitespace_and_space_after_tag(self) -> None:
        emotion, clean = parse_emotion_tag("  [emo:sad]  しょんぼり")
        self.assertEqual(emotion, "sad")
        self.assertEqual(clean, "しょんぼり")

    def test_fullwidth_brackets_and_colon(self) -> None:
        emotion, clean = parse_emotion_tag("［emo：angry］むむ")
        self.assertEqual(emotion, "angry")
        self.assertEqual(clean, "むむ")

    def test_tag_only_in_middle_is_not_stripped(self) -> None:
        emotion, clean = parse_emotion_tag("前置き[emo:happy]後")
        self.assertEqual(emotion, "neutral")
        self.assertEqual(clean, "前置き[emo:happy]後")

    def test_newline_after_tag_is_stripped(self) -> None:
        emotion, clean = parse_emotion_tag("[emo:surprise]\n本文")
        self.assertEqual(emotion, "surprise")
        self.assertEqual(clean, "本文")


class EmotionToStyleTest(unittest.TestCase):
    def test_all_emotions_map(self) -> None:
        expected = {
            "happy": "Happy",
            "sad": "Sad",
            "angry": "Angry",
            "surprise": "Surprise",
            "fear": "Fear",
            "disgust": "Disgust",
            "neutral": "Neutral",
        }
        for emo, style in expected.items():
            self.assertEqual(emotion_to_sbv2_style(emo), style)

    def test_unknown_and_none_default_to_neutral(self) -> None:
        self.assertEqual(emotion_to_sbv2_style("bogus"), "Neutral")
        self.assertEqual(emotion_to_sbv2_style(None), "Neutral")
        self.assertEqual(emotion_to_sbv2_style(""), "Neutral")

    def test_valid_emotions_all_map_to_known_styles(self) -> None:
        known = {"Happy", "Sad", "Angry", "Surprise", "Fear", "Disgust", "Neutral"}
        for emo in VALID_EMOTIONS:
            self.assertIn(emotion_to_sbv2_style(emo), known)


class EmotionTagStreamFilterTest(unittest.TestCase):
    @staticmethod
    def _run(chunks: list[str]) -> tuple[str, str]:
        f = EmotionTagStreamFilter()
        out = "".join(f.feed(c) for c in chunks)
        out += f.flush()
        return f.emotion, out

    def test_single_chunk(self) -> None:
        emotion, out = self._run(["[emo:happy]こんにちは"])
        self.assertEqual(emotion, "happy")
        self.assertEqual(out, "こんにちは")

    def test_tag_split_across_chunks(self) -> None:
        emotion, out = self._run(["[em", "o:ha", "ppy]こん", "にちは"])
        self.assertEqual(emotion, "happy")
        self.assertEqual(out, "こんにちは")

    def test_tag_split_at_every_char(self) -> None:
        emotion, out = self._run(list("[emo:sad]しょんぼり"))
        self.assertEqual(emotion, "sad")
        self.assertEqual(out, "しょんぼり")

    def test_no_tag_passes_through(self) -> None:
        emotion, out = self._run(["ふつうの", "テキスト", "です"])
        self.assertEqual(emotion, "neutral")
        self.assertEqual(out, "ふつうのテキストです")

    def test_invalid_tag_neutral_but_removed(self) -> None:
        emotion, out = self._run(["[emo:xyz]", "本文"])
        self.assertEqual(emotion, "neutral")
        self.assertEqual(out, "本文")

    def test_text_starting_with_bracket_but_not_tag(self) -> None:
        emotion, out = self._run(["[メモ] 買い物"])
        self.assertEqual(emotion, "neutral")
        self.assertEqual(out, "[メモ] 買い物")

    def test_scan_limit_gives_up_when_tag_incomplete(self) -> None:
        f = EmotionTagStreamFilter(scan_limit=4)
        # 最初のチャンクで上限 4 文字を超え、まだタグが完成していない
        # → 諦めて素通しに切り替える (以降タグ扱いしない)
        out = f.feed("[emo:") + f.feed("happy]x")
        self.assertEqual(f.emotion, "neutral")
        self.assertEqual(out, "[emo:happy]x")

    def test_empty_chunks_ignored(self) -> None:
        emotion, out = self._run(["", "[emo:fear]", "", "こわい"])
        self.assertEqual(emotion, "fear")
        self.assertEqual(out, "こわい")


if __name__ == "__main__":
    unittest.main()
