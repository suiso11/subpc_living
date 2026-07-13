"""
感情タグのプロトコル・パースユーティリティ。

LLM 応答の冒頭に付与される `[emo:happy]` 形式のタグを解釈し、
Style-Bert-VITS2 のスタイル名へマップする。ストリーミング用の
逐次フィルタ (チャンク境界でタグが分断されても正しく処理する) も提供する。
"""
from __future__ import annotations

import re

# 有効な感情ラベル (小文字)
VALID_EMOTIONS = frozenset(
    {"happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"}
)

# 感情ラベル → Style-Bert-VITS2 スタイル名 (jvnv-F1-jp のスタイル)
_EMOTION_TO_STYLE = {
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
    "surprise": "Surprise",
    "fear": "Fear",
    "disgust": "Disgust",
    "neutral": "Neutral",
}

# システムプロンプト末尾へ追加する指示文 (build_messages() で一元的に付与)
EMOTION_TAG_INSTRUCTION = (
    "応答の冒頭に、応答全体の感情を表すタグを必ず1つだけ付けてください。"
    "形式: [emo:happy]。使える感情: happy / sad / angry / surprise / fear / "
    "disgust / neutral。タグは応答の先頭以外に書かないでください。"
)

# デフォルトのスキャン上限: 先頭からこの文字数以内にタグが現れなければ以降素通し
_DEFAULT_SCAN_LIMIT = 24

# 完全なタグ (先頭)。前後空白・全角括弧 ［］・全角コロン ： の揺れに寛容。
_TAG_RE = re.compile(
    r"^[\s　]*[\[［]\s*emo\s*[:：]\s*([A-Za-z]+)\s*[\]］]",
    re.IGNORECASE,
)

# 途中まで入力された「タグになりうる接頭辞」を判定する部分マッチ用。
# 空文字・空白・"["・"[e"・"[em"・"[emo"・"[emo:"・"[emo:ha" などにマッチする。
_PARTIAL_TAG_RE = re.compile(
    r"^[\s　]*"
    r"(?:[\[［]"
    r"(?:\s*"
    r"(?:e(?:m(?:o"
    r"(?:\s*[:：]"
    r"(?:\s*[A-Za-z]*)?"
    r")?)?)?)?"
    r")?)?$",
    re.IGNORECASE,
)


def emotion_to_sbv2_style(emotion: str | None) -> str:
    """感情ラベルを Style-Bert-VITS2 のスタイル名へマップする。

    未知・None は Neutral。
    """
    if not emotion:
        return "Neutral"
    return _EMOTION_TO_STYLE.get(emotion.strip().lower(), "Neutral")


def parse_emotion_tag(text: str) -> tuple[str, str]:
    """応答冒頭の感情タグを解釈し、(emotion, タグ除去後テキスト) を返す。

    - タグが無ければ ("neutral", 原文)。
    - 形式は合っているが感情が不正な場合 (例 [emo:xyz]) は "neutral" とし、
      タグ自体は除去する (ユーザーには見せない)。
    """
    if not text:
        return "neutral", text
    match = _TAG_RE.match(text)
    if match is None:
        return "neutral", text
    emotion = match.group(1).strip().lower()
    if emotion not in VALID_EMOTIONS:
        emotion = "neutral"
    clean = text[match.end():].lstrip()
    return emotion, clean


class EmotionTagStreamFilter:
    """ストリーミング応答から冒頭の感情タグを逐次的に取り除くフィルタ。

    使い方::

        f = EmotionTagStreamFilter()
        for chunk in stream:
            visible = f.feed(chunk)   # タグ部分を除いた表示用テキスト
            ...
        emotion = f.emotion            # 確定した感情 (無ければ "neutral")

    タグがチャンク境界で分断されても ("[em" + "o:ha" + "ppy]") 正しく処理する。
    先頭 `scan_limit` 文字以内にタグが現れなければ、以降はそのまま素通しする。
    """

    def __init__(self, scan_limit: int = _DEFAULT_SCAN_LIMIT) -> None:
        self.scan_limit = scan_limit
        self.emotion = "neutral"
        self._buffer = ""
        self._resolved = False

    def feed(self, chunk: str) -> str:
        """チャンクを与え、表示すべきテキスト (タグ除去後) を返す。"""
        if not chunk:
            return ""
        if self._resolved:
            return chunk

        self._buffer += chunk

        match = _TAG_RE.match(self._buffer)
        if match is not None:
            emotion = match.group(1).strip().lower()
            self.emotion = emotion if emotion in VALID_EMOTIONS else "neutral"
            rest = self._buffer[match.end():].lstrip()
            self._buffer = ""
            self._resolved = True
            return rest

        # まだタグが完成しうるなら、上限までは保留する。
        if _PARTIAL_TAG_RE.match(self._buffer) and len(self._buffer) <= self.scan_limit:
            return ""

        # タグにはなり得ない or 上限超過 → バッファをそのまま吐き出して素通しへ。
        out = self._buffer
        self._buffer = ""
        self._resolved = True
        return out

    def flush(self) -> str:
        """ストリーム終了時に保留中のバッファを吐き出す (タグ未確定のまま終了した場合)。"""
        if self._resolved:
            return ""
        out = self._buffer
        self._buffer = ""
        self._resolved = True
        return out
