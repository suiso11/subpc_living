"""応答品質の軽量フィルタ。

長期記憶/RAGへ保存・注入すると会話を汚染しやすい、沈黙記号や
拒否定形だけの低情報応答を検出する。
"""
from __future__ import annotations

import re

_EMOTION_TAG_RE = re.compile(r"^\s*\[(?:emo:)?[a-zA-Z_:-]+\]\s*")
_SPACE_RE = re.compile(r"\s+")

# 以前の会話で大量に増殖した、情報量の低い終了・拒否定形。
_BAD_FIXED_PHRASES = (
    "さっさと、寝なさい",
    "さっさと寝なさい",
    "また、これですか",
    "またこれですか",
)

_SOFT_FIXED_PHRASES = (
    "もう、ええわ",
    "もうええわ",
)


def _normalize(text: str) -> str:
    text = _EMOTION_TAG_RE.sub("", str(text or ""))
    return _SPACE_RE.sub("", text)


def is_low_value_memory_text(text: str) -> bool:
    """RAG長期記憶に入れない方がよい低情報応答なら True。

    出力そのものを禁止するためではなく、過去記憶として再注入されて
    反復を誘発する文だけを弾く目的の判定。
    """
    normalized = _normalize(text)
    if not normalized:
        return True

    if any(phrase in normalized for phrase in _BAD_FIXED_PHRASES):
        return True

    # 「……。」が複数回続く沈黙応答。全角三点リーダ・中黒なしの点・句点だけの
    # バリエーションをざっくり拾う。
    ellipsis_runs = normalized.count("……") + normalized.count("………") + normalized.count("...")
    silence_sentences = len(re.findall(r"(?:…{2,}|\.{3,}|。{2,})。?", normalized))
    if ellipsis_runs >= 3 or silence_sentences >= 3:
        return True

    if any(phrase in normalized for phrase in _SOFT_FIXED_PHRASES) and (ellipsis_runs >= 1 or silence_sentences >= 1):
        return True

    # 記号と短い定型句だけで実質的な情報がないもの。
    content_chars = re.sub(r"[…。．.、，,\[\]（）()「」『』\-ー_\s]", "", normalized)
    if len(content_chars) <= 8 and (ellipsis_runs >= 1 or "寝なさい" in normalized):
        return True

    return False


def should_store_rag_turn(user_message: str, assistant_message: str) -> bool:
    """1ターンをRAGに保存してよいかを返す。"""
    _ = user_message  # 将来、ユーザー側のノイズ判定を足す余地を残す。
    return not is_low_value_memory_text(assistant_message)
