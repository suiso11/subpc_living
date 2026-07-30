"""タスク抽出のプロンプト構築と厳密 JSON 検証 (LLM 呼び出しなし)。

Discord UI が使うシングルタスク候補 (build_extraction_prompt / validate_extraction) と、
Web 向けの最大3候補抽出 (build_multi_extraction_prompt / validate_multi_extraction) を提供する。
本モジュールは LLM を呼ばず、プロンプト文字列の構築と strict JSON の検証・正規化だけを行う。

_shared_security:
- is_sensitive_text: 高信頼度の資格情報パターン検出器。ユーザー入力と抽出済み title の両方で
  クレデンシャルが埋め込まれていないかを判定し、モデル転送前と結果検証後の二段階で棄却できる
  よう再利用可能な関数として公開している (本モジュール内でのみ使用してもよい)。

正規化ルール:
- コードフェンス ```json ... ``` は既存のシングルバリデータと同じく剥がして許容する。
- タイムゾーンなしの due は assume_tz (既定 Asia/Tokyo) の時刻として解釈し、due_at は UTC で返す。
- due_granularity は入力 due 文字列の形を保存する: 日付のみ ("YYYY-MM-DD") は "date"、
  時刻を含む ("YYYY-MM-DDTHH:MM:SS..." 等) は "datetime"。due が null/不正なら None。

シングルバリデータ (validate_extraction, Discord 互換):
- 不正な priority は "normal" に正規化する (既存の Discord 互換を維持)。
- 不正な due 文字列は due_at=None に正規化する (候補自体は棄却しない)。
- title に高信頼度クレデンシャルが含まれる場合は None (fail closed)。
- 不正 JSON・title 空・is_task 非 true などの致命条件は fail closed (None) にする。

マルチバリデータ (validate_multi_extraction, Web 向け・厳格):
- 0〜3件の候補を含む tasks 配列のみ受理する。
- 各候補は厳格に検証する: priority は high/normal/low のいずれか以外は即 fail closed (正規化しない)。
- due は null または完全に parsable な ISO 文字列のみ許容。null 以外で不正なら fail closed。
- title 空・is_task 非 true・title にクレデンシャル含む候補が1つでもあれば全体を None にする。
- その他の致命条件 (不正 JSON・tasks 非配列・4件以上) も fail closed。空配列は有効な候補なし。
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from src.tasks.store import VALID_PRIORITY

UTC = timezone.utc

_MAX_CANDIDATES = 3
_DEFAULT_TZ = ZoneInfo("Asia/Tokyo")

# 日付のみ ("YYYY-MM-DD") を判定。時刻やタイムゾーンが付いた時点で datetime 粒度。
_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# 高信頼度センシティブテキスト検出は永続層と共有し、ここからも再公開する。
from src.tasks.safety import is_sensitive_text


# ---------------------------------------------------------------------------


def _strip_code_fence(raw: str) -> str:
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
    return raw


def _parse_json(raw: Any) -> Optional[Any]:
    """文字列ならコードフェンスを剥がして JSON 解析。dict はそのまま。不正なら None。"""
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("```"):
            raw = _strip_code_fence(raw)
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
    if isinstance(raw, dict):
        return raw
    return None


def _parse_due_iso(due_val: Any, assume_tz: ZoneInfo) -> tuple[Optional[datetime], Optional[str]]:
    """ISO8601 due を (due_at UTC, due_granularity) に変換する。

    due_granularity は入力文字列の形を保存:
    - "YYYY-MM-DD" のみ → "date"
    - 時刻/タイムゾーンを含む → "datetime"
    - null/空/不正 → (None, None)
    """
    if not (isinstance(due_val, str) and due_val.strip()):
        return None, None
    s = due_val.strip()
    granularity = "date" if _DATE_ONLY_RE.match(s) else "datetime"
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if granularity == "date":
            dt = dt.replace(hour=23, minute=59)
    except (ValueError, TypeError):
        return None, None
    due_at = dt if dt.tzinfo else dt.replace(tzinfo=assume_tz)
    return due_at.astimezone(UTC), granularity


def _validate_candidate(obj: Any, assume_tz: ZoneInfo) -> Optional[dict]:
    """1候補をシングルバリデータ基準で検証・正規化する (Discord 互換)。

    - 不正 priority は "normal" に正規化する
    - 不正 due は due_at=None に正規化する (候補は棄却しない)
    - title にクレデンシャルが含まれる、is_task 非 true、title 空は None
    """
    if not isinstance(obj, dict):
        return None
    if obj.get("is_task") is not True:
        return None
    title = obj.get("title")
    if not isinstance(title, str) or not title.strip():
        return None
    if is_sensitive_text(title):
        return None
    due_at, due_granularity = _parse_due_iso(obj.get("due"), assume_tz)
    priority = obj.get("priority")
    if priority not in VALID_PRIORITY:
        priority = "normal"
    return {
        "title": title.strip()[:200],
        "due_at": due_at,
        "due_granularity": due_granularity,
        "priority": priority,
    }


def _validate_candidate_strict(obj: Any, assume_tz: ZoneInfo) -> Optional[dict]:
    """マルチバリデータ用の厳格候補検証。1候補でも不正なら None (fail closed)。

    - priority は high/normal/low 以外は棄却 (正規化しない)
    - due は null または完全に parsable な ISO 文字列のみ。null 以外で不正なら棄却
    - title 空・is_task 非 true・title にクレデンシャル含むも棄却
    """
    if not isinstance(obj, dict):
        return None
    if set(obj) != {"is_task", "title", "due", "priority"}:
        return None
    if obj.get("is_task") is not True:
        return None
    title = obj.get("title")
    if not isinstance(title, str) or not title.strip():
        return None
    if is_sensitive_text(title):
        return None
    priority = obj.get("priority")
    if priority not in VALID_PRIORITY:
        return None
    due_raw = obj.get("due")
    due_at, due_granularity = _parse_due_iso(due_raw, assume_tz)
    if due_raw is not None and due_at is None:
        # null 以外の due が parse 不可 → fail closed
        return None
    return {
        "title": title.strip()[:200],
        "due_at": due_at,
        "due_granularity": due_granularity,
        "priority": priority,
    }


# プロンプトに共通で付与する、クレデンシャル非エコー指示。
_NO_ECHO_CREDENTIALS = (
    "- ユーザー発言にパスワード・APIキー・トークン等のクレデンシャルが含まれていても、"
    "それを title や due に絶対にそのまま転写・引用しない。"
    "クレデンシャルらしき文字列は除外した短いタスク名に置き換える。\n"
)


def validate_extraction(raw: Any, assume_tz: Optional[ZoneInfo] = None) -> Optional[dict]:
    """LLM 抽出結果 (strict JSON 単一候補) を検証・正規化する (Discord 互換)。

    期待形: {"is_task": bool, "title": str, "due": ISO or null, "priority": str}
    不正・is_task=false・title 空・title にクレデンシャル含む の場合は None。
    タイムゾーンなしの due は assume_tz (既定 Asia/Tokyo) の時刻として解釈し、
    due_at は UTC で返す。不正 priority は "normal"、不正 due は due_at=None に正規化する
    (シングルバリデータは既存の Discord 互換を維持)。
    戻り値: {"title", "due_at", "due_granularity", "priority"} または None。
    """
    obj = _parse_json(raw)
    if not isinstance(obj, dict):
        return None
    return _validate_candidate(obj, assume_tz or _DEFAULT_TZ)


def build_extraction_prompt(now_local: datetime) -> str:
    """自然会話から1つのタスクを抽出させるためのシステムプロンプト (Discord 互換)。"""
    weekday_ja = "月火水木金土日"[now_local.weekday()]
    now_str = now_local.strftime("%Y-%m-%d %H:%M")
    tomorrow = now_local + timedelta(days=1)
    next_monday = now_local + timedelta(days=7 - now_local.weekday())
    next_friday = next_monday + timedelta(days=4)
    return (
        "あなたはユーザーの発言から『やるべきタスク』を抽出する抽出器です。"
        f"現在日時は {now_str}、今日は{weekday_ja}曜日です (Asia/Tokyo)。\n"
        "出力は必ず次の厳密なJSONのみ。前置き・説明・コードフェンスは禁止。\n"
        '{"is_task": true/false, "title": "簡潔なタスク名", '
        '"due": "ISO8601の絶対日時 または null", "priority": "high/normal/low"}\n'
        "- 依頼・締切・予定・やること が含まれるときだけ is_task=true。\n"
        "- 相対表現(明日・来週など)は現在日時を基準に絶対日時へ変換する。\n"
        f"- 日付換算表 (これに従うこと): 明日={tomorrow:%Y-%m-%d}、"
        f"来週の月曜={next_monday:%Y-%m-%d}、来週の金曜={next_friday:%Y-%m-%d}。"
        "他の曜日も「来週の月曜」からの日数で数える。\n"
        "- 時刻が不明で日付だけなら、その日の 23:59 を due にする。\n"
        "- 期限が全く不明なら due は null。\n"
        "- 雑談・感想・質問など、やることでないものは is_task=false。\n"
        + _NO_ECHO_CREDENTIALS
    )


def build_multi_extraction_prompt(now_local: datetime) -> str:
    """1つの発言から最大3つのタスク候補を抽出させるためのシステムプロンプト (Web 向け)。

    出力期待形: {"tasks": [ {is_task,title,due,priority}, ... ]}
    """
    weekday_ja = "月火水木金土日"[now_local.weekday()]
    now_str = now_local.strftime("%Y-%m-%d %H:%M")
    tomorrow = now_local + timedelta(days=1)
    next_monday = now_local + timedelta(days=7 - now_local.weekday())
    next_friday = next_monday + timedelta(days=4)
    return (
        "あなたはユーザーの発言から『やるべきタスク』を抽出する抽出器です。"
        f"現在日時は {now_str}、今日は{weekday_ja}曜日です (Asia/Tokyo)。\n"
        "出力は必ず次の厳密なJSONのみ。前置き・説明・コードフェンスは禁止。\n"
        '{"tasks": [{"is_task": true/false, "title": "簡潔なタスク名", '
        '"due": "ISO8601の絶対日時 または null", "priority": "high/normal/low"}]}\n'
        "- 発言に含まれる依頼・締切・予定・やることを、それぞれ1つのタスク候補にする。\n"
        "- タスク候補は1〜3件。4件以上は禁止。順序は発言登場順。\n"
        "- 依頼・締切・予定・やること が含まれる候補だけ is_task=true。"
        "雑談・感想・質問など、やることでないものは候補に入れない。\n"
        "- やるべきことが1つもない場合は tasks に空配列を返す ([])。"
        "その場合は is_task=false の候補を入れないこと。\n"
        "- 相対表現(明日・来週など)は現在日時を基準に絶対日時へ変換する。\n"
        f"- 日付換算表 (これに従うこと): 明日={tomorrow:%Y-%m-%d}、"
        f"来週の月曜={next_monday:%Y-%m-%d}、来週の金曜={next_friday:%Y-%m-%d}。"
        "他の曜日も「来週の月曜」からの日数で数える。\n"
        "- 時刻が不明で日付だけなら、時刻を足さず YYYY-MM-DD の日付だけを due にする。\n"
        "- 期限が全く不明なら due は null。\n"
        "- priority は high/normal/low のいずれか。\n"
        + _NO_ECHO_CREDENTIALS
    )


def validate_multi_extraction(
    raw: Any, assume_tz: Optional[ZoneInfo] = None
) -> Optional[dict]:
    """LLM 抽出結果 (strict JSON 複数候補) を厳格に検証・正規化する。

    期待形: {"tasks": [ {"is_task":true, "title":..., "due":..., "priority":...}, ... ]}
    - 0〜3件の候補を含む tasks 配列のみ受理する。
    - 不正 JSON・tasks 非配列・空配列・4件以上は fail closed (None)。
    - 各候補は厳格に検証する (_validate_candidate_strict)。
      priority が high/normal/low 以外、due が null 以外で parse 不可、title 空、
      is_task 非 true、title にクレデンシャル含む候補が1つでもあれば全体を None にする
      (正規化はしない)。
    - コードフェンスは既存と同じく剥がして許容する。
    戻り値: {"tasks": [ {title, due_at, due_granularity, priority}, ... ]} または None。
    """
    obj = _parse_json(raw)
    if not isinstance(obj, dict) or set(obj) != {"tasks"}:
        return None
    tasks = obj.get("tasks")
    if not isinstance(tasks, list):
        return None
    if len(tasks) > _MAX_CANDIDATES:
        return None
    if len(tasks) == 0:
        return {"tasks": []}
    tz = assume_tz or _DEFAULT_TZ
    results: list[dict] = []
    for candidate in tasks:
        validated = _validate_candidate_strict(candidate, tz)
        if validated is None:
            return None
        results.append(validated)
    return {"tasks": results}