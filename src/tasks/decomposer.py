"""記録タスクの「最初の一歩」と最大3ステップを決定論的に分解する純粋関数。

ネットワーク推論やローカルLLMを使わず、日本語キーワードルールでカテゴリを
推定してテンプレートを埋める。結果は TaskBreakdown として返し、
呼び出し側は TaskStore.update(action_hint=...) 等で元タスクへ付帯情報として保存する。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


MAX_TITLE_LEN = 200


@dataclass(frozen=True)
class TaskBreakdown:
    """分解結果の値オブジェクト。

    first_step は5分以内に着手可能な最初の一歩。steps は1〜3件の後続ステップ。
    """

    category: str
    first_step: str
    steps: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "category": self.category,
            "first_step": self.first_step,
            "steps": list(self.steps),
        }


# カテゴリ -> (判定キーワード, first_step テンプレート, steps テンプレート)
# 上位のカテゴリから順に判定する。{title} はタイトルに置換される。
_RULES: tuple[tuple[str, tuple[str, ...], str, tuple[str, ...]], ...] = (
    (
        "提出",
        ("提出", "アップロード", "納品"),
        "提出先・締切・必要な形式を開いて確認する",
        ("提出するファイルを1箇所に集める", "ファイル名と内容を確認して提出する"),
    ),
    (
        "執筆",
        ("書く", "執筆", "記事", "レポート", "資料", "ドキュメント",
         "論文", "まとめ", "メモする", "記録する", "書き出す"),
        "「{title}」の見出し候補を3つ箇条書きする",
        ("構成案を1枚に書き出す", "1章分だけ書いて保存する"),
    ),
    (
        "調査",
        ("調べる", "調査", "リサーチ", "比較", "検索", "確かめる",
         "わかる", "調べ", "確認する"),
        "「{title}」の検索キーワードを3つ書き出す",
        ("最初の3件だけ読んでメモを残す", "内容を3行で要約する"),
    ),
    (
        "連絡",
        ("連絡", "伝える", "報告", "通知", "依頼", "メール", "電話",
         "チャット", "返信", "送る", "送信"),
        "伝える要点を2行にまとめる",
        ("宛先と送信手段を確認する", "送信内容を下書きで残す"),
    ),
    (
        "買い物",
        ("買う", "購入", "買い物", "発注", "注文", "選ぶ"),
        "必要な数量と予算を1行でメモする",
        ("買う場所か候補を1つ決める", "店頭か注文画面で購入する"),
    ),
    (
        "予約",
        ("予約", "申し込む", "申請", "受付", "予約する"),
        "希望日時を第3候補まで挙げる",
        ("予約に必要な情報を揃える", "予約先を開いて申し込む"),
    ),
    (
        "外出",
        ("外出", "出かけ", "お出かけ", "訪問", "会いに行く", "待ち合わせ"),
        "日時と行き先・待ち合わせ場所を確認する",
        ("必要な持ち物を3つまで用意する", "移動時間を確認して出発時刻を決める"),
    ),
    (
        "片付け",
        ("片付", "片づけ", "掃除", "整理", "断捨離", "しまう"),
        "作業する範囲を1箇所だけ決める",
        ("5分だけ手近な物を分ける", "出した道具は必ず戻す時間を1分とる"),
    ),
    (
        "学習",
        ("勉強", "学習", "学ぶ", "復習", "読む", "問題を解く", "練習"),
        "今日やる範囲だけ1つに絞る",
        ("3行要約で今日の成果を残す", "分からなかった点を1つ書き出す"),
    ),
    (
        "コーディング",
        ("実装", "コード", "作る", "バグ", "不具合", "リファクタ",
         "関数", "テスト", "修正する", "直す"),
        "期待する結果と現状の違いを1行で書く",
        ("再現する最小手順を1つ用意する", "直す対象を1ファイルに絞る"),
    ),
)

_FALLBACK = (
    "汎用",
    "{title}を5分で終えられる範囲だけ1文で書く",
    ("始める前の準備物を3つまで並べる", "終わりの目安を1つ決める"),
)


def _normalize_title(title: Optional[str]) -> str:
    if title is None:
        raise ValueError("title は必須です")
    t = str(title).strip()
    if not t:
        raise ValueError("title は必須です")
    if len(t) > MAX_TITLE_LEN:
        t = t[:MAX_TITLE_LEN]
    return t


def _detect_category(title: str) -> tuple[str, str, tuple[str, ...]]:
    for category, keywords, first_template, steps in _RULES:
        if any(kw in title for kw in keywords):
            return category, first_template, steps
    return _FALLBACK[0], _FALLBACK[1], _FALLBACK[2]


def decompose_task(
    title: str,
    *,
    note: Optional[str] = None,
    action_hint: Optional[str] = None,
) -> TaskBreakdown:
    """記録タスクから5分以内の最初の一歩と1〜3件のステップを純粋関数で生成する。

    note / action_hint はTaskStoreの付帯情報として将来利用される想定だが、
    分解ルールは title のみで決定論的に定まる (-note/action_hint が空でも安全)。
    入力の空文字・異常長は _normalize_title で安全に処理する。
    """
    t = _normalize_title(title)
    # note/action_hint は現状ではルールに使わないが、将来のフックとして
    # 引数にだけ受け取り、副作用を持たない (将来ここで補強キーワードを足す)。
    _ = (note, action_hint)

    category, first_template, step_templates = _detect_category(t)
    first_step = first_template.format(title=t)
    steps = tuple(s.format(title=t) for s in step_templates)

    # 安全網: テンプレート埋めで空になったら空文字を除外。
    first_step = first_step.strip() or _FALLBACK[1].format(title=t)
    steps = tuple(s for s in steps if s.strip())
    if not steps:
        steps = (tuple(_FALLBACK[2])[0].format(title=t),)

    return TaskBreakdown(category=category, first_step=first_step, steps=steps)
