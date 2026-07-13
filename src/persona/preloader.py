"""
セッションプリロード
会話開始時にユーザープロファイル・要約・時刻情報を統合して
リッチなシステムプロンプトを自動構築する

既存の system_prompt の先頭に、以下を自動追加:
- 現在の日時・曜日
- ユーザープロファイル (嗜好・習慣・メモ)
- 今日のスケジュール
- 直近の会話要約
"""
from datetime import datetime
from typing import Optional

from src.persona.profile import UserProfile
from src.persona.summarizer import ConversationSummarizer


# 曜日の日本語変換
_WEEKDAYS_JA = ["月曜日", "火曜日", "水曜日", "木曜日", "金曜日", "土曜日", "日曜日"]


class SessionPreloader:
    """
    セッションプリロード

    会話開始時に呼ばれ、コンテキスト豊富なシステムプロンプトを構築する。
    - 現在の日時・曜日を注入
    - ユーザープロファイルを注入
    - 今日のスケジュール・近日中の予定を注入
    - 直近の会話要約を注入
    """

    def __init__(
        self,
        profile: UserProfile,
        summarizer: ConversationSummarizer,
    ):
        self.profile = profile
        self.summarizer = summarizer

    def build_preload_context(self) -> str:
        """
        セッション開始時にシステムプロンプトに追加するプリロードコンテキストを構築

        Returns:
            プリロードテキスト（空の場合は ""）
        """
        sections = []

        # 1. 現在の日時
        now = datetime.now()
        weekday = _WEEKDAYS_JA[now.weekday()]
        time_context = (
            f"\n--- 現在の状況 ---\n"
            f"- 日時: {now.strftime('%Y年%m月%d日')} ({weekday}) {now.strftime('%H:%M')}"
        )

        # 時間帯に応じた挨拶ヒント
        hour = now.hour
        if 5 <= hour < 10:
            time_context += "\n- 時間帯: 朝"
        elif 10 <= hour < 12:
            time_context += "\n- 時間帯: 午前中"
        elif 12 <= hour < 14:
            time_context += "\n- 時間帯: 昼"
        elif 14 <= hour < 18:
            time_context += "\n- 時間帯: 午後"
        elif 18 <= hour < 22:
            time_context += "\n- 時間帯: 夜"
        else:
            time_context += "\n- 時間帯: 深夜"

        sections.append(time_context)

        # 2. ユーザープロファイル
        profile_text = self.profile.get_profile_text()
        if profile_text:
            sections.append(profile_text)

        # 3. 今日のスケジュール
        schedule_text = self.profile.get_schedule_text()
        if schedule_text:
            sections.append(schedule_text)

        # 4. 直近の会話要約
        summaries_text = self.summarizer.get_recent_summaries_text(count=3)
        if summaries_text:
            sections.append(summaries_text)

        if not sections:
            return ""

        return "\n".join(sections)

    def get_status(self) -> dict:
        """APIレスポンス用"""
        now = datetime.now()
        weekday = _WEEKDAYS_JA[now.weekday()]
        today_schedule = self.profile.get_today_schedule()
        recent_summaries = self.summarizer.get_recent_summaries(count=5)

        return {
            "current_datetime": now.isoformat(),
            "weekday": weekday,
            "profile": self.profile.get_status(),
            "today_schedule_count": len(today_schedule),
            "recent_summaries_count": len(recent_summaries),
        }
