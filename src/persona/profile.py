"""
ユーザープロファイル管理
JSON ベースでユーザーの嗜好・習慣・スケジュール・抽出済み事実を永続化

手動編集 + LLMによる自動抽出の両方に対応。
"""
import json
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional


class UserProfile:
    """
    ユーザープロファイル

    data/profile/user_profile.json に永続化される。
    LLMのシステムプロンプトに注入し、パーソナライズ応答を実現する。
    """

    DEFAULT_PROFILE = {
        "name": "",
        "nickname": "",
        "preferences": {},       # {"food": "カレー", "music": "ジャズ", ...}
        "habits": {},            # {"wake_time": "07:00", "sleep_time": "24:00", ...}
        "schedule": [],          # [{"date": "2026-02-12", "time": "14:00", "title": "会議", "note": ""}]
        "notes": [],             # 自由メモ: ["猫を飼っている", "プログラマー"]
        "extracted_facts": [],   # LLMが会話から抽出した事実 (自動追加)
        "updated_at": "",
    }

    def __init__(self, profile_path: str = "data/profile/user_profile.json"):
        self.profile_path = Path(profile_path)
        self._data: dict = {}

    def load(self) -> dict:
        """プロファイルをロード（なければデフォルト作成）"""
        if self.profile_path.exists():
            with open(self.profile_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
            # デフォルトキーで補完
            for key, default_val in self.DEFAULT_PROFILE.items():
                if key not in self._data:
                    self._data[key] = default_val
        else:
            self._data = dict(self.DEFAULT_PROFILE)
            self.save()
        return self._data

    def save(self) -> None:
        """プロファイルを保存"""
        self._data["updated_at"] = datetime.now().isoformat()
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.profile_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    @property
    def data(self) -> dict:
        if not self._data:
            self.load()
        return self._data

    # --- フィールドアクセサ ---

    @property
    def name(self) -> str:
        return self.data.get("name", "")

    @name.setter
    def name(self, value: str):
        self.data["name"] = value
        self.save()

    @property
    def preferences(self) -> dict:
        return self.data.get("preferences", {})

    @property
    def habits(self) -> dict:
        return self.data.get("habits", {})

    @property
    def schedule(self) -> list[dict]:
        return self.data.get("schedule", [])

    @property
    def notes(self) -> list[str]:
        return self.data.get("notes", [])

    @property
    def extracted_facts(self) -> list[str]:
        return self.data.get("extracted_facts", [])

    # --- 操作メソッド ---

    def set_preference(self, key: str, value: str) -> None:
        """嗜好を設定"""
        self.data.setdefault("preferences", {})[key] = value
        self.save()

    def set_habit(self, key: str, value: str) -> None:
        """習慣を設定"""
        self.data.setdefault("habits", {})[key] = value
        self.save()

    def add_schedule(self, title: str, date_str: str, time_str: str = "", note: str = "") -> None:
        """スケジュールを追加"""
        entry = {
            "date": date_str,
            "time": time_str,
            "title": title,
            "note": note,
            "added_at": datetime.now().isoformat(),
        }
        self.data.setdefault("schedule", []).append(entry)
        self.save()

    def remove_past_schedule(self) -> int:
        """過去のスケジュールを削除。削除件数を返す"""
        today = date.today().isoformat()
        before = len(self.schedule)
        self.data["schedule"] = [
            s for s in self.schedule if s.get("date", "") >= today
        ]
        after = len(self.data["schedule"])
        if before != after:
            self.save()
        return before - after

    def get_today_schedule(self) -> list[dict]:
        """今日のスケジュールを取得"""
        today = date.today().isoformat()
        return [s for s in self.schedule if s.get("date", "") == today]

    def get_upcoming_schedule(self, days: int = 7) -> list[dict]:
        """今日から days 日以内のスケジュールを取得"""
        from datetime import timedelta
        today = date.today()
        end_date = (today + timedelta(days=days)).isoformat()
        today_str = today.isoformat()
        return [
            s for s in self.schedule
            if today_str <= s.get("date", "") <= end_date
        ]

    def add_note(self, note: str) -> None:
        """メモを追加"""
        self.data.setdefault("notes", []).append(note)
        self.save()

    def add_extracted_fact(self, fact: str) -> bool:
        """
        LLMが抽出した事実を追加（重複チェック付き）

        Returns:
            追加されたら True
        """
        facts = self.data.setdefault("extracted_facts", [])
        # 完全一致の重複は除外
        if fact in facts:
            return False
        facts.append(fact)
        # 最大100件に制限（古いものから削除）
        if len(facts) > 100:
            self.data["extracted_facts"] = facts[-100:]
        self.save()
        return True

    def add_extracted_facts(self, facts: list[str]) -> int:
        """複数の事実を追加。追加件数を返す"""
        count = 0
        for fact in facts:
            if self.add_extracted_fact(fact.strip()):
                count += 1
        return count

    # --- プロンプト生成 ---

    def get_profile_text(self) -> str:
        """LLMのシステムプロンプトに注入するプロファイルテキスト"""
        lines = []

        if self.name:
            lines.append(f"- ユーザーの名前: {self.name}")
            if self.data.get("nickname"):
                lines.append(f"  ({self.data['nickname']}と呼ばれたい)")

        if self.preferences:
            prefs = ", ".join(f"{k}: {v}" for k, v in self.preferences.items())
            lines.append(f"- 好み・嗜好: {prefs}")

        if self.habits:
            habits = ", ".join(f"{k}: {v}" for k, v in self.habits.items())
            lines.append(f"- 習慣: {habits}")

        if self.notes:
            for note in self.notes[-10:]:  # 最新10件
                lines.append(f"- {note}")

        if self.extracted_facts:
            for fact in self.extracted_facts[-15:]:  # 最新15件
                lines.append(f"- {fact}")

        if not lines:
            return ""

        return "\n--- ユーザープロファイル ---\n" + "\n".join(lines)

    def get_schedule_text(self) -> str:
        """今日のスケジュールをテキスト化"""
        today_schedule = self.get_today_schedule()
        if not today_schedule:
            return ""

        lines = [f"\n--- 今日のスケジュール ({date.today().strftime('%m/%d %A')}) ---"]
        for s in sorted(today_schedule, key=lambda x: x.get("time", "")):
            time_str = s.get("time", "")
            title = s.get("title", "")
            note = s.get("note", "")
            if time_str:
                lines.append(f"- {time_str} {title}")
            else:
                lines.append(f"- {title}")
            if note:
                lines.append(f"  ({note})")

        upcoming = self.get_upcoming_schedule(days=3)
        future = [s for s in upcoming if s.get("date", "") != date.today().isoformat()]
        if future:
            lines.append("")
            lines.append("近日中の予定:")
            for s in sorted(future, key=lambda x: (x.get("date", ""), x.get("time", ""))):
                d = s.get("date", "")
                t = s.get("time", "")
                lines.append(f"  - {d} {t} {s.get('title', '')}")

        return "\n".join(lines)

    def get_status(self) -> dict:
        """APIレスポンス用"""
        return {
            "name": self.name,
            "preferences_count": len(self.preferences),
            "habits_count": len(self.habits),
            "schedule_count": len(self.schedule),
            "today_schedule_count": len(self.get_today_schedule()),
            "notes_count": len(self.notes),
            "extracted_facts_count": len(self.extracted_facts),
            "updated_at": self.data.get("updated_at", ""),
        }
