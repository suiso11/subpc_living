"""
会話要約 + 知識抽出エンジン
セッション終了時にLLMを使って会話を要約し、ユーザー情報を自動抽出する

- 会話要約: 5行以内のサマリーを生成して data/profile/summaries/ に保存
- 知識抽出: 会話からユーザーの嗜好・予定・事実を抽出してプロファイルに追記
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.chat.client import OllamaClient
    from src.persona.profile import UserProfile


# --- 要約用プロンプト ---

SUMMARIZE_PROMPT = """\
以下の会話を3〜5行で簡潔に要約してください。
重要なトピック、ユーザーが求めたこと、AIの回答のポイントを含めてください。
要約のみを出力し、他の説明は不要です。

会話:
{conversation}
"""

EXTRACT_FACTS_PROMPT = """\
以下の会話から、ユーザーについて新たに分かった情報を抽出してください。

抽出する情報の例:
- 好み・嗜好（好きな食べ物、趣味、興味のある分野など）
- 個人情報（職業、住んでいる場所、家族構成など）
- 習慣・生活パターン
- 予定・計画
- 困っていること・課題

JSON配列で出力してください。新しい情報がない場合は空配列 [] を返してください。
他の説明は一切不要です。JSON配列のみ出力してください。

例: ["プログラマーとして働いている", "猫を2匹飼っている", "来週末に旅行を計画中"]

会話:
{conversation}
"""


class ConversationSummarizer:
    """
    会話要約 + 知識抽出

    セッション終了時に自動で呼ばれ:
    1. 会話を要約してファイル保存
    2. ユーザー情報を抽出してプロファイルに追記
    """

    def __init__(
        self,
        summaries_dir: str = "data/profile/summaries",
        max_summaries: int = 50,
    ):
        self.summaries_dir = Path(summaries_dir)
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        self.max_summaries = max_summaries

    def _format_conversation(self, messages: list[dict], max_turns: int = 10) -> str:
        """会話メッセージを文字列に変換"""
        # system メッセージは除外、最新 max_turns ターン
        filtered = [m for m in messages if m.get("role") in ("user", "assistant")]
        if len(filtered) > max_turns * 2:
            filtered = filtered[-(max_turns * 2):]

        lines = []
        for msg in filtered:
            role = "ユーザー" if msg["role"] == "user" else "AI"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def summarize_session(
        self,
        llm: "OllamaClient",
        messages: list[dict],
        session_id: str,
        *,
        temperature: float = 0.3,
        num_ctx: int = 4096,
    ) -> Optional[str]:
        """
        セッションを要約してファイル保存

        Args:
            llm: OllamaClient
            messages: セッションの全メッセージ
            session_id: セッションID

        Returns:
            要約テキスト（失敗時 None）
        """
        conversation = self._format_conversation(messages)
        if not conversation:
            return None

        prompt = SUMMARIZE_PROMPT.format(conversation=conversation)
        try:
            summary = llm.generate(
                [{"role": "user", "content": prompt}],
                temperature=temperature,
                num_ctx=num_ctx,
            )
            summary = summary.strip()
        except Exception as e:
            print(f"⚠️  会話要約失敗: {e}")
            return None

        # 保存
        summary_data = {
            "session_id": session_id,
            "summarized_at": datetime.now().isoformat(),
            "turn_count": sum(1 for m in messages if m.get("role") == "user"),
            "summary": summary,
        }

        filepath = self.summaries_dir / f"summary_{session_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)

        # 古いサマリーの整理
        self._cleanup_old_summaries()

        return summary

    def extract_facts(
        self,
        llm: "OllamaClient",
        messages: list[dict],
        profile: "UserProfile",
        *,
        temperature: float = 0.1,
        num_ctx: int = 4096,
    ) -> list[str]:
        """
        会話からユーザー情報を抽出してプロファイルに追加

        Args:
            llm: OllamaClient
            messages: セッションの全メッセージ
            profile: ユーザープロファイル

        Returns:
            抽出された事実のリスト
        """
        conversation = self._format_conversation(messages)
        if not conversation:
            return []

        prompt = EXTRACT_FACTS_PROMPT.format(conversation=conversation)
        try:
            response = llm.generate(
                [{"role": "user", "content": prompt}],
                temperature=temperature,
                num_ctx=num_ctx,
            )
            response = response.strip()

            # JSON配列をパース
            # LLM出力に ```json ... ``` があれば除去
            if "```" in response:
                import re
                match = re.search(r'\[.*?\]', response, re.DOTALL)
                if match:
                    response = match.group(0)

            facts = json.loads(response)
            if not isinstance(facts, list):
                return []

            # プロファイルに追加
            added = profile.add_extracted_facts([str(f) for f in facts if f])
            return facts

        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️  知識抽出失敗: {e}")
            return []

    def process_session_end(
        self,
        llm: "OllamaClient",
        messages: list[dict],
        session_id: str,
        profile: Optional["UserProfile"] = None,
    ) -> dict:
        """
        セッション終了時の一括処理: 要約 + 知識抽出

        Returns:
            {"summary": str|None, "extracted_facts": list}
        """
        result = {"summary": None, "extracted_facts": []}

        # 短い会話（2ターン未満）は処理しない
        user_turns = sum(1 for m in messages if m.get("role") == "user")
        if user_turns < 2:
            return result

        # 要約
        summary = self.summarize_session(llm, messages, session_id)
        result["summary"] = summary

        # 知識抽出
        if profile is not None:
            facts = self.extract_facts(llm, messages, profile)
            result["extracted_facts"] = facts

        return result

    def get_recent_summaries(self, count: int = 5) -> list[dict]:
        """直近の要約を取得"""
        files = sorted(self.summaries_dir.glob("summary_*.json"), reverse=True)
        summaries = []
        for fp in files[:count]:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    summaries.append(json.load(f))
            except Exception:
                pass
        return summaries

    def get_recent_summaries_text(self, count: int = 3) -> str:
        """直近の要約をテキストとして返す（プリロード用）"""
        summaries = self.get_recent_summaries(count=count)
        if not summaries:
            return ""

        lines = ["\n--- 最近の会話の要約 ---"]
        for s in summaries:
            dt = s.get("summarized_at", "")[:10]  # 日付のみ
            summary_text = s.get("summary", "")
            if summary_text:
                lines.append(f"[{dt}] {summary_text}")

        return "\n".join(lines)

    def _cleanup_old_summaries(self) -> None:
        """古いサマリーを削除（max_summaries件を超えたら）"""
        files = sorted(self.summaries_dir.glob("summary_*.json"))
        if len(files) > self.max_summaries:
            for fp in files[: len(files) - self.max_summaries]:
                fp.unlink(missing_ok=True)
