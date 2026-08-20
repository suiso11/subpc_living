"""
プロアクティブ発話エンジン
AI側から能動的に話しかけるトリガーを管理する

バックグラウンドで条件を定期チェックし、
トリガー発火時にコールバックで通知する。

トリガー例:
- スケジュールのリマインド（開始15分前）
- 長時間作業の休憩提案（2時間連続PC使用）
- 時間帯の挨拶（朝の挨拶、深夜の声かけ）
- PC異常の通知（高温、メモリ逼迫）
- プロフィール学習につながる会話のきっかけ
"""
import time
import threading
from datetime import datetime
from typing import Optional, Callable

from src.persona.profile import UserProfile
from src.companion.calendar import CalendarSource
from src.companion.contracts import CompanionState
from src.companion.policy import DeterministicProactivePolicy


CONVERSATION_PROMPTS = (
    "最近、時間を忘れてやってしまうくらいハマっていることってある？",
    "作業するとき、静かな方が集中しやすい？ それとも音がある方がいい？",
    "疲れているとき、どんなふうに声をかけられると一番助かる？",
    "逆に、私にしてほしくない話し方や対応ってある？",
    "最近よく聴く音楽や、つい選んでしまう音ってある？",
    "一日の中で、いちばん頭が動きやすいのはだいたい何時ごろ？",
    "今後、私に覚えておいてほしいことを一つ挙げるなら何？",
    "気分転換したいとき、普段は何をすることが多い？",
)

# この時間以上ユーザーの操作が空いたら、前の作業は終了したとみなす。
WORK_SESSION_GAP_SECONDS = 30 * 60


class ProactiveEngine:
    """
    プロアクティブ発話エンジン

    バックグラウンドスレッドで定期的に条件をチェックし、
    トリガー発火時にコールバックを呼ぶ。
    """

    def __init__(
        self,
        profile: UserProfile,
        check_interval: float = 60.0,
        monitor_context=None,
        conversation_enabled: bool = False,
        conversation_interval: float = 21600.0,
        conversation_idle: float = 3600.0,
        quiet_hours: tuple[int, int] = (1, 9),
        conversation_gate: Optional[Callable[[], bool]] = None,
        companion_getter: Optional[Callable[[], Optional[CompanionState]]] = None,
        companion_policy: Optional[DeterministicProactivePolicy] = None,
        calendar_source: Optional[CalendarSource] = None,
    ):
        """
        Args:
            profile: ユーザープロファイル
            check_interval: チェック間隔 (秒)
            monitor_context: MonitorContext (Phase 6) — PC状態チェック用
            conversation_enabled: 学習向けの会話開始を有効にする
            conversation_interval: 会話開始の最短間隔 (秒)
            conversation_idle: 最後のユーザー活動から発話までの待機時間 (秒)
            quiet_hours: 会話を開始しない時間帯 (開始時, 終了時)
            conversation_gate: 永続状態や日次上限による追加の送信判定
            companion_getter: 現在の CompanionState を返す (None なら従来挙動)
            companion_policy: 決定的 Proactive Policy (None ならデフォルト閾値)
            calendar_source: 次の予定を返す CalendarSource (None なら従来の
                profile ベース判定を維持)
        """
        self.profile = profile
        self.check_interval = check_interval
        self.monitor_context = monitor_context
        self.conversation_enabled = conversation_enabled
        self.conversation_idle = max(0.0, conversation_idle)
        self.quiet_hours = quiet_hours
        self.conversation_gate = conversation_gate
        self.companion_getter = companion_getter
        self.companion_policy = (
            companion_policy
            if companion_policy is not None
            else DeterministicProactivePolicy()
        )
        self.calendar_source = calendar_source
        # EventReminderEngine (src/tasks/event_reminder.py) が有効なとき True。
        # gcal: 由来の schedule エントリはそちらが通知するため、ここでは
        # スキップして二重通知を防ぐ。
        self.skip_gcal_schedule = False

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[str, str], None]] = None

        # 発火抑制: 同じトリガーが短時間で連続発火しないように
        self._last_fired: dict[str, float] = {}
        self._cooldown: dict[str, float] = {
            "schedule_remind": 1800,    # 30分
            "break_suggest": 3600,      # 1時間
            "greeting": 43200,          # 12時間
            "pc_alert": 600,            # 10分
            "conversation_start": max(60.0, conversation_interval),
        }

        # 作業開始時刻の追跡
        self._session_start_time: Optional[float] = None
        self._started_at: Optional[float] = None
        self._last_user_activity: float = time.time()
        self._conversation_prompt_index = datetime.now().toordinal() % len(CONVERSATION_PROMPTS)

    def start(self, callback: Callable[[str, str], None]) -> None:
        """
        プロアクティブエンジンを起動

        Args:
            callback: トリガー発火時に呼ばれる callback(trigger_type, message)
                trigger_type: "schedule_remind", "break_suggest", "greeting",
                    "pc_alert", "conversation_start"
                message: AIが発話すべきメッセージテキスト
        """
        self._callback = callback
        self._running = True
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._check_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """エンジンを停止"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    def notify_user_activity(self) -> None:
        """ユーザーの操作を通知し、連続作業セッションを更新する。

        30分以上操作が空いた後の最初の操作は新しいセッションとする。
        サービス起動時刻を作業開始としないことで、Bot の長時間稼働を
        「数十時間の連続作業」と誤認するのを防ぐ。
        """
        now = time.time()
        if (
            self._session_start_time is None
            or now - self._last_user_activity >= WORK_SESSION_GAP_SECONDS
        ):
            self._session_start_time = now
        self._last_user_activity = now

    def _can_fire(self, trigger_type: str) -> bool:
        """クールダウン中でなければTrue"""
        last = self._last_fired.get(trigger_type, 0)
        cooldown = self._cooldown.get(trigger_type, 600)
        return (time.time() - last) >= cooldown

    def _companion_gate(self) -> Optional[str]:
        """CompanionState に基づく介入ゲート。

        Returns:
            None: ゲートしない (従来挙動 or 発火許可)
            "focused": 集中中で割り込み不可 (黙る)
            "away": 離席中・不在 (黙る)

        companion_getter が None の場合は一切ゲートしない。
        getter が例外を投げても None を返し、従来どおり動く (best-effort)。
        """
        if self.companion_getter is None:
            return None
        try:
            state = self.companion_getter()
        except Exception:
            return None
        if state is None:
            return None
        if state.activity_mode == "focused" and not state.interruptible:
            return "focused"
        if state.activity_mode == "away" or not state.present:
            return "away"
        return None

    def _fire(self, trigger_type: str, message: str) -> None:
        """トリガーを発火"""
        if self._callback and self._can_fire(trigger_type):
            self._last_fired[trigger_type] = time.time()
            try:
                self._callback(trigger_type, message)
            except Exception:
                pass

    def _check_loop(self) -> None:
        """バックグラウンドチェックループ"""
        # 起動直後は少し待つ
        time.sleep(5.0)

        while self._running:
            try:
                self._check_schedule_remind()
                self._check_break_suggest()
                self._check_greeting()
                self._check_pc_alert()
                self._check_conversation_start()
            except Exception:
                pass

            time.sleep(self.check_interval)

    # --- トリガーチェック ---

    def _check_schedule_remind(self) -> None:
        """スケジュールリマインド: 予定の15分前に通知"""
        if self._companion_gate() == "away":
            return
        if not self._can_fire("schedule_remind"):
            return

        if self.calendar_source is not None:
            self._check_schedule_remind_calendar()
            return

        now = datetime.now()
        today_schedule = self.profile.get_today_schedule()

        for item in today_schedule:
            time_str = item.get("time", "")
            title = item.get("title", "")
            if not time_str or not title:
                continue
            if self.skip_gcal_schedule and str(item.get("note", "")).startswith("gcal:"):
                continue

            try:
                # HH:MM 形式をパース
                hour, minute = map(int, time_str.split(":"))
                event_time = now.replace(hour=hour, minute=minute, second=0)

                # 15分前〜5分前の範囲でリマインド
                diff_minutes = (event_time - now).total_seconds() / 60
                if 5 <= diff_minutes <= 15:
                    msg = f"あと{int(diff_minutes)}分で「{title}」の時間です。準備は大丈夫ですか？"
                    self._fire("schedule_remind", msg)
                    return
            except (ValueError, TypeError):
                pass

    def _check_schedule_remind_calendar(self) -> None:
        """CalendarSource ベースのスケジュールリマインド。"""
        now = time.time()
        try:
            next_event = self.calendar_source.next_event(now=now)
        except Exception:
            return
        if next_event.start_at is None:
            return
        delta = next_event.start_at - now
        if not (0 <= delta <= self.companion_policy.schedule_lead_seconds):
            return
        minutes = int(round(delta / 60))
        title = next_event.title or ""
        msg = f"あと{minutes}分で「{title}」の時間です。準備は大丈夫ですか？"
        self._fire("schedule_remind", msg)

    def _check_break_suggest(self) -> None:
        """長時間作業の休憩提案: 2時間連続で通知"""
        if self._companion_gate() is not None:
            return
        if not self._can_fire("break_suggest"):
            return

        if self._session_start_time is None:
            return

        elapsed_hours = (time.time() - self._session_start_time) / 3600
        activity_hours = (time.time() - self._last_user_activity) / 3600

        # セッション開始から2時間以上で、最近もアクティブ（30分以内に操作あり）
        if elapsed_hours >= 2.0 and activity_hours < 0.5:
            hours = int(elapsed_hours)
            msg = f"{hours}時間くらい連続で作業していますね。少し休憩しませんか？目や体を休めることも大切ですよ。"
            self._fire("break_suggest", msg)

    def _check_greeting(self) -> None:
        """時間帯の挨拶"""
        if self._companion_gate() is not None:
            return
        if not self._can_fire("greeting"):
            return

        now = datetime.now()
        hour = now.hour

        # エンジン開始直後（60秒以内）のみ挨拶
        if self._started_at is None:
            return
        if (time.time() - self._started_at) > 60:
            return

        # 時間帯に応じた挨拶
        name = self.profile.name
        name_part = f"、{name}さん" if name else ""

        if 5 <= hour < 10:
            msg = f"おはようございます{name_part}。今日も一日頑張りましょう。"
        elif 22 <= hour or hour < 5:
            msg = f"遅い時間ですね{name_part}。無理しないでくださいね。"
        else:
            return  # 日中は挨拶不要

        # 今日の予定があれば追加
        today = self.profile.get_today_schedule()
        if today:
            titles = [s.get("title", "") for s in today if s.get("title")]
            if titles:
                msg += f" 今日の予定は{len(titles)}件あります。"

        self._fire("greeting", msg)

    def _check_pc_alert(self) -> None:
        """PC異常アラート"""
        if self._companion_gate() is not None:
            return
        if not self._can_fire("pc_alert"):
            return

        if self.monitor_context is None:
            return

        status = self.monitor_context.get_status()
        if not status.get("running"):
            return

        alerts = []
        cpu = status.get("cpu_percent")
        mem = status.get("mem_percent")
        gpu_temp = status.get("gpu_temp_c")
        cpu_temp = status.get("cpu_temp_c")

        if cpu is not None and cpu > 95:
            alerts.append(f"CPU使用率が{cpu:.0f}%と非常に高いです")
        if mem is not None and mem > 95:
            alerts.append(f"メモリ使用率が{mem:.0f}%で逼迫しています")
        if gpu_temp is not None and gpu_temp > 90:
            alerts.append(f"GPU温度が{gpu_temp:.0f}°Cと高温です")
        if cpu_temp is not None and cpu_temp > 90:
            alerts.append(f"CPU温度が{cpu_temp:.0f}°Cと高温です")

        if alerts:
            msg = "PCの状態に注意が必要です。" + "、".join(alerts) + "。"
            self._fire("pc_alert", msg)

    def _check_conversation_start(self) -> None:
        """会話が途切れているとき、プロフィール学習につながる質問を一つ送る。"""
        if self._companion_gate() is not None:
            return
        if not self.conversation_enabled or not self._can_fire("conversation_start"):
            return
        now = datetime.now()
        if self._is_quiet_hour(now.hour):
            return
        if (time.time() - self._last_user_activity) < self.conversation_idle:
            return
        if self.conversation_gate is not None:
            try:
                if not self.conversation_gate():
                    return
            except Exception:
                # 状態ストア障害時は通知を増やさない側に倒す。
                return

        prompt = CONVERSATION_PROMPTS[
            self._conversation_prompt_index % len(CONVERSATION_PROMPTS)
        ]
        self._conversation_prompt_index += 1
        self._fire("conversation_start", prompt)

    def _is_quiet_hour(self, hour: int) -> bool:
        """日跨ぎにも対応した quiet hours 判定。start == end は無効扱い。"""
        start, end = self.quiet_hours
        start %= 24
        end %= 24
        if start == end:
            return False
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end

    def get_status(self) -> dict:
        """APIレスポンス用"""
        now = time.time()
        return {
            "running": self.is_running,
            "check_interval": self.check_interval,
            "session_duration_min": round((now - self._session_start_time) / 60, 1) if self._session_start_time else 0,
            "last_activity_sec_ago": round(now - self._last_user_activity, 0),
            "conversation_enabled": self.conversation_enabled,
            "conversation_idle_sec": self.conversation_idle,
            "quiet_hours": list(self.quiet_hours),
            "fired_triggers": {
                k: round(now - v, 0) for k, v in self._last_fired.items()
            },
        }

    def get_pending_triggers(self) -> list[dict]:
        """
        現在発火可能なトリガーを一括チェックして返す（ポーリング用）

        Returns:
            [{"type": "schedule_remind", "message": "..."}, ...]
        """
        triggers = []

        # スケジュール
        now = datetime.now()
        for item in self.profile.get_today_schedule():
            time_str = item.get("time", "")
            title = item.get("title", "")
            if not time_str or not title:
                continue
            try:
                hour, minute = map(int, time_str.split(":"))
                event_time = now.replace(hour=hour, minute=minute, second=0)
                diff_minutes = (event_time - now).total_seconds() / 60
                if 5 <= diff_minutes <= 15 and self._can_fire("schedule_remind"):
                    triggers.append({
                        "type": "schedule_remind",
                        "message": f"あと{int(diff_minutes)}分で「{title}」の時間です。",
                    })
            except (ValueError, TypeError):
                pass

        return triggers
