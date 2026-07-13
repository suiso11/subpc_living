"""Qt/QML bridge for the native desktop client."""
from __future__ import annotations

import base64
import json
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Property, QRunnable, QThreadPool, QTimer, QUrl, Signal, Slot
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtNetwork import QAbstractSocket
from PySide6.QtWebSockets import QWebSocket

from .api import DesktopApi
from .audio import NativeAudioRecorder
from .config import DesktopSettings
from .windows import is_autostart_enabled, set_autostart


class _WorkerSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


class _Worker(QRunnable):
    def __init__(self, call: Callable[[], Any]) -> None:
        super().__init__()
        self.call = call
        self.signals = _WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.signals.result.emit(self.call())
        except Exception as exc:  # boundary: message is surfaced in the native UI
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


class DesktopBridge(QObject):
    tasksChanged = Signal()
    messagesChanged = Signal()
    gameChanged = Signal()
    historiesChanged = Signal()
    historyMessagesChanged = Signal()
    logsChanged = Signal()
    statusChanged = Signal()
    serverUrlChanged = Signal()
    loadingChanged = Signal()
    recordingChanged = Signal()
    autostartChanged = Signal()
    ttsEnabledChanged = Signal()
    toast = Signal(str, str)
    nativeNotification = Signal(str, str)

    def __init__(
        self,
        settings: DesktopSettings,
        parent: QObject | None = None,
        *,
        settings_path: str | Path | None = None,
        offline: bool = False,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.settings_path = settings_path
        self.offline = offline
        self.api = DesktopApi(settings.server_url)
        self.pool = QThreadPool.globalInstance()
        self._workers: set[_Worker] = set()
        self._loading_count = 0
        self._tasks: list[dict[str, Any]] = []
        self._messages: list[dict[str, Any]] = []
        self._game: dict[str, Any] = {}
        self._histories: list[dict[str, Any]] = []
        self._history_messages: list[dict[str, Any]] = []
        self._logs = ""
        self._status_text = "接続を確認中"
        self._connected = False
        self._recording = False
        self._streaming = False
        self._stream_text = ""
        self._pending_chat: list[dict[str, Any]] = []
        self._session_id = settings.session_id
        self._chat_ready = False
        self._shutting_down = False
        self._notified_due: set[str] = set()
        self.recorder = NativeAudioRecorder()

        self.reminder_timer = QTimer(self)
        self.reminder_timer.setInterval(60_000)
        self.reminder_timer.timeout.connect(self.loadTasks)

        self.socket = QWebSocket("SUBPC BUDDY")
        self.socket.connected.connect(self._socket_connected)
        self.socket.disconnected.connect(self._socket_disconnected)
        self.socket.textMessageReceived.connect(self._socket_message)
        self.socket.errorOccurred.connect(self._socket_error)
        self.reconnect_timer = QTimer(self)
        self.reconnect_timer.setSingleShot(True)
        self.reconnect_timer.setInterval(5_000)
        self.reconnect_timer.timeout.connect(self._connect_socket)

        self.audio_output = QAudioOutput(self)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)
        self._audio_temp: Path | None = None
        self.player.mediaStatusChanged.connect(self._media_status_changed)

    @Property("QVariantList", notify=tasksChanged)
    def tasks(self) -> list[dict[str, Any]]:
        return self._tasks

    @Property("QVariantList", notify=messagesChanged)
    def messages(self) -> list[dict[str, Any]]:
        return self._messages

    @Property("QVariantMap", notify=gameChanged)
    def game(self) -> dict[str, Any]:
        return self._game

    @Property("QVariantList", notify=historiesChanged)
    def histories(self) -> list[dict[str, Any]]:
        return self._histories

    @Property("QVariantList", notify=historyMessagesChanged)
    def historyMessages(self) -> list[dict[str, Any]]:
        return self._history_messages

    @Property(str, notify=logsChanged)
    def logs(self) -> str:
        return self._logs

    @Property(str, notify=statusChanged)
    def statusText(self) -> str:
        return self._status_text

    @Property(bool, notify=statusChanged)
    def connected(self) -> bool:
        return self._connected

    @Property(str, notify=serverUrlChanged)
    def serverUrl(self) -> str:
        return self.api.server_url

    @Property(bool, notify=loadingChanged)
    def loading(self) -> bool:
        return self._loading_count > 0

    @Property(bool, notify=recordingChanged)
    def recording(self) -> bool:
        return self._recording

    @Property(bool, notify=autostartChanged)
    def autostartEnabled(self) -> bool:
        return is_autostart_enabled()

    @Property(bool, notify=ttsEnabledChanged)
    def ttsEnabled(self) -> bool:
        return self.settings.tts_enabled

    @Slot()
    def initialize(self) -> None:
        if self.offline:
            self._connected = True
            self._status_text = "ネイティブUI起動検査"
            self._game = {
                "rank": {"name": "相棒"},
                "points": 0,
                "badges": [],
                "missions": [],
            }
            self.statusChanged.emit()
            self.gameChanged.emit()
            return
        self.refreshStatus()
        self.resumeChat()
        self.loadTasks()
        self.loadGame()
        self.reminder_timer.start()

    def _run(self, call: Callable[[], Any], success: Callable[[Any], None]) -> None:
        worker = _Worker(call)
        self._workers.add(worker)
        self._loading_count += 1
        self.loadingChanged.emit()
        worker.signals.result.connect(success)
        worker.signals.error.connect(self._request_error)

        def finished() -> None:
            self._workers.discard(worker)
            self._loading_count = max(0, self._loading_count - 1)
            self.loadingChanged.emit()

        worker.signals.finished.connect(finished)
        self.pool.start(worker)

    @Slot(str)
    def setServerUrl(self, value: str) -> None:
        self.socket.abort()
        self.api.set_server_url(value)
        self.settings.server_url = self.api.server_url
        self._save_settings()
        self.serverUrlChanged.emit()
        self._session_id = ""
        self._chat_ready = False
        self._pending_chat = []
        self._messages = []
        self.messagesChanged.emit()
        self.settings.session_id = ""
        self._save_settings()
        self.initialize()
        self.toast.emit("接続先を変更しました", self.api.server_url)

    @Slot(bool)
    def setTtsEnabled(self, enabled: bool) -> None:
        if self.settings.tts_enabled == enabled:
            return
        self.settings.tts_enabled = enabled
        self._save_settings()
        self.ttsEnabledChanged.emit()

    @Slot(bool)
    def setAutostart(self, enabled: bool) -> None:
        try:
            if not set_autostart(enabled):
                raise RuntimeError("Windowsでのみ設定できます")
        except Exception as exc:
            self.toast.emit("自動起動を変更できません", str(exc))
        self.autostartChanged.emit()

    @Slot(str, str)
    def notify(self, title: str, message: str) -> None:
        self.nativeNotification.emit(title, message)

    @Slot()
    def refreshStatus(self) -> None:
        self._run(self.api.status, self._status_loaded)

    def _status_loaded(self, data: dict[str, Any]) -> None:
        self._connected = True
        model = data.get("model") or "モデル未設定"
        self._status_text = f"接続済み · {model}"
        self.statusChanged.emit()

    @Slot()
    def loadTasks(self) -> None:
        if self.offline:
            return
        self._run(self.api.tasks, self._tasks_loaded)

    def _tasks_loaded(self, data: dict[str, Any]) -> None:
        self._tasks = list(data.get("tasks") or [])
        self.tasksChanged.emit()
        self._notify_due_tasks()

    def _notify_due_tasks(self) -> None:
        now = datetime.now(timezone.utc)
        for task in self._tasks:
            value = task.get("due_at")
            if not value:
                continue
            try:
                due = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                if due.tzinfo is None:
                    due = due.replace(tzinfo=timezone.utc)
                seconds = (due.astimezone(timezone.utc) - now).total_seconds()
            except (TypeError, ValueError):
                continue
            key = f"{task.get('id')}:{value}"
            if key in self._notified_due or not (-300 <= seconds <= 900):
                continue
            self._notified_due.add(key)
            title = str(task.get("title") or "タスク")
            first = str(task.get("action_hint") or "").strip()
            self.nativeNotification.emit(
                "まもなく期限です" if seconds >= 0 else "期限を迎えました",
                title + (f"\n最初の一歩: {first}" if first else ""),
            )

    @Slot(str, str, str)
    def addTask(self, text: str, priority: str = "normal", note: str = "") -> None:
        text = text.strip()
        if not text:
            self.toast.emit("入力が空です", "やることを1行で入力してください")
            return
        self._run(lambda: self.api.add_task(text, priority, note), self._task_changed)

    @Slot(int)
    def completeTask(self, task_id: int) -> None:
        self._run(lambda: self.api.complete_task(task_id), self._task_completed)

    @Slot(int)
    def dropTask(self, task_id: int) -> None:
        self._run(lambda: self.api.drop_task(task_id), self._task_changed)

    @Slot(int)
    def regenerateTask(self, task_id: int) -> None:
        self._run(lambda: self.api.regenerate_task(task_id), self._task_changed)

    @Slot(int, str, str, str, str, str)
    def updateTask(
        self,
        task_id: int,
        title: str,
        due: str,
        priority: str,
        note: str,
        action_hint: str,
    ) -> None:
        fields: dict[str, Any] = {
            "title": title.strip(),
            "due": due.strip(),
            "priority": priority,
            "note": note.strip(),
            "action_hint": action_hint.strip(),
        }
        self._run(lambda: self.api.update_task(task_id, fields), self._task_changed)

    def _task_completed(self, _: dict[str, Any]) -> None:
        self.toast.emit("完了しました", "積み重ねを実績へ記録しました")
        self.loadTasks()
        self.loadGame()

    def _task_changed(self, _: dict[str, Any]) -> None:
        self.loadTasks()

    @Slot()
    def loadGame(self) -> None:
        if self.offline:
            return
        self._run(self.api.game, self._game_loaded)

    def _game_loaded(self, data: dict[str, Any]) -> None:
        self._game = data
        self.gameChanged.emit()

    @Slot(str)
    def claimMission(self, mission_id: str) -> None:
        self._run(lambda: self.api.claim_mission(mission_id), self._mission_claimed)

    def _mission_claimed(self, data: dict[str, Any]) -> None:
        reward = int(data.get("reward") or 0)
        self.toast.emit("報酬を受け取りました", f"+{reward} GP")
        self.loadGame()

    @Slot(str, int)
    def loadLogs(self, unit: str = "subpc-web", lines: int = 200) -> None:
        self._run(lambda: self.api.journal(unit, lines), self._logs_loaded)

    def _logs_loaded(self, data: dict[str, Any]) -> None:
        self._logs = "\n".join(data.get("lines") or []) or "(ログなし)"
        self.logsChanged.emit()

    @Slot()
    def loadHistories(self) -> None:
        if self.offline:
            return
        self._run(self.api.histories, self._histories_loaded)

    def _histories_loaded(self, data: dict[str, Any]) -> None:
        self._histories = list(data.get("sessions") or [])
        self.historiesChanged.emit()

    @Slot(str)
    def loadHistory(self, filename: str) -> None:
        self._run(lambda: self.api.history(filename), self._history_loaded)

    def _history_loaded(self, data: dict[str, Any]) -> None:
        self._history_messages = [
            item for item in data.get("messages") or []
            if item.get("role") in ("user", "assistant")
        ]
        self.historyMessagesChanged.emit()

    @Slot(str)
    def deleteHistory(self, filename: str) -> None:
        self._run(lambda: self.api.delete_history(filename), lambda _: self.loadHistories())

    @Slot()
    def resumeChat(self) -> None:
        session = self._session_id or None
        self._run(lambda: self.api.resume_chat(session), self._chat_resumed)

    def _chat_resumed(self, data: dict[str, Any]) -> None:
        self._session_id = str(data.get("session_id") or "")
        self._chat_ready = True
        self.settings.session_id = self._session_id
        self._save_settings()
        self._messages = list(data.get("messages") or [])
        self.messagesChanged.emit()
        self._connect_socket()

    def _connect_socket(self) -> None:
        if self._shutting_down or not self._chat_ready:
            return
        if self.socket.state() in (
            QAbstractSocket.SocketState.ConnectingState,
            QAbstractSocket.SocketState.ConnectedState,
        ):
            return
        self.socket.open(QUrl(self.api.websocket_url))

    @Slot(str)
    def sendMessage(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self._messages = [*self._messages, {"role": "user", "content": text}]
        self.messagesChanged.emit()
        payload = {
            "type": "message",
            "text": text,
            "session_id": self._session_id or "desktop",
            "tts": self.settings.tts_enabled,
        }
        self._queue_or_send(payload)

    def _queue_or_send(self, payload: dict[str, Any]) -> None:
        if not self._chat_ready:
            self._pending_chat.append(payload)
            return
        payload["session_id"] = self._session_id or "desktop"
        if self.socket.state() == QAbstractSocket.SocketState.ConnectedState:
            self.socket.sendTextMessage(json.dumps(payload, ensure_ascii=False))
            return
        self._pending_chat.append(payload)
        self._connect_socket()

    def _socket_connected(self) -> None:
        self._connected = True
        self._status_text = "会話サーバーに接続済み"
        self.statusChanged.emit()
        pending, self._pending_chat = self._pending_chat, []
        for payload in pending:
            payload["session_id"] = self._session_id or "desktop"
            self.socket.sendTextMessage(json.dumps(payload, ensure_ascii=False))

    def _socket_disconnected(self) -> None:
        self._connected = False
        self._status_text = "再接続待ち"
        self.statusChanged.emit()
        if not self._shutting_down:
            self.reconnect_timer.start()

    def _socket_error(self, _error: Any) -> None:
        self._connected = False
        self._status_text = self.socket.errorString() or "接続エラー"
        self.statusChanged.emit()
        if not self._shutting_down and not self.reconnect_timer.isActive():
            self.reconnect_timer.start()

    def _socket_message(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        kind = data.get("type")
        if kind == "token":
            self._append_token(str(data.get("content") or ""))
        elif kind == "done":
            self._finish_stream(str(data.get("full_text") or self._stream_text))
        elif kind == "stt_result":
            text = str(data.get("text") or "").strip()
            if text:
                self._messages = [*self._messages, {"role": "user", "content": text}]
                self.messagesChanged.emit()
        elif kind == "audio":
            self._play_audio(str(data.get("data") or ""))
        elif kind == "error":
            self._streaming = False
            self.toast.emit("会話エラー", str(data.get("message") or "不明なエラー"))

    def _append_token(self, token: str) -> None:
        if not self._streaming:
            self._streaming = True
            self._stream_text = ""
            self._messages = [*self._messages, {"role": "assistant", "content": ""}]
        self._stream_text += token
        updated = list(self._messages)
        updated[-1] = {"role": "assistant", "content": self._stream_text}
        self._messages = updated
        self.messagesChanged.emit()

    def _finish_stream(self, full_text: str) -> None:
        if self._streaming and self._messages:
            updated = list(self._messages)
            updated[-1] = {"role": "assistant", "content": full_text}
            self._messages = updated
        elif full_text:
            self._messages = [*self._messages, {"role": "assistant", "content": full_text}]
        self._streaming = False
        self._stream_text = ""
        self.messagesChanged.emit()
        self.loadGame()

    @Slot()
    def startRecording(self) -> None:
        try:
            self.recorder.start()
        except Exception as exc:
            self.toast.emit("マイクを開始できません", str(exc))
            return
        self._recording = True
        self.recordingChanged.emit()

    @Slot()
    def stopRecording(self) -> None:
        try:
            wav = self.recorder.stop()
        except Exception as exc:
            self.toast.emit("録音を処理できません", str(exc))
            wav = b""
        self._recording = False
        self.recordingChanged.emit()
        if not wav:
            return
        self._queue_or_send({
            "type": "audio_message",
            "data": base64.b64encode(wav).decode("ascii"),
            "format": "wav",
            "session_id": self._session_id or "desktop",
            "tts": self.settings.tts_enabled,
        })

    def _play_audio(self, encoded: str) -> None:
        try:
            audio = base64.b64decode(encoded, validate=True)
            handle = tempfile.NamedTemporaryFile(prefix="subpc-tts-", suffix=".wav", delete=False)
            handle.write(audio)
            handle.close()
        except Exception:
            return
        self._remove_audio_temp()
        self._audio_temp = Path(handle.name)
        self.player.setSource(QUrl.fromLocalFile(handle.name))
        self.player.play()

    def _media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if status in (QMediaPlayer.MediaStatus.EndOfMedia, QMediaPlayer.MediaStatus.InvalidMedia):
            self._remove_audio_temp()

    def _remove_audio_temp(self) -> None:
        if self._audio_temp is None:
            return
        # Windows keeps the media file locked until the source is released.
        self.player.stop()
        self.player.setSource(QUrl())
        try:
            self._audio_temp.unlink(missing_ok=True)
        except OSError:
            pass
        self._audio_temp = None

    def _request_error(self, message: str) -> None:
        self._connected = False
        self._status_text = "バックエンドに接続できません"
        self.statusChanged.emit()
        self.toast.emit("読み込みに失敗しました", message)

    def _save_settings(self) -> None:
        try:
            self.settings.save(self.settings_path)
        except OSError as exc:
            self.toast.emit("設定を保存できません", str(exc))

    def shutdown(self) -> None:
        self._shutting_down = True
        self.reminder_timer.stop()
        self.reconnect_timer.stop()
        if self.recorder.recording:
            self.recorder.stop()
        self.socket.close()
        self.api.close()
        self._remove_audio_temp()
