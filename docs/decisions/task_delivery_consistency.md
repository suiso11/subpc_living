# タスク配送一貫性の決定記録

> **状態**: active / canonical
> **位置付け**: リマインド通知とタスク⇔カレンダー同期の配送一貫性契約の決定記録
> **対象範囲**: tasks.rev 楽観的バージョン・BEGIN IMMEDIATE claim・lease owner /
>   revalidate-before-callback・expected_rev 条件付き record・state-driven
>   カレンダー同期・マーカー照合/重複整理・at-least-once best-effort 契約
> **作成日**: 2026-08-29
> **更新日**: 2026-08-29
> **日付根拠**: Git commit date

## 背景

タスクのリマインド (期限エスカレーション) は Discord bot プロセス内の
`TaskReminderEngine` (src/tasks/reminder.py) が、カレンダー同期は
`TaskCalendarSync` / `CalendarPullWorker` (src/tasks/calendar_sync.py) が担う。
`TaskStore` (src/tasks/store.py) は SQLite WAL で複数プロセスからの同時アクセスを
想定する。通知は「二重送信しない」ことと「必ず届ける」ことが両立しきれないため、
どこまでを保証しどこを best-effort と割り切るかを明示する必要がある。

## 経緯（時系列）

| 日付 | 区分 | 内容 |
| --- | --- | --- |
| 2026-08-29 | 議論 | リマインド・カレンダー同期の配送一貫性を「at-least-once best-effort」で統一する方針を検討 |
| 2026-08-29 | 決定 | 本記録の決定を採択（rev 楽観制御・BEGIN IMMEDIATE claim・revalidate-before-callback・expected_rev 条件付き record・state-driven 同期・マーカー照合/重複整理を契約として固定） |
| 2026-08-29 | 実装 | 契約は既存実装と整合する（reminder / store / calendar_sync は変更せず）。オフライン unit テストで固定（live / deployed / verified は主張しない） |

## 現時点の決定

1. **配送契約は at-least-once の best-effort であり、exactly-once を保証しない**:
   通知は「二重送信があり得るが、1件も取りこぼさないことを目指す」契約。
   厳密な一度きりは対象外と明示する。
2. **`tasks.rev` は楽観的並行制御のリビジョンである**:
   リマインドを無効化しうる変更（`update` / `done` / `drop` / `snooze` /
   `regenerate_breakdown`）は同一トランザクションで `rev = rev + 1` する。
   冪等マイグレーションで旧 DB にも追加される。
3. **通知の claim は `BEGIN IMMEDIATE` で行う**:
   `claim_due_notifications` は書込ロックを先取りして lease
   （`lease_owner` / `lease_until`）を取り、対象行を返す。due あり open タスクのうち
   next_notify_at が即評価または期限到達・snooze 済みでない・lease が空/期限切れ/
   自分のもののみが対象。他 owner の未満了 lease は横取りしない。
4. **外部コールバックの直前に lease と rev を再検証する** (revalidate-before-callback):
   `revalidate_notification_lease` が「open かつ rev 一致かつ lease_owner が自分の
   ときだけ」lease を延長して True を返す。それ以外は stale lease のみ解放して
   False（発火スキップ）。
5. **`record_notification` は `expected_rev` 条件付きで更新する**:
   claim 時に得た rev を `expected_rev` として渡し、タスクが open かつ rev 一致
   かつ lease_owner が自分のときだけ通知状態を書き込む。不一致なら何も上書きせず
   自 owner の stale lease だけを解放して False（並行の done/drop/update/snooze を
   上書きしない）。`expected_rev` 省略時は旧挙動の無条件更新（後方互換）。
6. **カレンダー同期は state-driven である**:
   `TaskCalendarSync._sync_task` は queue イベントのラベル (add/update/done) を
   信頼せず、常に先に現在のタスク状態（open/done/dropped、due 有無、
   `calendar_event_id`）を読んで行動を決める。dropped はイベント削除、done は
   完了サマリ更新、open は due 有無に応じて作成/更新/削除。ハード削除済みタスク
   の残存イベントは pull 側のマーカー整理が掃除する。
7. **マーカー照合と重複整理が crash / queue-drop の回復経路である**:
   `CalendarPullWorker._reconcile_markers` は `subpc-task:{id}` マーカー付きイベントを
   タスク id でグループ化し、タスク不存在/dropped は削除、done/open は正準
   マーカーを1件残して対応付けを復元し、重複マーカーは決定的に1件へ収束する
   （`_canonical_marker` は対応付け済み id 優先、無ければ `(start, event_id)` の
   辞書順最小）。イベント本文はログしない。
8. **同時に動くリマインドエンジンの owner 名は一意でなければならない**:
   lease は owner 名で同一判定されるため、owner 名が衝突すると互いの lease を
   自分のものと誤認して二重送信し得る。現実装の owner は `"discord"` のみで、
   新規エンジンを同じ DB で並走させる場合は必ず別名を割り当てる。
9. **外部カレンダー呼び出しはトランザクションにできない**:
   DB コミットと Google Calendar API の結果は原子的に結ばない。DB は先行コミット、
   カレンダーは数秒かかる npx MCP 呼び出しで、失敗時はリトライまたは pull 回復に
   委ねる。カレンダー障害でタスク操作を失敗させない（enqueue 非ブロッキング、
   ワーカー内例外は握り潰してリトライ）。

## 判断理由

- **1（at-least-once）**: 「必ず届ける」と「必ず一度きり」はクラッシュ点の前後
  （後述の残課題）で両立しない。配送保証を at-least-once に割り切ることで、
  発火側の実装は「無効化を上書きしない」一点に集中できる。
- **2（rev）**: done/drop/update/snooze と通知評価は別トランザクションで走る。
  rev が無ければ、claim 後に完了されたタスクへ「発火済み」を書き戻して
  再通知を永久に殺してしまう。rev 比較でこの競合を検出する。
- **3（BEGIN IMMEDIATE claim）**: 複数プロセス（Discord bot / Web / 音声）が同じ
  WAL DB を読むため、評価対象の選択と lease 付与を書込ロック下で原子的に行う。
- **4・5（revalidate-before-callback / expected_rev）**: コールバック（通知送信）と
  record（発火済み記録）の間でユーザーが完了・延期した場合に、record 側が
  古い情報で next_notify_at を上書きして再通知を無効化する事故を防ぐ。
- **6（state-driven）**: queue イベントは enqueue 時点の状態であり、処理時に
  古い可能性がある。現在状態を正とすることで stale イベントが壊れた外部状態を
  作らない。
- **7（マーカー照合/重複整理）**: queue-full で drop された同期や、イベント作成
  後に id を取得できなかったケース（レスポンス形式差）は、write 経路では回復
  できない。pull が範囲内のマーカーイベントを再照合して対応付けを復元する。
  重複マーカーは決定的ルールで1件へ収束させ、どのプロセスが掃除しても結果が
  揺れないようにする。
- **8（owner 一意）**: owner は「自分の lease」を判定する唯一の識別子。衝突は
  lease 排他性の破壊に直結するため、呼び出し側の規律として必須にする。
- **9（非トランザクション外部呼び出し）**: 外部 API を DB トランザクションで包む
  と数秒のロック保持と部分失敗のロールバック不能という2重の問題になる。
  二段階（DB 先行・外部後追い）に割り切る。

## 前提（assumptions）

| ID | 前提 | 推奨既定 |
| --- | --- | --- |
| A1 | 単一 `TaskStore` が複数プロセスから共有される | WAL + `synchronous=NORMAL` + `busy_timeout`、書込は `BEGIN IMMEDIATE` |
| A2 | 同時に動くリマインドエンジンの owner 名は一意 | 現状 `"discord"`。追加エンジンは別名必須（決定8） |
| A3 | 外部カレンダーは非トランザクション API（Google Calendar / npx MCP） | DB と原子的にしない。リトライ＋pull 回復に委ねる |
| A4 | pull は `past_days` / `days_ahead` の取得範囲内のイベントのみ照合する | 既定 `past_days=14` / `days_ahead=45`。範囲外は回復対象外 |
| A5 | 配送コールバックはベストエフォート | 例外は記録して握り潰し、タスク操作を失敗させない |

## 解決済み（resolved）

| ID | 事項 | 実装 |
| --- | --- | --- |
| U1 | rev の増加対象と冪等マイグレーション | `update` / `done` / `drop` / `snooze` / `regenerate_breakdown` で `rev = rev + 1`。旧 DB へは冪等 `ALTER`（`test_rev_migration_idempotent`） |
| U2 | claim の排他性（多重送信防止） | `BEGIN IMMEDIATE` で lease 付与。他 owner の未満了 lease は対象外（`test_claim_lease_is_exclusive` / `test_claim_skips_tasks_without_due`） |
| U3 | コールバック直前の再検証 | `revalidate_notification_lease` は owner + rev + open を確認して延長/解放（`test_revalidate_lease_valid_and_invalid` / `_requires_owner` / `_requires_open` / `_unknown_task`） |
| U4 | record の条件付き更新 | `expected_rev` 不一致・done/dropped 時は書き込まず stale lease だけ解放（`test_record_notification_expected_rev_stale_rejected` / `test_record_notification_stale_rejects_when_closed`） |
| U5 | 並行 done/update のスキップ | リマインドエンジンは再検証失敗時に発火せず record もされない（`test_concurrent_done_before_callback_skips` / `test_concurrent_update_before_callback_skips` / `test_concurrent_done_via_second_connection`） |
| U6 | state-driven 同期とマーカー回復 | `_sync_task` は現在状態を最優先、`_reconcile_markers` は正準1件へ収束して対応付け復元（`test_reconcile_backfills_missing_mapping` / `test_marker_events_excluded_from_upcoming` / `test_reconcile_error_does_not_abort_other_markers`） |
| U7 | 別接続からの変更との競合 | `expected_rev` 付き record を別 sqlite 接続で検証（`test_revalidate_and_record_cross_connection`） |

## 未解決（unresolved）と推奨

| ID | 未解決事項 | 推奨既定（デフォルト明示） |
| --- | --- | --- |
| U8 | revalidate とコールバック実行の間の micro-TOCTOU | 原理的に残る。この窓を縮小（revalidate 直後に発火）するのが限界で、無くすことは保証対象外。二重送信は受け入れ、頻度は低い |
| U9 | 実機・実サービス検証（実 Discord 常駐・実 Google Calendar・実複数プロセス同時起動） | 未実施。本記録はリポジトリ実装＋オフライン分離テストのみで、deployed / verified は主張しない |
| U10 | owner 名の中央管理・起動時衝突検出 | 現状は呼び出し側の規律（決定8）に依存。要望が出てから、エンジン起動時に `PRAGMA` 等で owner 重複を検出する仕組みを再検討 |
| U11 | durable outbox の有無 | 非目標（下記）。今後「確実に届ける」を強めたい場合は、enqueue を drop せず永続化する durable outbox を別 ADR で検討する |

## 非目標（non-goals）

- **durable outbox は持たない**: `TaskCalendarSync.enqueue` は非同期キューで、
  `queue.Full` 時は drop する。永続化・再送キューは対象外。
- **DB トランザクションで外部コールバックを包まない**: 外部カレンダー呼び出しを
  `BEGIN`〜`COMMIT` に含めない（決定9）。ロックの長時間保持と部分失敗を防ぐ。
- **グローバルな exactly-once 配送はしない**: 決定1の通り。再送は起こり得る。
- **外部マーカー検索はしない**: カレンダー全体から `subpc-task:` マーカーを検索
  して同期する方式は採らない。照合は `CalendarPullWorker` の pull 範囲内の
  `list_events_range` 結果のみ（前提 A4）。

## 検証・確認

- `tests/test_tasks_store.py`（69件）、`tests/test_tasks_reminder.py`（19件）、
  `tests/test_calendar_sync.py`（32件）を Windows 開発機で
  `python -m unittest discover -s tests -p <ファイル> -t . -q` で実行し全て成功
  （オフライン・一時 DB / fake client のみ。実サービス・実プロセス多重起動・
  実 Google Calendar は使っていない）。
- 確認コマンドは AGENTS.md の「Windows 開発機は変更範囲だけ回す」に従い、
  本記録の対象範囲のみを実行した。
- 本記録は live / deployed / verified を主張しない（U9）。
- 実装状態は [implementation_status.md](../implementation_status.md) を正とする。

## 関連文書

- [src/tasks/store.py](../../src/tasks/store.py): `rev` / `claim_due_notifications` /
  `revalidate_notification_lease` / `record_notification`
- [src/tasks/reminder.py](../../src/tasks/reminder.py): `TaskReminderEngine` の
  revalidate-before-callback と `expected_rev` 付き record
- [src/tasks/calendar_sync.py](../../src/tasks/calendar_sync.py): state-driven
  `_sync_task` / `_reconcile_markers` / 正準マーカー収束
- [src/tasks/event_reminder.py](../../src/tasks/event_reminder.py): 予定リマインドの
  `INSERT OR IGNORE` claim（重複防止の類似パターン）
- [tests/test_tasks_store.py](../../tests/test_tasks_store.py) /
  [tests/test_tasks_reminder.py](../../tests/test_tasks_reminder.py) /
  [tests/test_calendar_sync.py](../../tests/test_calendar_sync.py): オフライン検証