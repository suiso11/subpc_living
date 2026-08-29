# 優先順位オーケストレーション

> **状態**: active / supporting
> **位置付け**: タスク優先順位付けの設計文書
> **対象範囲**: TaskStore inbox からの理由付き採点、1件選定、見送り・手動上書き
> **作成日**: 2026-07-14
> **更新日**: 2026-08-28
> **日付根拠**: Git commit date

## 目的

入力を記憶するだけでなく、「このシステムを通せば次に何をするか決まる」状態を作る。
継続利用の動機は通知量ではなく、判断負荷が実際に減ることへ置く。

```text
Discord / 音声 / Web / Calendar
              ↓
         TaskStore (inbox)
              ↓
  PriorityController (理由付き採点)
              ↓
       今やる1件を固定
      ↙       ↓       ↘
   完了      見送り     手動上書き
    ↓         ↓          ↓
 次を選定   一時除外    指定を固定
```

## 決定規則

LLMの自由推論ではなく、再現可能なスコアを使う。

1. 期限超過を最上位に置く。超過タスクは過去の見送りで埋もれさせない。
2. 2時間以内、8時間以内、当日、3日以内、1週間以内の順に期限点を加える。
3. `high / normal / low` の明示優先度を加える。
4. 古い未処理タスクと `action_hint` がある実行可能なタスクを少し上げる。
5. `/focus next` で見送ったタスクは一定時間除外し、その後も見送り回数を弱い好みとして反映する。
6. 同点は期限、タスクIDの順で決める。

選定後はスコアを再計算しても自動で切り替えない。これは、優先順位を頻繁に変えて
着手を妨げる「再最適化」を防ぐためである。

## 継続利用の仕組み

- 1回の操作で1件だけ返し、選択肢過多を減らす。
- 完了すると次の1件を自動選定し、判断のループを閉じる。
- 今日の完了数、連続完了日、委任した決定数を手応えとして可視化する。
- 次の予定までの実作業枠を出し、「時間が足りないから始めない」を減らす。
- 見送り、手動指定、機能無効化を常に残す。罪悪感を煽る文面や連続記録の喪失警告は使わない。

## 永続化と境界

`data/tasks/priority_state.json` は atomic replace で更新する。タスク本文や会話内容は保存せず、
SQLite側のタスクIDを参照する。Web・音声・Discordの各チャットはこの共有状態を読み、
同じ現在フォーカスをシステムプロンプトへ注入する。書き込み操作は現在Discordの
`/focus` に集約し、複数プロセスからの競合を避ける。

## 実装状況（リポジトリ基準・6段階）

`integrated` は「通常の入口経路へ配線済み」を意味する。`deployed` / `verified` は本リポジトリでは未実施。

| 項目 | planned | implemented | tested | integrated | deployed | verified | 根拠（主要） |
|---|---|---|---|---|---|---|---|
| TaskStore（inbox / 候補 / 完了・見送り・永続化） | - | ○ | ○ | ○（Web / Discord / VoicePipeline） | - | - | `src/tasks/store.py`, `tests/test_tasks_store.py` |
| PriorityController（理由付き採点・1件選定・見送り・手動上書き） | - | ○ | ○ | △（Discord `/focus` のみ） | - | - | `src/tasks/prioritizer.py`, `tests/test_prioritizer.py` |
| Web 経路 | - | ○ | ○ | ○（TaskStore注入・Context注入。`/focus` 等の採点UIは無し） | - | - | `src/web/server.py` |
| Discord 経路 | - | ○ | ○ | ○（TaskStore + PriorityController + `/focus` 一連コマンド） | - | - | `src/discord_bot/bot.py` |
| VoicePipeline 経路 | - | ○ | ○ | △（pipeline 内部で TaskStore を自前生成） | - | - | `src/audio/pipeline.py` |

凡例: ○=該当・成立、△=一部/未完成、-=該当なし・未実施。

主なギャップ:
- **`src/audio/main.py` は TaskStore を注入しない**。`run_voice_mode` は `VoicePipeline` へ
  `task_store` を渡しておらず、pipeline が内部で自前生成した TaskStore しか使えない
  （`src/audio/pipeline.py` の初期化処理）。したがって Web / Discord / 音声で同じフォーカスを
  共有する「全入口共通の現在フォーカス」は未完成。
- Web には TaskStore の注入と Tasks Context 注入はあるが、`PriorityController`（採点・1件選定・
  見送り・手動上書き）の経路は無い。
- 採点・選定ロジックは実装・テスト済みだが、ライブ運用でのデプロイ・検証は未実施。
