# ローカル推論 backend の決定記録

> **状態**: active / canonical
> **位置付け**: ローカル推論 backend（Ollama 並行の OpenAI-compatible ローカルサーバー）の決定記録
> **対象範囲**: ローカル LLM 推論 backend の選択・登録・認証・性能主張・ScreenDescriber の扱い
> **作成日**: 2026-08-28
> **更新日**: 2026-08-28
> **日付根拠**: Git commit date

## 背景

現状のローカル推論は Ollama 単一（`src/llm/providers/ollama.py` の `OllamaProvider`）。
llama.cpp 系サーバー・LM Studio・vLLM など OpenAI-compatible なローカルサーバーを
ハードウェア・運用制約に応じて使いたい要求がある。既存の
`src/llm/providers/cloud_http.py` の `OpenAICompatibleProvider` は**クラウド経路（Phase K）前提**で、
`api_key` 必須・承認・redaction セマンティクスを伴うため、ローカル用途にそのまま流用できない。

## 経緯（時系列）

| 日付 | 区分 | 内容 |
| --- | --- | --- |
| 2026-08-28 | 議論 | Ollama 以外のローカルサーバー（llama.cpp / LM Studio / vLLM）への対応方針を検討 |
| 2026-08-28 | 決定 | 本記録の決定を採択（Ollama 既定維持＋keyless ローカル provider 追加） |
| 2026-08-28 | 修正 | P0-2 review 反映: openai_compatible の `local_base_url` を loopback 限定の厳格検証にし、U1 / U3 / U5 を実装済みとして確定。既定 URL を `http://localhost:8080/v1`、provider_id を `local-openai` に統一 |

## 現時点の決定

1. **汎用の keyless 対応ローカル provider を追加する**: `LocalOpenAICompatibleProvider` を新設し、
   llama.cpp / LM Studio / vLLM を Ollama と**並行**して利用可能にする。
2. **Ollama は後方互換の既定として維持する**: 無設定時の挙動は変更しない。
3. **ローカル provider は `local=True` で登録され、cloud の承認・redaction セマンティクスを受けない**:
   CloudRouteBridge / 承認フローを通さず、context は local target のみ。
4. **key が無い場合は Authorization ヘッダを送らない**: 認証不要のローカルサーバーへそのまま接続できる。
5. **Ollama 固有のマルチモーダル ScreenDescriber の移行は見送る**:
   `src/screen/describer.py` は当面 Ollama `/api/chat`（画像付き）のまま。
6. **性能主張はハードウェアベンチマークなしでは行わない**:
   実機 GPU・実測の tokens/sec・レイテンシ・VRAM を示せない限り、文書・会話で性能比較を主張しない。
7. **openai_compatible の `local_base_url` は本マイルストーンでは loopback に限定する**:
   scheme は `http`/`https` のみ、host 必須、URL userinfo 不可、host は `localhost` または
   `ipaddress.is_loopback` が真の IP（IPv4/IPv6）のみ。任意の LAN / 公開 / 曖昧な host 名は拒否する。
   リモートの信頼済みノード対応は deferred とし、別途の明示的な信頼設計（trust design）を要求する。

## 判断理由

- **1・2（並行導入・Ollama既定）**: ローカルサーバーは構成差が大きく「一択」が定まらない。
  backend を差し替え可能にしつつ既定を維持すれば、既存の全入口（CLI / Web / Discord / Voice）と
  テストを壊さずに移行判断できる。
- **3（local=True・cloud セマンティクス除外）**: ローカル送信はクラウドの承認・redaction と同じ
  扱いにしてはならない。送信先がローカルであることを Router / ContextPolicy に伝えることで、
  誤ってクラウド経路として扱われる事故を防ぐ（既存の Vision / Screen Context Provider の
  `local_only` 方針と整合）。
- **4（keyless）**: ローカルサーバーは認証不要が多く、空の Authorization ヘッダは実サーバーで
  400 等の原因になり得る。key が無ければヘッダ自体を送らない。
- **5（ScreenDescriber 見送り）**: Ollama の画像付き `/api/chat` は `images` 配列・`think` など
  Ollama 固有仕様に依存しており、OpenAI-compatible へ機械的に移せない。移行の価値は HUD や
  センサー統合が先に確定してから再評価する。
- **6（性能主張禁止）**: backend の優劣はモデル・量子化・VRAM・サーバー実装に依存し、
  ベンチマークなしの主張は誤った選択を誘導する。
- **7（loopback 限定）**: `local=True` で登録された provider は cloud の承認・redaction
  セマンティクスを受けない。任意の LAN / 公開 / 曖昧な host 名を許すと、クラウド経路の保護を
  迂回して機密情報を送信できる境界になるため、既定値と同じ loopback に限定する。リモートの
  信頼済みノード（LAN / VPN / 公開）は「接続先が本当に信頼できるか」の明示的な設計と
  承認フローの整合が必要で、本マイルストーンの範囲外として deferred にする。

## 前提（assumptions）

| ID | 前提 | 推奨既定 |
| --- | --- | --- |
| A1 | ローカルサーバーは OpenAI-compatible な `/v1/chat/completions`（`base_url`＋モデル名指定）を提供する | `base_url` は設定で与え、既定は `http://localhost:8080/v1`（Ollama は不変） |
| A2 | Ollama が既定 backend である（後方互換） | 設定未指定時は `OllamaProvider` |
| A3 | ローカル provider の送信先は常にローカルであり、`local=True` で表現する | Router / ContextPolicy は local target のみ許可 |
| A4 | 認証付きローカルサーバー（例: LM Studio の API key 設定）では key を渡せる | key があれば Authorization を送る。無ければ送らない |
| A5 | openai_compatible の送信先は同一マシンの loopback のみ（本マイルストーン） | `localhost` / loopback IP（IPv4 / IPv6）。任意 host 名・非 loopback IP は `ValueError` |

## 解決済み（resolved）

| ID | 事項 | 実装 |
| --- | --- | --- |
| U1 | モデル一覧・存在確認（Ollama の `list_models` / `has_model` 相当）を OpenAI-compatible でどう扱うか | `/v1/models` を best-effort で試し、失敗・非2xx・不正JSON時は空を返す（raise しない）。chat-only サーバーでも生成に必須としない |
| U3 | Registry / Router の選択キー（provider_id 命名） | kind 別既定 `local-openai`（openai_compatible）/ `ollama`、明示 `local_provider_id` による上書き対応。OllamaProvider にも解決済み provider_id を渡し、エラー・ログの ID を Registry キーと一致させる |
| U5 | ストリーミングと `is_available`（ネットワーク probe の是非） | `is_available` はライフサイクルのみ（未 close なら True）で probe しない。ストリーミングは `generate_stream` 契約を維持し、到達性は `generate` / `generate_stream` で遅延確立 |

## 未解決（unresolved）と推奨

| ID | 未解決事項 | 推奨既定（デフォルト明示） |
| --- | --- | --- |
| U2 | Ollama 固有パラメータ（`num_ctx` 等）の OpenAI-compatible backend でのマッピング | OpenAI 互換項目のみ送信し、非対応項目は**無視**（`cloud_http.py` と同方針）。`num_ctx` は Ollama 時のみ使用 |
| U4 | 複数ローカルサーバーの同時運用表現 | 当面は単一 provider_id に `base_url` / `model` を差し替える形で対応。複数同時は要望が出てから再検討 |
| U6 | ベンチマーク定義（指標・ハードウェア・対象モデル） | tokens/sec・レイテンシ・VRAM を、使用機種・モデル・量子化・サーバー実装を明記して計測。計測条件なしの主張はしない |
| U7 | リモートの信頼済みノード（LAN / VPN / 公開ホスト）の `local=True` 登録 | deferred。`local=True` は cloud の承認・redaction を迂回するため、接続先の信頼・認証・通知を定めた**別途の明示的な trust design** を経た後にのみ許可する。それまでは loopback 限定（決定7） |

## 検証・確認

- 本記録は実装と整合する（U1 / U3 / U5 は実装済みとして確定、決定7・A5 は
  `src/chat/config.py` の `validate_local_base_url` で強制され、
  `tests/chat/test_config.py` / `tests/assistant/test_factory.py` で検証される）。
- 実装状態は [implementation_status.md](../implementation_status.md) を正とする。

## 関連文書

- [implementation_plan.md](../implementation_plan.md): P0-2（交換可能なローカル推論 backend）
- [implementation_status.md](../implementation_status.md): 実装状態の正典
- [src/llm/providers/cloud_http.py](../../src/llm/providers/cloud_http.py): 既存 OpenAI-compatible（cloud 前提）の参照実装
- [archive/assistant_platform_plan.md](../archive/assistant_platform_plan.md): Phase K（クラウド経路）
- [plans/companion_roadmap.md](../plans/companion_roadmap.md): モデル利用方針（§6）