# Discord Voice STT Corruption & Hallucination Fix

## Goal
- discord-ext-voice_recv の `OpusError: corrupted stream` で voice listener が止まるのを防ぐ。
- STT 品質を維持しつつ、Whisper の日本語ハルシネーションを transcript から除去する。
- voice_recv の rtcp 系 INFO ログを静音化し、STT 診断ログを見やすくする。

## Files
- `src/discord_bot/voice_stt.py`
- `config/discord.env`
- `config/discord.env.example`
- `tests/test_discord_voice_stt.py`

## Background
- 最初の修正で `wants_opus()` を `True` にし sink 側で手動 Opus デコードを行った。
- これにより crash 自体は起きなくなったが、ライブラリの jitter buffer / FEC / シーケンス管理を無視して1パケットずつ `Decoder.decode(opus, fec=False)` するため、順序入れ替わりや欠落のたびに `corrupted stream` が発生。実測で `packets=2000` に対し `decode_errors=1240`（約62%ドロップ）、音声が寸断されて「ブーブー」や空文字ばかりになった。
- また `DISCORD_VOICE_STT_MIN_MS` を 400 → 1200 に上げすぎ、1.2秒未満の発話を潰してしまっていた。
- 次に `wants_opus()=False` へ戻し、`PacketDecoder._decode_packet` を monkeypatch して `OpusError` を「フレーム破棄（空PCM）＋デコーダ作り直し」で握りつぶした。crash は止まったが**精度が壊滅的に低いまま**だった（4.5秒の発話が `ブーブー` になる等、全区間ノイズ扱い）。
- 原因を実証: Opus は状態を持つコーデックのため、この処理は (1) 破損フレーム分の音声を**削除**して以降を時間圧縮し、(2) デコーダ作り直しで後続フレームを別途破壊する。合成音声を実 Opus エンコードし 15% パケットロスを模擬したところ、旧方式は **1.7秒消失・波形相関 0.02**（クリーン基準比）まで崩壊した。これが「ブーブー／チャッチャッ」の正体。

## Steps
1. `wants_opus()` を `False` に戻し、ライブラリの `PacketDecoder` に PCM デコードを任せる（`data.pcm` を使用）。
2. `PacketDecoder._decode_packet` をモンキーパッチし、`OpusError` 時は**フレーム破棄やデコーダ作り直しをせず**、`decoder.decode(None, fec=False)` によるパケットロス補間（PLC）で ~20ms の PCM を生成して返す。デコーダ状態を保持するため音声の連続性が保たれ、listener も停止しない。握りつぶした件数はモジュールレベル `_OPUS_DECODE_ERRORS` に集計し、`opus_decode_error_count()` 経由で status / ログに反映（旧 `sink.decode_errors` は常時0の死んだカウンタだったので削除）。
   - 診断用に `DISCORD_VOICE_STT_DEBUG_AUDIO_DIR`（任意）を追加。設定すると Whisper へ渡す 16kHz mono をそのまま WAV 保存し、実音声の品質を客観確認できる。通常運用では未設定。
3. VAD パラメータを元に戻す（`ENERGY_THRESHOLD=0.008 / SILENCE_MS=700 / MIN_MS=400`）。
4. Whisper ハルシネーションフィルタを追加:
   - `_is_likely_hallucination()` で既知のジャンクフレーズ（ご視聴ありがとうございました / おつかれさまでした / ありがとうございました 等）と、2文字以下のかなのみ断片（すっ / ん 等）を除外。
   - 新規 env `DISCORD_VOICE_STT_HALLUCINATION_FILTER=true`（デフォルト ON）で制御可能。
5. `logging.getLogger("discord.ext.voice_recv.reader").setLevel(WARNING)` で `Received unexpected rtcp packet` の INFO ログを抑制。
6. 診断ログ（`receiving audio` / `speech segment queued` / `empty transcript` / `filtered hallucination` / `listening started/stopped`）と status 項目（`voice_received_audio_sec` / `voice_decode_errors`）は維持。

## Verification
- `.venv/bin/python -m py_compile src/discord_bot/voice_stt.py`
- `.venv/bin/python -m unittest tests.test_discord_voice_stt -v`（ハルシネーションフィルタのテスト2件を追加、計4件）
- `bash scripts/service_ctl.sh restart discord` で起動後、ログに `voice STT: enabled=True available=True` を確認。
- **精度確認（オフライン E2E）**: 実 Opus エンコード → PLC デコード（本修正）→ `pcm48_stereo_to_16k_mono` → Whisper の全経路を通して測定。
  - リサンプラ: 440Hz 入力 → 440Hz 出力、エイリアスなし。
  - 破損耐性（旧 vs 新, 15% ロス）: 旧 = 1.7秒消失 / 波形相関 0.02、新 = 消失0 / 相関 0.87。
  - 英語全文（espeak）: クリーン〜30% ロスまで**完璧・劣化なし**。
  - 日本語（KokoroTTS `こんにちは`）: 0%・15% ロスで**完璧**、30% でも `コンニチイエ` と認識可能（≠ ブーブー）。
- 実機確認手順（任意の最終サインオフ）: `/voice start` → 話す → `/voice status`
  - 良い兆候: `voice_listening: True`、`voice_received_audio_sec` が素直に増加、`voice_decode_errors`（実カウンタ）が低い、意味のあるテキストが `voice_transcripts` に増える。
  - corruption が起きても `opus decode error concealed` ログのみで listener は停止しない。
- 備考: KokoroTTS(日本語) はこの環境で pyopenjtalk 音素化が多くの文をループ・誤変換する既知の別問題あり（本修正とは無関係）。
