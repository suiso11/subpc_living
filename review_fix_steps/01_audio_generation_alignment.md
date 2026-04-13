# Audio Generation Alignment

## Goal
- `ChatConfig` と `OllamaClient` と音声系呼び出しの引数不整合を解消する。

## Files
- `src/chat/config.py`
- `src/chat/client.py`
- `src/audio/main.py`
- `src/audio/pipeline.py`

## Steps
1. `ChatConfig` に音声系が参照している生成パラメータを追加する。
2. `OllamaClient.generate()` / `generate_stream()` / `generate_stream_queue()` で同じパラメータを受け取れるようにする。
3. Ollama の `options` に渡す値を一元化して、CLI / Web / 音声パイプラインの全経路で同じ設定を使う。
4. 既存の `config/chat_config.json` に新キーがなくても既定値で動作することを保つ。

## Verification
- `python3 -m compileall src`
- `ChatConfig` の定義と `generate_stream()` のシグネチャを静的に確認する。

