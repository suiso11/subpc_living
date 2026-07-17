#!/usr/bin/env bash
set -euo pipefail

model="${PI_CODEX_MODEL:-openai-codex/gpt-5.5}"
thinking="${PI_CODEX_THINKING:-high}"
coordinator_only_turn="${PI_SUBAGENT_DISABLE_COORDINATOR_ONLY_TURN:-}"

if [[ -z "$coordinator_only_turn" ]]; then
  coordinator_only_turn=0
  for arg in "$@"; do
    if [[ "$arg" == "-p" || "$arg" == "--print" ]]; then
      coordinator_only_turn=1
      break
    fi
  done
fi

export PI_ORCHESTRATOR_MODE=1
export PI_SUBAGENT_DISABLE_COORDINATOR_ONLY_TURN="$coordinator_only_turn"

exec pi \
  --approve \
  --model "$model" \
  --thinking "$thinking" \
  "$@"
