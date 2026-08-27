#!/usr/bin/env bash
set -euo pipefail

model="${PI_CODEX_MODEL:-}"
thinking="${PI_CODEX_THINKING:-}"
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

pi_args=(--approve)
if [[ -n "$model" ]]; then
  pi_args+=(--model "$model")
fi
if [[ -n "$thinking" ]]; then
  pi_args+=(--thinking "$thinking")
fi

exec pi "${pi_args[@]}" "$@"
