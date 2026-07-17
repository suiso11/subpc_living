#!/usr/bin/env bash
# Convert an already BF16-merged Hugging Face model to a full GGUF and quantize it.
set -euo pipefail
MERGED_MODEL=""; OUTPUT_DIR=""; BASENAME=""; QUANT="Q4_K_M"
LLAMA_DIR="${LLAMA_DIR:-}"; PYTHON="${PYTHON:-python3}"; DRY_RUN=0; FORCE=0
usage(){ cat <<'USAGE'
Usage: convert_personal_model_to_gguf.sh --merged-model DIR --output-dir DIR \
  --basename NAME [--quant Q4_K_M|Q5_K_M] [--llama-dir DIR] [--python BIN] \
  [--force] [--dry-run]
The input must be the fresh BF16 output of training/merge_adapter.py. The base
snapshot and LoRA adapter are never modified.
USAGE
}
err(){ printf 'convert_personal_model_to_gguf: %s\n' "$*" >&2; }
emit(){ printf '%q ' "$@"; printf '\n'; }
while (($#)); do
 case "$1" in
  --merged-model) MERGED_MODEL="${2:-}"; shift 2;;
  --output-dir) OUTPUT_DIR="${2:-}"; shift 2;;
  --basename) BASENAME="${2:-}"; shift 2;;
  --quant) QUANT="${2:-}"; shift 2;;
  --llama-dir) LLAMA_DIR="${2:-}"; shift 2;;
  --python) PYTHON="${2:-}"; shift 2;;
  --dry-run) DRY_RUN=1; shift;; --force) FORCE=1; shift;;
  -h|--help) usage; exit 0;; *) err "unknown argument: $1"; usage; exit 2;;
 esac
done
[[ -n "$MERGED_MODEL" && -n "$OUTPUT_DIR" && -n "$BASENAME" ]] || { err 'required argument missing'; usage; exit 2; }
[[ "$QUANT" =~ ^Q[45]_K_M$ ]] || { err "unsupported quantization: $QUANT"; exit 2; }
[[ "$OUTPUT_DIR" != "$MERGED_MODEL" ]] || { err '--output-dir must differ from --merged-model'; exit 2; }
RAW="$OUTPUT_DIR/$BASENAME-f16.gguf"; OUT="$OUTPUT_DIR/$BASENAME-${QUANT}.gguf"
CONVERT="${LLAMA_DIR:+$LLAMA_DIR/}convert_hf_to_gguf.py"
QUANTIZE="${LLAMA_DIR:+$LLAMA_DIR/}llama-quantize"
if ((DRY_RUN)); then
 echo '# dry-run: no files written'
 emit "$PYTHON" "$CONVERT" "$MERGED_MODEL" --outfile "$RAW" --outtype f16
 emit "$QUANTIZE" "$RAW" "$OUT" "$QUANT"
 exit 0
fi
[[ -d "$MERGED_MODEL" && -f "$MERGED_MODEL/config.json" ]] || { err "merged HF model/config.json not found: $MERGED_MODEL"; exit 1; }
[[ -n "$LLAMA_DIR" && -f "$CONVERT" ]] || { err 'llama.cpp convert_hf_to_gguf.py not found'; exit 1; }
[[ -x "$QUANTIZE" ]] || { err 'llama-quantize not executable'; exit 1; }
[[ -x "$(command -v "$PYTHON" || true)" ]] || { err "python not found: $PYTHON"; exit 1; }
if ((!FORCE)) && { [[ -e "$RAW" ]] || [[ -e "$OUT" ]]; }; then err 'output exists; use --force'; exit 1; fi
mkdir -p "$OUTPUT_DIR"
"$PYTHON" "$CONVERT" "$MERGED_MODEL" --outfile "$RAW" --outtype f16
"$QUANTIZE" "$RAW" "$OUT" "$QUANT"
printf 'GGUF ready: %s\n' "$OUT"
