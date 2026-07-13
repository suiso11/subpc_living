#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_ROOT}/.venv-sbv2"
ASSETS_ROOT="${PROJECT_ROOT}/models/tts/style_bert_vits2/model_assets"

# Model selection (env-overridable).
# Default: JVNV F1 JP (litagin/style_bert_vits2_jvnv).
# Tsukuyomi-chan example:
#   SBV2_REPO=ayousanz/tsukuyomi-chan-style-bert-vits2-model \
#   SBV2_MODEL_NAME=tsukuyomi-chan \
#   SBV2_MODEL_FILE=tsukuyomi-chan_e200_s5200.safetensors \
#   SBV2_FILES=tsukuyomi-chan_e200_s5200.safetensors,config.json,style_vectors.npy \
#   bash scripts/setup_style_bert_vits2.sh
SBV2_REPO="${SBV2_REPO:-litagin/style_bert_vits2_jvnv}"
SBV2_MODEL_NAME="${SBV2_MODEL_NAME:-jvnv-F1-jp}"
SBV2_MODEL_FILE="${SBV2_MODEL_FILE:-${SBV2_MODEL_NAME}_e160_s14000.safetensors}"
# Comma-separated repo-internal file paths.
# Default assumes repo stores files under {model_name}/ (jvnv layout).
# For repos with files at root (tsukuyomi-chan), set SBV2_FILES without the prefix.
SBV2_FILES="${SBV2_FILES:-${SBV2_MODEL_NAME}/${SBV2_MODEL_FILE},${SBV2_MODEL_NAME}/config.json,${SBV2_MODEL_NAME}/style_vectors.npy}"

# Create venv only when missing so model swaps don't reinstall torch.
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip wheel "setuptools<81"

  "${VENV_DIR}/bin/python" -m pip install \
    "torch<2.4" \
    "torchaudio<2.4" \
    --index-url https://download.pytorch.org/whl/cu121

  "${VENV_DIR}/bin/python" -m pip install \
    "style-bert-vits2==2.5.0" \
    "huggingface_hub<1" \
    "transformers<5" \
    "numpy<2" \
    "nltk<=3.8.1" \
    "scipy<1.17"

  "${VENV_DIR}/bin/python" -m pip check
else
  echo "venv already exists at ${VENV_DIR}, skipping install."
fi

mkdir -p "$ASSETS_ROOT"
export SBV2_REPO SBV2_MODEL_NAME SBV2_MODEL_FILE SBV2_FILES
export PROJECT_ROOT
"${VENV_DIR}/bin/python" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

project_root = Path(os.environ["PROJECT_ROOT"])
assets_root = project_root / "models" / "tts" / "style_bert_vits2" / "model_assets"
repo = os.environ["SBV2_REPO"]
model_name = os.environ["SBV2_MODEL_NAME"]
files = [f.strip() for f in os.environ["SBV2_FILES"].split(",") if f.strip()]

# Repo files may live under {model_name}/ (jvnv) or at root (tsukuyomi-chan).
# We normalise the local layout to assets_root/{model_name}/{basename} so the
# server's model_root/model_name lookup works for both.
has_prefix = any("/" in f for f in files)
local_dir = assets_root if has_prefix else assets_root / model_name
target_dir = assets_root / model_name
target_dir.mkdir(parents=True, exist_ok=True)

for file in files:
    downloaded = Path(hf_hub_download(repo, file, local_dir=local_dir))
    final = target_dir / downloaded.name
    if downloaded != final:
        final.write_bytes(downloaded.read_bytes())
    print(final)

print(f"downloaded {len(files)} files for model={model_name} repo={repo}")
PY

echo "Style-Bert-VITS2 setup complete: ${VENV_DIR}"
echo "Model: ${SBV2_MODEL_NAME} (${SBV2_REPO})"
