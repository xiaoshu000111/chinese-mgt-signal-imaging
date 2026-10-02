#!/usr/bin/env bash
# Prepare one Colab checkout and one isolated Google Drive run directory.
# Drive must be mounted at /content/drive before calling this script.
#
# Example:
#   bash scripts/colab_bootstrap.sh \
#     --drive-root /content/drive/MyDrive/chinese-mgt-signal-imaging \
#     --run-id D1_smoke_qwen025b_T512
set -euo pipefail

DRIVE_ROOT="/content/drive/MyDrive/chinese-mgt-signal-imaging"
RUN_ID="D0_smoke"
ENV_FILE="scripts/colab.env"
INSTALL=1

while [ "$#" -gt 0 ]; do
  case "$1" in
    --drive-root) DRIVE_ROOT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --no-install) INSTALL=0; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ ! -d "/content/drive" ]; then
  echo "Google Drive is not mounted. In Colab, run: from google.colab import drive; drive.mount('/content/drive')" >&2
  exit 1
fi

case "$RUN_ID" in
  *[!A-Za-z0-9._-]*) echo "run-id contains unsafe characters: $RUN_ID" >&2; exit 2 ;;
esac

RUN_ROOT="$DRIVE_ROOT/runs/$RUN_ID"
mkdir -p \
  "$DRIVE_ROOT/raw" \
  "$DRIVE_ROOT/hf_cache" \
  "$RUN_ROOT/processed" \
  "$RUN_ROOT/splits" \
  "$RUN_ROOT/outputs" \
  "$RUN_ROOT/checkpoints" \
  "$RUN_ROOT/logs" \
  "$RUN_ROOT/analysis"

# This file is intentionally untracked and contains only the current run paths.
# It is sourced by colab_run.sh; do not hand-edit source code paths into it.
cat > "$ENV_FILE" <<EOF
export MGT_DATA_RAW='$DRIVE_ROOT/raw'
export MGT_PROCESSED_ROOT='$RUN_ROOT/processed'
export MGT_SPLITS_ROOT='$RUN_ROOT/splits'
export MGT_OUTPUT_ROOT='$RUN_ROOT/outputs'
export MGT_CHECKPOINT_ROOT='$RUN_ROOT/checkpoints'
export MGT_LOG_ROOT='$RUN_ROOT/logs'
export MGT_ANALYSIS_ROOT='$RUN_ROOT/analysis'
export MGT_RUN_ID='$RUN_ID'
export HF_HOME='$DRIVE_ROOT/hf_cache'
export TRANSFORMERS_CACHE='$DRIVE_ROOT/hf_cache'
export TOKENIZERS_PARALLELISM=false
EOF

if [ "$INSTALL" -eq 1 ]; then
  python -m pip install -q -r requirements-colab.txt
fi

echo "[colab] checkout: $(pwd)"
echo "[colab] drive root: $DRIVE_ROOT"
echo "[colab] run root: $RUN_ROOT"
echo "[colab] env file: $ENV_FILE"
echo "[colab] raw data must be placed in: $DRIVE_ROOT/raw"
echo "[colab] next: source $ENV_FILE && python run.py preflight"
