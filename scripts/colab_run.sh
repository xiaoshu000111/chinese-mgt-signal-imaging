#!/usr/bin/env bash
# Run the project with the current Drive-backed Colab environment.
# Usage: bash scripts/colab_run.sh extract --name D1_smoke_extract --purpose "..."
set -euo pipefail

ENV_FILE="${COLAB_ENV_FILE:-scripts/colab.env}"
if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE. Run scripts/colab_bootstrap.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"
exec python run.py "$@"
