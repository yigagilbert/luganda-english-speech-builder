#!/usr/bin/env bash
# ============================================================
#  run_stage.sh  — run a single named pipeline stage
#
#  Usage:
#    bash scripts/run_stage.sh <stage> [--force]
#
#  Example:
#    bash scripts/run_stage.sh translation
#    bash scripts/run_stage.sh filtering --force
# ============================================================
set -euo pipefail

STAGE="${1:-}"
FORCE="${2:-}"

if [ -z "$STAGE" ]; then
    echo "Usage: $0 <stage> [--force]"
    echo "Stages: ingestion preprocessing filtering translation tts assembly qa"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a && source "$PROJECT_ROOT/.env" && set +a
fi

cd "$PROJECT_ROOT"

echo "Running stage: $STAGE $FORCE"
python -m luganda_pipeline.pipeline --stage "$STAGE" $FORCE
