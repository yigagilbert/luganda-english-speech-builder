#!/usr/bin/env bash
# ============================================================
#  run_pipeline.sh
#  Full end-to-end Luganda–English Speech Dataset Pipeline
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="${CONFIG:-$PROJECT_ROOT/config/config.yaml}"

echo "=============================================="
echo "  Luganda–English Speech Dataset Pipeline"
echo "  Config: $CONFIG"
echo "=============================================="

# Activate virtual environment if present
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "Virtual environment activated."
fi

# Load .env if present
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
    echo ".env loaded."
fi

cd "$PROJECT_ROOT"

# Run the full pipeline (resumes from last checkpoint automatically)
python -m luganda_pipeline.pipeline --config "$CONFIG"

echo "=============================================="
echo "  Pipeline finished successfully."
echo "=============================================="
