#!/usr/bin/env bash
# ============================================================
#  run_gpu_fast.sh
#  GPU-optimized pipeline launcher for rented cloud instances
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="${CONFIG:-$PROJECT_ROOT/config/config.yaml}"
STAGE="${STAGE:-}"   # Optional: translation | tts | ...
FORCE="${FORCE:-}"   # Optional: --force

if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a && source "$PROJECT_ROOT/.env" && set +a
fi

# Put model/dataset caches on fast local disk (override on your VM if needed).
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$HF_HOME/torch}"

# Runtime throughput knobs.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"

cd "$PROJECT_ROOT"

PY_BIN="python3"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
    PY_BIN="python"
fi

echo "=============================================="
echo "  GPU-Optimized Luganda Pipeline Run"
echo "  Config: $CONFIG"
echo "  Stage : ${STAGE:-all}"
echo "  HF_HOME: $HF_HOME"
echo "=============================================="

if [ -n "$STAGE" ]; then
    "$PY_BIN" -m luganda_pipeline.pipeline --config "$CONFIG" --stage "$STAGE" $FORCE
else
    "$PY_BIN" -m luganda_pipeline.pipeline --config "$CONFIG"
fi
