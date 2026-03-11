#!/usr/bin/env bash
# ============================================================
#  run_cv24_gpu_fast.sh
#  GPU-optimized launcher for cv24_luganda/common_voice_24_luganda.py
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Prefer NVMe/ephemeral disk for model + dataset caches.
if [ -d "/mnt/nvme" ]; then
    FAST_DISK_ROOT="/mnt/nvme"
elif [ -d "/local" ]; then
    FAST_DISK_ROOT="/local"
else
    FAST_DISK_ROOT="$PROJECT_ROOT/.cache"
fi

export HF_HOME="${HF_HOME:-$FAST_DISK_ROOT/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$HF_HOME/torch}"

# Throughput knobs for translation/TTS heavy stages.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

if command -v nproc >/dev/null 2>&1; then
    CPU_COUNT="$(nproc)"
else
    CPU_COUNT="$(sysctl -n hw.ncpu 2>/dev/null || echo 8)"
fi

if [ "$CPU_COUNT" -gt 24 ]; then
    THREADS=24
elif [ "$CPU_COUNT" -gt 4 ]; then
    THREADS="$CPU_COUNT"
else
    THREADS=4
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$THREADS}"

# Runtime controls (override via env if desired).
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/cv24_luganda}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/data/cv24_luganda/processed}"
PAIRED_OUTPUT_DIR="${PAIRED_OUTPUT_DIR:-$OUTPUT_DIR/paired_general_steps}"
SPLITS="${SPLITS:-validated}"
PIPELINE_CONFIG="${PIPELINE_CONFIG:-$PROJECT_ROOT/config/config.gpu_fast.yaml}"
RUN_GENERAL_STEPS="${RUN_GENERAL_STEPS:-1}"
PUSH_LUG_ONLY="${PUSH_LUG_ONLY:-0}"
PUSH_PAIRED="${PUSH_PAIRED:-1}"
LUGANDA_REPO_ID="${LUGANDA_REPO_ID:-}"
PAIRED_REPO_ID="${PAIRED_REPO_ID:-}"
NUM_WORKERS="${NUM_WORKERS:-$(( CPU_COUNT > 4 ? CPU_COUNT - 2 : 2 ))}"
TARGET_SR="${TARGET_SR:-16000}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
HF_VISIBILITY="${HF_VISIBILITY:-private}"  # private | public

if [ -z "${CV24_DOWNLOAD_URL:-}" ] && [ "$SKIP_DOWNLOAD" != "1" ]; then
    echo "ERROR: CV24_DOWNLOAD_URL is not set and SKIP_DOWNLOAD != 1."
    echo "Set CV24_DOWNLOAD_URL to your Mozilla signed URL."
    exit 1
fi

PY_BIN="python3"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
    PY_BIN="python"
fi

cd "$PROJECT_ROOT"

echo "=============================================="
echo "  CV24 Luganda GPU-Optimized Run"
echo "  Data dir    : $DATA_DIR"
echo "  Output dir  : $OUTPUT_DIR"
echo "  Splits      : $SPLITS"
echo "  Workers     : $NUM_WORKERS"
echo "  HF_HOME     : $HF_HOME"
echo "=============================================="

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
fi

CMD=(
    "$PY_BIN" "$PROJECT_ROOT/cv24_luganda/common_voice_24_luganda.py"
    "--data-dir" "$DATA_DIR"
    "--output-dir" "$OUTPUT_DIR"
    "--splits" "$SPLITS"
    "--num-workers" "$NUM_WORKERS"
    "--target-sr" "$TARGET_SR"
    "--pipeline-config" "$PIPELINE_CONFIG"
    "--paired-output-dir" "$PAIRED_OUTPUT_DIR"
)

if [ "$SKIP_DOWNLOAD" = "1" ]; then
    CMD+=("--skip-download")
fi

if [ "$SKIP_EXTRACT" = "1" ]; then
    CMD+=("--skip-extract")
fi

if [ "$RUN_GENERAL_STEPS" = "1" ]; then
    CMD+=("--run-general-steps")
fi

if [ "$PUSH_LUG_ONLY" = "1" ]; then
    if [ -z "$LUGANDA_REPO_ID" ]; then
        echo "ERROR: PUSH_LUG_ONLY=1 but LUGANDA_REPO_ID is empty."
        exit 1
    fi
    CMD+=("--push-to-hub" "--hub-repo-id" "$LUGANDA_REPO_ID")
fi

if [ "$PUSH_PAIRED" = "1" ]; then
    if [ "$RUN_GENERAL_STEPS" != "1" ]; then
        echo "ERROR: PUSH_PAIRED=1 requires RUN_GENERAL_STEPS=1."
        exit 1
    fi
    if [ -z "$PAIRED_REPO_ID" ]; then
        echo "ERROR: PUSH_PAIRED=1 but PAIRED_REPO_ID is empty."
        exit 1
    fi
    CMD+=("--push-paired-to-hub" "--paired-hub-repo-id" "$PAIRED_REPO_ID")
fi

if [ "$HF_VISIBILITY" = "public" ]; then
    CMD+=("--hub-public" "--paired-hub-public")
else
    CMD+=("--hub-private" "--paired-hub-private")
fi

"${CMD[@]}"
