#!/bin/bash
# Run 7-feature extraction smoke test (N=10) for all 8 raw VLMs.
# Usage: bash extractors/run_all7_smoke.sh [--full]
#   --full: run on all samples (N=7762 Bridge), not just N=10

set -euo pipefail

# Source paths.env from repo root for portable paths.
# Override PROJECT_DIR / DATA_DIR by exporting them before sourcing this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"
[[ -f "$PROJECT_DIR/paths.env" ]] && { set -a; source "$PROJECT_DIR/paths.env"; set +a; }
DATA_DIR="${DATA_DIR:-${DATA_STORE:?DATA_STORE not set — source paths.env}/cknna_data_bridge}"
EXTRACTOR="$PROJECT_DIR/extractors/extract_all_features.py"
PYTHON="python3"

export PYTHONPATH="$PROJECT_DIR/repos/Qwen3-VL/qwen-vl-utils/src/:${PYTHONPATH:-}"
# HF_TOKEN must be set externally (export HF_TOKEN=... or put it in paths.env)
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN not set — required to fetch gated HF models." >&2
    exit 1
fi

# Disable cuDNN (not needed for transformer inference, avoids init error)
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# N=10 by default, or full dataset with --full
MAX_SAMPLES_ARG="--max_samples 10"
if [[ "${1:-}" == "--full" ]]; then
    MAX_SAMPLES_ARG=""
    echo "=== FULL EXTRACTION MODE ==="
else
    echo "=== SMOKE TEST MODE (N=10) ==="
fi

# All 8 models: (name, family, hf_path)
declare -A MODEL_FAMILY=(
    ["qwen25vl-3b-raw"]="qwen2.5"
    ["qwen25vl-7b-raw"]="qwen2.5"
    ["qwen3vl-2b-raw"]="qwen3"
    ["qwen3vl-4b-raw"]="qwen3"
    ["qwen3vl-8b-raw"]="qwen3"
    ["paligemma1-raw"]="paligemma"
    ["paligemma2-raw"]="paligemma"
    ["kosmos2-raw"]="kosmos2"
)

declare -A MODEL_PATH=(
    ["qwen25vl-3b-raw"]="Qwen/Qwen2.5-VL-3B-Instruct"
    ["qwen25vl-7b-raw"]="Qwen/Qwen2.5-VL-7B-Instruct"
    ["qwen3vl-2b-raw"]="Qwen/Qwen3-VL-2B-Instruct"
    ["qwen3vl-4b-raw"]="Qwen/Qwen3-VL-4B-Instruct"
    ["qwen3vl-8b-raw"]="Qwen/Qwen3-VL-8B-Instruct"
    ["paligemma1-raw"]="google/paligemma-3b-pt-224"
    ["paligemma2-raw"]="google/paligemma2-3b-pt-224"
    ["kosmos2-raw"]="microsoft/kosmos-2-patch14-224"
)

# Order by VRAM (smallest first): KosMos-2(4GB), Qwen3-2B(5), PG-1(6), PG-2(6), Qwen2.5-3B(8), Qwen3-4B(8), Qwen3-8B(11), Qwen2.5-7B(17)
MODEL_ORDER=(kosmos2-raw qwen3vl-2b-raw paligemma1-raw paligemma2-raw qwen25vl-3b-raw qwen3vl-4b-raw qwen3vl-8b-raw qwen25vl-7b-raw)

TOTAL=${#MODEL_ORDER[@]}
PASSED=0
FAILED=0
START_TIME=$(date +%s)

echo "Models: ${MODEL_ORDER[*]}"
echo "Data: $DATA_DIR"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

for idx in "${!MODEL_ORDER[@]}"; do
    MODEL_NAME="${MODEL_ORDER[$idx]}"
    FAMILY="${MODEL_FAMILY[$MODEL_NAME]}"
    HF_PATH="${MODEL_PATH[$MODEL_NAME]}"
    OUTPUT_DIR="$DATA_DIR/$MODEL_NAME"
    NUM=$((idx + 1))

    echo ""
    echo "[$NUM/$TOTAL] $MODEL_NAME ($FAMILY) — $(date '+%H:%M:%S')"
    echo "  HF: $HF_PATH"
    echo "  Out: $OUTPUT_DIR"

    if $PYTHON -u "$EXTRACTOR" \
        --model_path "$HF_PATH" \
        --model_family "$FAMILY" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        $MAX_SAMPLES_ARG 2>&1 | while IFS= read -r line; do echo "  $line"; done; then
        echo "  ✓ $MODEL_NAME PASSED"
        PASSED=$((PASSED + 1))
    else
        echo "  ✗ $MODEL_NAME FAILED"
        FAILED=$((FAILED + 1))
    fi
done

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo "============================================"
echo "DONE: $PASSED passed, $FAILED failed out of $TOTAL"
echo "Total time: ${MINS}m ${SECS}s"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
