#!/bin/bash
# Full 7-feature extraction on DROID dataset (N=6000) for all 8 raw VLMs.
# Run in tmux: tmux new-session -s all7_droid "bash extractors/run_all7_droid_full.sh 2>&1 | tee /tmp/all7_droid.log"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"
[[ -f "$PROJECT_DIR/paths.env" ]] && { set -a; source "$PROJECT_DIR/paths.env"; set +a; }
DATA_DIR="${DATA_DIR:-${DATA_STORE:?DATA_STORE not set — source paths.env}/cknna_data_droid}"
EXTRACTOR="$PROJECT_DIR/extractors/extract_all_features.py"
PYTHON="python3"

export PYTHONPATH="$PROJECT_DIR/repos/Qwen3-VL/qwen-vl-utils/src/:${PYTHONPATH:-}"
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN not set — required to fetch gated HF models." >&2
    exit 1
fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8

declare -A MODEL_FAMILY=(
    ["kosmos2-raw"]="kosmos2"
    ["qwen3vl-2b-raw"]="qwen3"
    ["paligemma1-raw"]="paligemma"
    ["paligemma2-raw"]="paligemma"
    ["qwen25vl-3b-raw"]="qwen2.5"
    ["qwen3vl-4b-raw"]="qwen3"
    ["qwen3vl-8b-raw"]="qwen3"
    ["qwen25vl-7b-raw"]="qwen2.5"
)

declare -A MODEL_PATH=(
    ["kosmos2-raw"]="microsoft/kosmos-2-patch14-224"
    ["qwen3vl-2b-raw"]="Qwen/Qwen3-VL-2B-Instruct"
    ["paligemma1-raw"]="google/paligemma-3b-pt-224"
    ["paligemma2-raw"]="google/paligemma2-3b-pt-224"
    ["qwen25vl-3b-raw"]="Qwen/Qwen2.5-VL-3B-Instruct"
    ["qwen3vl-4b-raw"]="Qwen/Qwen3-VL-4B-Instruct"
    ["qwen3vl-8b-raw"]="Qwen/Qwen3-VL-8B-Instruct"
    ["qwen25vl-7b-raw"]="Qwen/Qwen2.5-VL-7B-Instruct"
)

# Order by VRAM (smallest first)
MODEL_ORDER=(kosmos2-raw qwen3vl-2b-raw paligemma1-raw paligemma2-raw qwen25vl-3b-raw qwen3vl-4b-raw qwen3vl-8b-raw qwen25vl-7b-raw)

TOTAL=${#MODEL_ORDER[@]}
PASSED=0
FAILED=0
START_TIME=$(date +%s)

echo "=== FULL DROID EXTRACTION (N=6000, 7 features) ==="
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

    MODEL_START=$(date +%s)
    if $PYTHON -u "$EXTRACTOR" \
        --model_path "$HF_PATH" \
        --model_family "$FAMILY" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" 2>&1 | while IFS= read -r line; do echo "  $line"; done; then
        MODEL_END=$(date +%s)
        MODEL_ELAPSED=$(( MODEL_END - MODEL_START ))
        echo "  ✓ $MODEL_NAME PASSED (${MODEL_ELAPSED}s)"
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
