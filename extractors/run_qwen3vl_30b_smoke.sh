#!/bin/bash
# Smoke test: Qwen3-VL-30B-A3B 7-feature extraction on DROID (N=10)
# Uses 4-bit quantization (NF4) to fit on A100-40GB.
#
# Usage:
#   bash extractors/run_qwen3vl_30b_smoke.sh           # N=10 smoke test
#   bash extractors/run_qwen3vl_30b_smoke.sh --full     # full N=6000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"
[[ -f "$PROJECT_DIR/paths.env" ]] && { set -a; source "$PROJECT_DIR/paths.env"; set +a; }
DATA_DIR="${DATA_DIR:-${DATA_STORE:?DATA_STORE not set — source paths.env}/cknna_data_droid}"
EXTRACTOR="$PROJECT_DIR/extractors/extract_all_features.py"
PYTHON="PYTHONNOUSERSITE=1 ${CONDA_ROOT}/envs/starVLA/bin/python"

export PYTHONPATH="$PROJECT_DIR/repos/Qwen3-VL/qwen-vl-utils/src/:${PYTHONPATH:-}"
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN not set — required to fetch gated HF models." >&2
    exit 1
fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8

MODEL_NAME="qwen3vl-30b-a3b-raw"
MODEL_FAMILY="qwen3_moe"
MODEL_PATH="Qwen/Qwen3-VL-30B-A3B-Instruct"
OUTPUT_DIR="$DATA_DIR/$MODEL_NAME"

MAX_SAMPLES_ARG="--max_samples 10"
if [[ "${1:-}" == "--full" ]]; then
    MAX_SAMPLES_ARG=""
    echo "=== FULL EXTRACTION MODE (N=6000) ==="
else
    echo "=== SMOKE TEST MODE (N=10) ==="
fi

echo "Model: $MODEL_NAME ($MODEL_FAMILY)"
echo "HF: $MODEL_PATH"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

START_TIME=$(date +%s)

eval $PYTHON -u "$EXTRACTOR" \
    --model_path "$MODEL_PATH" \
    --model_family "$MODEL_FAMILY" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    $MAX_SAMPLES_ARG 2>&1

EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo "============================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "PASSED in ${MINS}m ${SECS}s"
    echo "Output files:"
    ls -lh "$OUTPUT_DIR"/feats_A*.pt "$OUTPUT_DIR"/extraction_metadata_all7.json 2>/dev/null
else
    echo "FAILED (exit code $EXIT_CODE)"
fi
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
