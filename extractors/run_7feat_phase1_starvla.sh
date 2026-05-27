#!/bin/bash
# Phase 1: 7-feature extraction for all 5 StarVLA models on DROID (N=6000)
# VRAM: ~6GB (Qwen2.5-3B) or ~5GB (Qwen3-2B) per model
# Run this script directly OR launch individual models in parallel tmux panes
set -euo pipefail

PROJECT_DIR="/home/ubuntu/vla/cknna_project"
DATA_DIR="/lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid"
OUTPUT_BASE="/lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid_7feat"
EXTRACTOR="$PROJECT_DIR/extractors/extract_7feat_starvla.py"
CKPT_BASE="/lambda/nfs/vla/cache/starVLA/playground/Pretrained_models"

export STARVLA_ROOT="$PROJECT_DIR/repos/starVLA"
export PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Model name -> checkpoint path
declare -A CKPTS=(
    ["Qwen-GR00T-Bridge"]="$CKPT_BASE/Qwen-GR00T-Bridge/checkpoints/steps_45000_pytorch_model.pt"
    ["Qwen-GR00T-Bridge-RT-1"]="$CKPT_BASE/Qwen-GR00T-Bridge-RT-1/checkpoints/steps_30000_pytorch_model.pt"
    ["Qwen-OFT-Bridge-RT-1"]="$CKPT_BASE/Qwen-OFT-Bridge-RT-1/checkpoints/steps_10000_pytorch_model.pt"
    ["Qwen3VL-GR00T-Bridge-RT-1"]="$CKPT_BASE/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt"
    ["Qwen3VL-OFT-Bridge-RT-1"]="$CKPT_BASE/Qwen3VL-OFT-Bridge-RT-1/checkpoints/steps_5000_pytorch_model.pt"
)

# If a model name is passed as arg, run only that model
if [ "${1:-}" != "" ]; then
    MODEL_NAME="$1"
    CKPT="${CKPTS[$MODEL_NAME]}"
    OUTPUT_DIR="$OUTPUT_BASE/$MODEL_NAME"
    echo "=== Single model: $MODEL_NAME ==="
    echo "  Ckpt: $CKPT"
    echo "  Output: $OUTPUT_DIR"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"

    conda run -n starVLA python -u "$EXTRACTOR" \
        --ckpt_path "$CKPT" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR"

    echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
    exit 0
fi

# Run all models sequentially
MODEL_ORDER=(Qwen-GR00T-Bridge Qwen-GR00T-Bridge-RT-1 Qwen-OFT-Bridge-RT-1 Qwen3VL-GR00T-Bridge-RT-1 Qwen3VL-OFT-Bridge-RT-1)
TOTAL=${#MODEL_ORDER[@]}
PASSED=0
FAILED=0
START_TIME=$(date +%s)

echo "=== Phase 1: StarVLA 7-Feature Extraction (N=6000) ==="
echo "Models: ${MODEL_ORDER[*]}"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

for idx in "${!MODEL_ORDER[@]}"; do
    MODEL_NAME="${MODEL_ORDER[$idx]}"
    CKPT="${CKPTS[$MODEL_NAME]}"
    OUTPUT_DIR="$OUTPUT_BASE/$MODEL_NAME"
    NUM=$((idx + 1))

    echo ""
    echo "[$NUM/$TOTAL] $MODEL_NAME — $(date '+%H:%M:%S')"

    MODEL_START=$(date +%s)
    if conda run -n starVLA python -u "$EXTRACTOR" \
        --ckpt_path "$CKPT" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" 2>&1 | while IFS= read -r line; do echo "  $line"; done; then
        MODEL_END=$(date +%s)
        echo "  ✓ $MODEL_NAME PASSED ($((MODEL_END - MODEL_START))s)"
        PASSED=$((PASSED + 1))
    else
        echo "  ✗ $MODEL_NAME FAILED"
        FAILED=$((FAILED + 1))
    fi
done

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
echo ""
echo "============================================"
echo "DONE: $PASSED passed, $FAILED failed out of $TOTAL"
echo "Total time: $((ELAPSED/60))m $((ELAPSED%60))s"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
