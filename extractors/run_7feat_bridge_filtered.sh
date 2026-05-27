#!/bin/bash
# 7-feature extraction for 12 finetuned VLAs on BridgeV2 filtered (N=3245)
# Uses images_seq/ wrapper to handle index_map remapping.
# OpenVLA excluded (extraction bug).
#
# Usage:
#   bash extractors/run_7feat_bridge_filtered.sh [phase] [--max_samples N]
#   phase = dtw, smoke, starvla, cogact, spatialvla, pi0, groot, all
#   --max_samples N  (default: full N=3245; use 2 for smoke test)
#
# Examples:
#   bash extractors/run_7feat_bridge_filtered.sh dtw
#   bash extractors/run_7feat_bridge_filtered.sh smoke --max_samples 2
#   bash extractors/run_7feat_bridge_filtered.sh all

set -uo pipefail
# NOTE: not using set -e so that one model failure doesn't stop all others

PROJECT_DIR="/home/ubuntu/vla/cknna_project"
# Wrapper dir: images/ -> images_seq/ (index_map remapping)
DATA_DIR="/lambda/nfs/vla/cache/cknna_data_store/cknna_data_bridge"
# Original dir (for DTW, which doesn't use images)
DATA_DIR_ORIG="/lambda/nfs/vla/cache/cknna_data_store/cknna_data_bridge"
OUTPUT_BASE="/lambda/nfs/vla/cache/cknna_data_store/cknna_data_bridge_7feat"
DTW_CACHE_DIR="$DATA_DIR_ORIG/dtw_cache"
LOG_DIR="/lambda/nfs/vla/cache/cknna_data_store/logs/7feat_bridge"
mkdir -p "$LOG_DIR" "$OUTPUT_BASE" "$DTW_CACHE_DIR"

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Parse args
PHASE="${1:-all}"
shift || true
MAX_SAMPLES=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max_samples) MAX_SAMPLES="--max_samples $2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SAMPLE_ARG="$MAX_SAMPLES"

# ── Phase functions ──────────────────────────────────────────────────────────

run_dtw() {
    echo ">>> DTW cache computation — $(date '+%H:%M:%S')"
    conda run -n starVLA python -u \
        "$PROJECT_DIR/compute/compute_dtw.py" \
        --data_dir "$DATA_DIR_ORIG" \
        --output_dir "$DTW_CACHE_DIR" \
        --horizons 1 3 7 15 25 40 \
        --topk 10 \
        2>&1 | tee "$LOG_DIR/dtw.log"
    echo ">>> DTW DONE — $(date '+%H:%M:%S')"
}

run_starvla() {
    local name="$1" ckpt="$2"
    echo ">>> StarVLA: $name — $(date '+%H:%M:%S')"
    PYTHONNOUSERSITE=1 \
    STARVLA_ROOT="$PROJECT_DIR/repos/starVLA" \
    conda run -n starVLA python -u \
        "$PROJECT_DIR/extractors/extract_7feat_starvla.py" \
        --ckpt_path "$ckpt" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_BASE/$name" \
        $SAMPLE_ARG \
        2>&1 | tee "$LOG_DIR/starvla_${name}.log"
    echo ">>> $name DONE — $(date '+%H:%M:%S')"
}

run_cogact() {
    local ckpt="$1" dit="$2" name="$3"
    echo ">>> CogACT: $name ($dit) — $(date '+%H:%M:%S')"
    PYTHONNOUSERSITE=1 \
    COGACT_ROOT="$PROJECT_DIR/repos/CogACT" \
    conda run -n cogact python -u \
        "$PROJECT_DIR/extractors/extract_7feat_cogact.py" \
        --ckpt "$ckpt" \
        --action_model_type "$dit" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_BASE/$name" \
        $SAMPLE_ARG \
        2>&1 | tee "$LOG_DIR/cogact_${name}.log"
    echo ">>> $name DONE — $(date '+%H:%M:%S')"
}

run_spatialvla() {
    echo ">>> SpatialVLA — $(date '+%H:%M:%S')"
    PYTHONNOUSERSITE=1 conda run -n spatialvla_env python -u \
        "$PROJECT_DIR/extractors/extract_7feat_spatialvla.py" \
        --ckpt /lambda/nfs/vla/cache/spatialvla-bridge \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_BASE/spatialvla-sft-bridge" \
        $SAMPLE_ARG \
        2>&1 | tee "$LOG_DIR/spatialvla.log"
    echo ">>> SpatialVLA DONE — $(date '+%H:%M:%S')"
}

run_pi0() {
    echo ">>> Pi0 — $(date '+%H:%M:%S')"
    PYTHONNOUSERSITE=1 conda run -n pi0fast_env python -u \
        "$PROJECT_DIR/extractors/extract_7feat_pi0.py" \
        --ckpt_path HaomingSong/lerobot-pi0-bridge \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_BASE/pi0-lerobot-bridge" \
        $SAMPLE_ARG \
        2>&1 | tee "$LOG_DIR/pi0.log"
    echo ">>> Pi0 DONE — $(date '+%H:%M:%S')"
}

run_groot() {
    local version="$1" ckpt="$2" name="$3"
    local groot_var="ISAAC_GROOT_DIR"
    local groot_path="$PROJECT_DIR/repos/Isaac-GR00T-N15"
    if [ "$version" = "n16" ]; then
        groot_var="ISAAC_GROOT_N16_DIR"
        groot_path="$PROJECT_DIR/repos/Isaac-GR00T-N16"
    fi
    echo ">>> GR00T $version: $name — $(date '+%H:%M:%S')"
    export PYTHONNOUSERSITE=1
    export "${groot_var}=${groot_path}"
    conda run -n groot_libero python -u \
        "$PROJECT_DIR/extractors/extract_7feat_groot.py" \
        --version "$version" \
        --ckpt "$ckpt" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_BASE/$name" \
        $SAMPLE_ARG \
        2>&1 | tee "$LOG_DIR/groot_${name}.log"
    echo ">>> $name DONE — $(date '+%H:%M:%S')"
}

# ── StarVLA checkpoints ──────────────────────────────────────────────────────
CKPT_BASE="/lambda/nfs/vla/cache/starVLA/playground/Pretrained_models"

# ── Execution ─────────────────────────────────────────────────────────────────

echo "============================================"
echo "7-Feature Extraction: BridgeV2 Filtered (N=3245)"
echo "Phase: $PHASE  $SAMPLE_ARG"
echo "Data dir: $DATA_DIR"
echo "Output: $OUTPUT_BASE"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

START_TIME=$(date +%s)

if [[ "$PHASE" == "dtw" || "$PHASE" == "all" ]]; then
    run_dtw
fi

if [[ "$PHASE" == "smoke" || "$PHASE" == "all" ]]; then
    echo ""
    echo "=== SMOKE TEST (N=${MAX_SAMPLES:-3245}) ==="
    # Run one model from each architecture family
    run_starvla "Qwen-GR00T-Bridge" "$CKPT_BASE/Qwen-GR00T-Bridge/checkpoints/steps_45000_pytorch_model.pt"
    run_cogact "CogACT/CogACT-Small" "DiT-S" "cogact-small-bridge"
    run_spatialvla
    run_pi0
    run_groot "n15" "/lambda/nfs/vla/cache/GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2" "groot-n15-bridge"
fi

if [[ "$PHASE" == "starvla" || "$PHASE" == "all" ]]; then
    echo ""
    echo "=== StarVLA (5 models) ==="
    run_starvla "Qwen-GR00T-Bridge" "$CKPT_BASE/Qwen-GR00T-Bridge/checkpoints/steps_45000_pytorch_model.pt"
    run_starvla "Qwen-GR00T-Bridge-RT-1" "$CKPT_BASE/Qwen-GR00T-Bridge-RT-1/checkpoints/steps_30000_pytorch_model.pt"
    run_starvla "Qwen-OFT-Bridge-RT-1" "$CKPT_BASE/Qwen-OFT-Bridge-RT-1/checkpoints/steps_10000_pytorch_model.pt"
    run_starvla "Qwen3VL-GR00T-Bridge-RT-1" "$CKPT_BASE/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt"
    run_starvla "Qwen3VL-OFT-Bridge-RT-1" "$CKPT_BASE/Qwen3VL-OFT-Bridge-RT-1/checkpoints/steps_5000_pytorch_model.pt"
fi

if [[ "$PHASE" == "cogact" || "$PHASE" == "all" ]]; then
    echo ""
    echo "=== CogACT (3 models) ==="
    run_cogact "CogACT/CogACT-Small" "DiT-S" "cogact-small-bridge"
    run_cogact "CogACT/CogACT-Base" "DiT-B" "cogact-base-bridge"
    run_cogact "CogACT/CogACT-Large" "DiT-L" "cogact-large-bridge"
fi

if [[ "$PHASE" == "spatialvla" || "$PHASE" == "all" ]]; then
    echo ""
    echo "=== SpatialVLA ==="
    run_spatialvla
fi

if [[ "$PHASE" == "pi0" || "$PHASE" == "all" ]]; then
    echo ""
    echo "=== Pi0 ==="
    run_pi0
fi

if [[ "$PHASE" == "groot" || "$PHASE" == "all" ]]; then
    echo ""
    echo "=== GR00T ==="
    run_groot "n15" "/lambda/nfs/vla/cache/GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2" "groot-n15-bridge"
    run_groot "n16" "/lambda/nfs/vla/cache/GR00T-N1.6-bridge" "groot-n16-bridge"
fi

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
echo ""
echo "============================================"
echo "ALL DONE. Total time: $((ELAPSED/60))m $((ELAPSED%60))s"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
