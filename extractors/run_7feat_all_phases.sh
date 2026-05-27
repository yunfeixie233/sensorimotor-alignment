#!/bin/bash
# Master script: 7-feature extraction for ALL finetuned models on DROID (N=6000)
# Phases 2-4. Phase 1 (StarVLA) runs separately in parallel.
#
# VRAM budget (40GB A100):
#   Phase 2a: OpenVLA-7B (~15GB) — sequential, one at a time
#   Phase 2b: CogACT-7B (~15GB) — sequential, one at a time
#   Phase 3a: SpatialVLA-4B (~8GB) } can run in parallel (~16GB)
#   Phase 3b: Pi0-3B (~8GB)        }
#   Phase 4:  GR00T-3B (~6-8GB each) — can run 2 in parallel
#
# Usage: bash extractors/run_7feat_all_phases.sh [phase]
#   No arg = run all phases 2-4
#   phase = 2a, 2b, 3a, 3b, 4a, 4b

set -euo pipefail

PROJECT_DIR="/home/ubuntu/vla/cknna_project"
DATA_DIR="/lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid"
OUTPUT_BASE="/lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid_7feat"
LOG_DIR="/lambda/nfs/vla/cache/cknna_data_store/logs/7feat"
mkdir -p "$LOG_DIR" "$OUTPUT_BASE"

export CUBLAS_WORKSPACE_CONFIG=:4096:8

PHASE="${1:-all}"

run_openvla() {
    local ckpt="$1" name="$2"
    echo ">>> OpenVLA: $name — $(date '+%H:%M:%S')"
    PYTHONNOUSERSITE=1 conda run -n openvla_env python -u \
        "$PROJECT_DIR/extractors/extract_7feat_openvla.py" \
        --ckpt "$ckpt" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_BASE/$name" \
        2>&1 | tee "$LOG_DIR/openvla_${name}.log"
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
        2>&1 | tee "$LOG_DIR/groot_${name}.log"
    echo ">>> $name DONE — $(date '+%H:%M:%S')"
}

echo "============================================"
echo "7-Feature Extraction: Phases 2-4"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Phase: $PHASE"
echo "============================================"

START_TIME=$(date +%s)

# === Phase 2a: OpenVLA (2 models, ~15GB each, sequential) ===
if [[ "$PHASE" == "all" || "$PHASE" == "2a" ]]; then
    echo ""
    echo "=== Phase 2a: OpenVLA ==="
    run_openvla "/lambda/nfs/vla/cache/SimplerEnv-OpenVLA/checkpoints/openvla-7b" "openvla-7b-bridge"
    run_openvla "/lambda/nfs/vla/cache/SimplerEnv-OpenVLA/checkpoints/openvla-bridge-sft" "openvla-7b-bridge-ft-200k"
fi

# === Phase 2b: CogACT (3 models, ~15GB each, sequential) ===
if [[ "$PHASE" == "all" || "$PHASE" == "2b" ]]; then
    echo ""
    echo "=== Phase 2b: CogACT ==="
    run_cogact "CogACT/CogACT-Small" "DiT-S" "cogact-small-bridge"
    run_cogact "CogACT/CogACT-Base" "DiT-B" "cogact-base-bridge"
    run_cogact "CogACT/CogACT-Large" "DiT-L" "cogact-large-bridge"
fi

# === Phase 3: SpatialVLA + Pi0 (can run in parallel, ~16GB total) ===
if [[ "$PHASE" == "all" || "$PHASE" == "3a" ]]; then
    echo ""
    echo "=== Phase 3a: SpatialVLA ==="
    run_spatialvla
fi

if [[ "$PHASE" == "all" || "$PHASE" == "3b" ]]; then
    echo ""
    echo "=== Phase 3b: Pi0 ==="
    run_pi0
fi

# === Phase 4: GR00T N1.5 + N1.6 (can run in parallel, ~12-16GB total) ===
if [[ "$PHASE" == "all" || "$PHASE" == "4a" ]]; then
    echo ""
    echo "=== Phase 4a: GR00T N1.5 ==="
    run_groot "n15" "/lambda/nfs/vla/cache/GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2" "groot-n15-bridge"
fi

if [[ "$PHASE" == "all" || "$PHASE" == "4b" ]]; then
    echo ""
    echo "=== Phase 4b: GR00T N1.6 ==="
    run_groot "n16" "/lambda/nfs/vla/cache/GR00T-N1.6-bridge" "groot-n16-bridge"
fi

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
echo ""
echo "============================================"
echo "ALL DONE. Total time: $((ELAPSED/60))m $((ELAPSED%60))s"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
