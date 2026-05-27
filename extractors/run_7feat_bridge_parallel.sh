#!/bin/bash
# Parallel 7-feature extraction for BridgeV2 filtered (N=3245)
# 40GB A100 VRAM budget. Models run in parallel groups.
#
# Schedule (from DROID experience):
#   Group A: CogACT-base + CogACT-large (15GB each = 30GB) — sequential forced
#   Group B: SpatialVLA (8GB) + Pi0 (8GB) = 16GB — parallel
#   Group C: GR00T-N1.5 (8GB) + GR00T-N1.6 (8GB) = 16GB — parallel
#   Group D: StarVLA x4 (6GB each = 24GB) — parallel
#
# Already done: Qwen-GR00T-Bridge, cogact-small-bridge

set -uo pipefail

PROJECT_DIR="/home/ubuntu/vla/cknna_project"
DATA_DIR="/lambda/nfs/vla/cache/cknna_data_store/cknna_data_bridge"
OUTPUT_BASE="/lambda/nfs/vla/cache/cknna_data_store/cknna_data_bridge_7feat"
LOG_DIR="/lambda/nfs/vla/cache/cknna_data_store/logs/7feat_bridge"
CKPT_BASE="/lambda/nfs/vla/cache/starVLA/playground/Pretrained_models"

export CUBLAS_WORKSPACE_CONFIG=:4096:8

skip_if_done() {
    local name="$1"
    local f="$OUTPUT_BASE/$name/feats_A.pt"
    if [ -f "$f" ]; then
        local n
        n=$(/lambda/nfs/vla/conda/envs/starVLA/bin/python -c "import torch; print(torch.load('$f', weights_only=True).shape[0])" 2>/dev/null)
        if [ "$n" = "3245" ]; then
            echo "SKIP $name (already done, N=$n)"
            return 0
        fi
    fi
    return 1
}

echo "============================================"
echo "Parallel 7-Feature Extraction: BridgeV2 (N=3245)"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

START_TIME=$(date +%s)

# === Group A: CogACT-base + CogACT-large (sequential, ~15GB each) ===
echo ""
echo "=== Group A: CogACT (base + large, sequential) ==="
for model_dit_name in "CogACT/CogACT-Base DiT-B cogact-base-bridge" "CogACT/CogACT-Large DiT-L cogact-large-bridge"; do
    set -- $model_dit_name
    ckpt="$1" dit="$2" name="$3"
    if skip_if_done "$name"; then continue; fi
    echo ">>> CogACT: $name ($dit) — $(date '+%H:%M:%S')"
    PYTHONNOUSERSITE=1 COGACT_ROOT="$PROJECT_DIR/repos/CogACT" \
    conda run -n cogact python -u \
        "$PROJECT_DIR/extractors/extract_7feat_cogact.py" \
        --ckpt "$ckpt" --action_model_type "$dit" \
        --data_dir "$DATA_DIR" --output_dir "$OUTPUT_BASE/$name" \
        2>&1 | tee "$LOG_DIR/par_${name}.log"
    echo ">>> $name DONE — $(date '+%H:%M:%S')"
done

# === Group B: SpatialVLA + Pi0 (parallel, ~16GB total) ===
echo ""
echo "=== Group B: SpatialVLA + Pi0 (parallel) ==="
(
    if ! skip_if_done "spatialvla-sft-bridge"; then
        echo ">>> SpatialVLA — $(date '+%H:%M:%S')"
        PYTHONNOUSERSITE=1 conda run -n spatialvla_env python -u \
            "$PROJECT_DIR/extractors/extract_7feat_spatialvla.py" \
            --ckpt /lambda/nfs/vla/cache/spatialvla-bridge \
            --data_dir "$DATA_DIR" --output_dir "$OUTPUT_BASE/spatialvla-sft-bridge" \
            2>&1 | tee "$LOG_DIR/par_spatialvla.log"
        echo ">>> SpatialVLA DONE — $(date '+%H:%M:%S')"
    fi
) &
PID_SPATIAL=$!

(
    if ! skip_if_done "pi0-lerobot-bridge"; then
        echo ">>> Pi0 — $(date '+%H:%M:%S')"
        PYTHONNOUSERSITE=1 conda run -n pi0fast_env python -u \
            "$PROJECT_DIR/extractors/extract_7feat_pi0.py" \
            --ckpt_path HaomingSong/lerobot-pi0-bridge \
            --data_dir "$DATA_DIR" --output_dir "$OUTPUT_BASE/pi0-lerobot-bridge" \
            2>&1 | tee "$LOG_DIR/par_pi0.log"
        echo ">>> Pi0 DONE — $(date '+%H:%M:%S')"
    fi
) &
PID_PI0=$!

wait $PID_SPATIAL $PID_PI0
echo "Group B done."

# === Group C: GR00T N1.5 + N1.6 (parallel, ~16GB total) ===
echo ""
echo "=== Group C: GR00T N1.5 + N1.6 (parallel) ==="
(
    if ! skip_if_done "groot-n15-bridge"; then
        echo ">>> GR00T N1.5 — $(date '+%H:%M:%S')"
        export PYTHONNOUSERSITE=1
        export ISAAC_GROOT_DIR="$PROJECT_DIR/repos/Isaac-GR00T-N15"
        conda run -n groot_libero python -u \
            "$PROJECT_DIR/extractors/extract_7feat_groot.py" \
            --version n15 --ckpt /lambda/nfs/vla/cache/GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2 \
            --data_dir "$DATA_DIR" --output_dir "$OUTPUT_BASE/groot-n15-bridge" \
            2>&1 | tee "$LOG_DIR/par_groot_n15.log"
        echo ">>> GR00T N1.5 DONE — $(date '+%H:%M:%S')"
    fi
) &
PID_N15=$!

(
    if ! skip_if_done "groot-n16-bridge"; then
        echo ">>> GR00T N1.6 — $(date '+%H:%M:%S')"
        export PYTHONNOUSERSITE=1
        export ISAAC_GROOT_N16_DIR="$PROJECT_DIR/repos/Isaac-GR00T-N16"
        conda run -n groot_libero python -u \
            "$PROJECT_DIR/extractors/extract_7feat_groot.py" \
            --version n16 --ckpt /lambda/nfs/vla/cache/GR00T-N1.6-bridge \
            --data_dir "$DATA_DIR" --output_dir "$OUTPUT_BASE/groot-n16-bridge" \
            2>&1 | tee "$LOG_DIR/par_groot_n16.log"
        echo ">>> GR00T N1.6 DONE — $(date '+%H:%M:%S')"
    fi
) &
PID_N16=$!

wait $PID_N15 $PID_N16
echo "Group C done."

# === Group D: StarVLA x4 remaining (parallel, ~6GB each = 24GB) ===
echo ""
echo "=== Group D: StarVLA x4 (parallel) ==="
STARVLA_MODELS=(
    "Qwen-GR00T-Bridge-RT-1:$CKPT_BASE/Qwen-GR00T-Bridge-RT-1/checkpoints/steps_30000_pytorch_model.pt"
    "Qwen-OFT-Bridge-RT-1:$CKPT_BASE/Qwen-OFT-Bridge-RT-1/checkpoints/steps_10000_pytorch_model.pt"
    "Qwen3VL-GR00T-Bridge-RT-1:$CKPT_BASE/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt"
    "Qwen3VL-OFT-Bridge-RT-1:$CKPT_BASE/Qwen3VL-OFT-Bridge-RT-1/checkpoints/steps_5000_pytorch_model.pt"
)

PIDS=()
for entry in "${STARVLA_MODELS[@]}"; do
    name="${entry%%:*}"
    ckpt="${entry#*:}"
    if skip_if_done "$name"; then continue; fi
    (
        echo ">>> StarVLA: $name — $(date '+%H:%M:%S')"
        PYTHONNOUSERSITE=1 STARVLA_ROOT="$PROJECT_DIR/repos/starVLA" \
        conda run -n starVLA python -u \
            "$PROJECT_DIR/extractors/extract_7feat_starvla.py" \
            --ckpt_path "$ckpt" \
            --data_dir "$DATA_DIR" --output_dir "$OUTPUT_BASE/$name" \
            2>&1 | tee "$LOG_DIR/par_${name}.log"
        echo ">>> $name DONE — $(date '+%H:%M:%S')"
    ) &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
    wait $pid
done
echo "Group D done."

# === Summary ===
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
echo ""
echo "============================================"
echo "ALL DONE. Total time: $((ELAPSED/60))m $((ELAPSED%60))s"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "=== Output summary ==="
for d in "$OUTPUT_BASE"/*/; do
    name=$(basename "$d")
    f="$d/feats_A.pt"
    if [ -f "$f" ]; then
        n=$(/lambda/nfs/vla/conda/envs/starVLA/bin/python -c "import torch; print(torch.load('$f', weights_only=True).shape[0])" 2>/dev/null)
        echo "  $name: N=$n"
    else
        echo "  $name: MISSING"
    fi
done
