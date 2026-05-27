#!/usr/bin/env bash
# Download the Qwen2.5-VL-3B-Instruct weights and BridgeData V2 RLDS dataset.
#
# Storage: pass DATA_ROOT (defaults to /dev/shm/vlm4vla). On RAM-backed tmpfs
# the dataset I/O is ~10-25x faster than a typical container overlay; if your
# storage is persistent and large enough, point DATA_ROOT at it instead.
#
# Usage: bash scripts/download_assets.sh [DATA_ROOT]
# Optional env: HF_TOKEN (for huggingface), CHECK_ONLY=1 (just verify presence)

set -euo pipefail

DATA_ROOT="${1:-/dev/shm/vlm4vla}"
mkdir -p "$DATA_ROOT/models" "$DATA_ROOT/data" "$DATA_ROOT/hf_home" "$DATA_ROOT/tmp"

QWEN_DIR="$DATA_ROOT/models/Qwen2.5-VL-3B-Instruct"
BRIDGE_DIR="$DATA_ROOT/data/bridge_orig"

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate vlm4vla

if [ -n "${HF_TOKEN:-}" ]; then
    hf auth login --token "$HF_TOKEN"
fi

if [ -d "$QWEN_DIR" ] && [ -f "$QWEN_DIR/model.safetensors.index.json" ]; then
    echo "[assets] qwen already present: $QWEN_DIR"
else
    echo "[assets] downloading Qwen2.5-VL-3B-Instruct -> $QWEN_DIR"
    HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME="$DATA_ROOT/hf_home" \
        huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir "$QWEN_DIR"
fi

if [ -d "$BRIDGE_DIR/1.0.0" ]; then
    echo "[assets] bridge_orig already present: $BRIDGE_DIR ($(du -sh "$BRIDGE_DIR" | awk '{print $1}'))"
else
    SRC="$DATA_ROOT/data/bridge_dataset"
    echo "[assets] downloading BridgeData V2 RLDS -> $SRC"
    cd "$DATA_ROOT/data"
    # NOTE: --no-parent prevents wget from walking up to the public directory
    # listing and pulling the unrelated raw demos_*.zip (~133 GB, NOT used).
    wget -r -nH --cut-dirs=4 --no-parent --reject="index.html*" \
        https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/
    # OpenVLA's data loader expects the directory to be called bridge_orig
    mv "$SRC" "$BRIDGE_DIR"
fi

echo "[assets] sizes:"
du -sh "$QWEN_DIR" "$BRIDGE_DIR"
echo "[assets] DONE"
