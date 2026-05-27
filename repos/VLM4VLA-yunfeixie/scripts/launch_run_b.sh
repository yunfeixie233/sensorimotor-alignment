#!/usr/bin/env bash
# Launch Run (b) of paper Table 3: Qwen2.5-VL-3B + freeze vision encoder
# on SimplerBridge.  Standalone (does NOT wait for any other run).
#
# Spawns two tmux sessions:
#   run_b_train    -- the actual 8-GPU training
#   run_b_sidecar  -- watches the ckpt dir, converts each DeepSpeed stage-2
#                     save to a single FP32 .pt, rsyncs one to the archive,
#                     and rotates raw DS dirs to keep disk in budget.
#
# Optional env:
#   PORT       -- override the torchrun master port (default 6042)
#
# Usage: bash scripts/launch_run_b.sh

set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$REPO_ROOT/configs/oxe_training/bridge/finetune_qwen25vl-3b_bridge_LOCAL_freezevis.json"
EXP_NAME="bridge_-bs512-lr5e-05-ws1-FCDecoder-latent1-freeze_vision"

bash "$REPO_ROOT/scripts/launch_run.sh" run_b "$CONFIG" "$EXP_NAME"
