#!/usr/bin/env bash
# Launch Run (a) of paper Table 3: Qwen2.5-VL-3B fully finetuned (vision + text
# embeddings + LLM) on SimplerBridge.  Standalone.
#
# Spawns:
#   run_a_train, run_a_sidecar  (see launch_run.sh for tmux semantics)
#
# Usage: bash scripts/launch_run_a.sh

set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$REPO_ROOT/configs/oxe_training/bridge/finetune_qwen25vl-3b_bridge_LOCAL.json"
EXP_NAME="bridge_-bs512-lr5e-05-ws1-FCDecoder-latent1"

bash "$REPO_ROOT/scripts/launch_run.sh" run_a "$CONFIG" "$EXP_NAME"
