#!/usr/bin/env bash
# Launch a VLM4VLA training run (full FT or freeze-vision) + its ckpt-convert sidecar in tmux.
#
# Usage: launch_run.sh <tag> <config_path> <ckpt_dirname>
#   tag           short name used for tmux session names (e.g. run_a, run_b)
#   config_path   absolute path to the LOCAL json config
#   ckpt_dirname  exp dir name that main.py will produce (the last component under
#                 /dev/shm/vlm4vla/ckpts/qwen25vl/bridge_finetune/<DATE>/)

set -u

TAG="${1:?tag required}"
CONFIG="${2:?config path required}"
CKPT_NAME="${3:?ckpt dir name required}"

TODAY=$(date +%Y-%m-%d)
CKPT_DIR="/dev/shm/vlm4vla/ckpts/qwen25vl/bridge_finetune/${TODAY}/${CKPT_NAME}"
ARCHIVE_DIR="/workspace/ckpts_archive/${TAG}"
TRAIN_LOG="/dev/shm/vlm4vla/logs/${TAG}.log"
SIDECAR_LOG="/dev/shm/vlm4vla/tmp/sidecar_${TAG}.log"

mkdir -p /dev/shm/vlm4vla/logs /dev/shm/vlm4vla/tmp "$ARCHIVE_DIR"
: > "$TRAIN_LOG"
: > "$SIDECAR_LOG"

echo "=== launching $TAG ==="
echo "config: $CONFIG"
echo "ckpt dir: $CKPT_DIR"
echo "archive: $ARCHIVE_DIR"

# Training tmux
tmux kill-session -t "${TAG}_train" 2>/dev/null || true
tmux new-session -d -s "${TAG}_train" "
  source /opt/miniforge3/etc/profile.d/conda.sh
  conda activate vlm4vla
  cd /workspace/VLM4VLA
  export WANDB_MODE=offline HF_HOME=/dev/shm/vlm4vla/hf_home TMPDIR=/dev/shm/vlm4vla/tmp
  echo '[train] starting at \$(date) config=$CONFIG' | tee -a '$TRAIN_LOG'
  bash scripts/run.sh '$CONFIG' 2>&1 | tee -a '$TRAIN_LOG'
  echo '[train] ended at \$(date) exit=\$?' | tee -a '$TRAIN_LOG'
  sleep 7200
"

# Sidecar tmux
tmux kill-session -t "${TAG}_sidecar" 2>/dev/null || true
tmux new-session -d -s "${TAG}_sidecar" "
  bash /workspace/VLM4VLA/scripts/ckpt_convert_sidecar.sh \
      '$CKPT_DIR' '$ARCHIVE_DIR' 2 '$SIDECAR_LOG' 2>&1
  echo '[sidecar ${TAG}] exited at \$(date)'
  sleep 7200
"

tmux ls
