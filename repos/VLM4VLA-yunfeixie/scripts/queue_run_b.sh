#!/usr/bin/env bash
# Waits for Run (a) torchrun to exit cleanly, then launches Run (b).
set -u

POLL_INTERVAL=60
WAIT_PATTERN='main.py.*finetune_qwen25vl-3b_bridge_LOCAL\.json'
LAUNCH="/workspace/VLM4VLA/scripts/launch_run.sh"

LOG=/dev/shm/vlm4vla/tmp/queue_run_b.log
mkdir -p "$(dirname "$LOG")"
echo "[$(date -Is)] queue_run_b watcher starting, pattern='$WAIT_PATTERN'" | tee -a "$LOG"

# Wait for run_a's main.py to actually appear (avoid race at startup)
tries=0
while true; do
    if pgrep -f "$WAIT_PATTERN" >/dev/null 2>&1; then
        echo "[$(date -Is)] run_a main.py detected, starting to wait for exit" | tee -a "$LOG"
        break
    fi
    tries=$((tries+1))
    if [ $tries -ge 60 ]; then
        echo "[$(date -Is)] run_a never started after 60 min, giving up" | tee -a "$LOG"
        exit 1
    fi
    sleep $POLL_INTERVAL
done

# Now wait for it to exit
while pgrep -f "$WAIT_PATTERN" >/dev/null 2>&1; do
    sleep $POLL_INTERVAL
done

echo "[$(date -Is)] run_a done; verifying last checkpoint health" | tee -a "$LOG"
sleep 30  # let sidecar settle / ckpt save finish

# Look for any fp32 file in run_a's archive as a basic health check
FP32_COUNT=$(ls /workspace/ckpts_archive/run_a/*.fp32.pt 2>/dev/null | wc -l)
SHM_FP32=$(find /dev/shm/vlm4vla/ckpts -name '*.fp32.pt' 2>/dev/null | wc -l)
echo "[$(date -Is)] run_a post-check: archive fp32=$FP32_COUNT, shm fp32=$SHM_FP32" | tee -a "$LOG"

echo "[$(date -Is)] launching run_b" | tee -a "$LOG"
bash "$LAUNCH" run_b \
    /workspace/VLM4VLA/configs/oxe_training/bridge/finetune_qwen25vl-3b_bridge_LOCAL_freezevis.json \
    bridge_-bs512-lr5e-05-ws1-FCDecoder-latent1-freeze_vision 2>&1 | tee -a "$LOG"
