#!/usr/bin/env bash
set -euo pipefail

# Wait for bridge download wget to exit, then rename bridge_dataset -> bridge_orig,
# then launch the 50-step smoke training.

WGET_PID_FILE=/dev/shm/vlm4vla/tmp/wget.pid
DATA_DIR=/dev/shm/vlm4vla/data
SRC=$DATA_DIR/bridge_dataset
DST=$DATA_DIR/bridge_orig
LOG=/dev/shm/vlm4vla/tmp/smoke_waiter.log

mkdir -p /dev/shm/vlm4vla/tmp
echo "[$(date -Is)] waiter starting" | tee -a $LOG

while true; do
    running=$(pgrep -af 'wget -r' | grep -v 'bash' | grep -v grep || true)
    if [ -z "$running" ]; then
        echo "[$(date -Is)] wget no longer running" | tee -a $LOG
        break
    fi
    size=$(du -sh $SRC 2>/dev/null | awk '{print $1}')
    echo "[$(date -Is)] waiting for wget, bridge_dataset=$size" | tee -a $LOG
    sleep 60
done

if [ ! -d $DST ]; then
    echo "[$(date -Is)] renaming bridge_dataset -> bridge_orig" | tee -a $LOG
    mv $SRC $DST
fi

echo "[$(date -Is)] final dataset size:" | tee -a $LOG
du -sh $DST | tee -a $LOG
ls $DST | head -5 | tee -a $LOG

echo "[$(date -Is)] launching 50-step smoke test" | tee -a $LOG

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate vlm4vla
cd /workspace/VLM4VLA
export WANDB_MODE=offline
export HF_HOME=/dev/shm/vlm4vla/hf_home
export TMPDIR=/dev/shm/vlm4vla/tmp
export HF_DATASETS_CACHE=/dev/shm/vlm4vla/hf_home/datasets

bash scripts/run.sh configs/oxe_training/bridge/finetune_qwen25vl-3b_bridge_LOCAL_smoketest.json 2>&1 | tee -a /dev/shm/vlm4vla/logs/smoke.log
