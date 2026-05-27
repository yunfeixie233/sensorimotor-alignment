#!/bin/bash
MODEL_PATH="./aux_think_8B"
# Usage: ./run_infer.sh <RESULT_PATH> <GPU_ID_1> <GPU_ID_2> ...
RESULT_PATH=$1
shift
GPU_IDS=("$@")

if [ -z "$RESULT_PATH" ] || [ ${#GPU_IDS[@]} -eq 0 ]; then
    echo "❌ Usage: $0 <RESULT_PATH> <GPU_ID_1> <GPU_ID_2> ..."
    exit 1
fi

START_PORT=8891
mkdir -p logs

for i in "${!GPU_IDS[@]}"; do
    GPU_ID="${GPU_IDS[$i]}"
    PORT=$((START_PORT + GPU_ID))

    echo "🚀 Starting API server on GPU ${GPU_ID}, PORT ${PORT}"

    conda run -n aux_think CUDA_VISIBLE_DEVICES=$GPU_ID nohup python run_api.py \
        --model $MODEL_PATH \
        --port $PORT \
        > logs/api_gpu${GPU_ID}.log 2>&1 &
done

echo "🕐 Waiting for all APIs to become ready..."
check_ready() {
    local port=$1
    for _ in {1..30}; do
        if curl -s -X POST http://localhost:$port/infer -H "Content-Type: application/json" \
            -d '{"prompt": "ping", "images": []}' | grep -q "generated_text"; then
            echo "✅ Port $port is ready."
            return 0
        fi
        sleep 5
    done
    echo "❌ Port $port not responding."
    return 1
}

for GPU_ID in "${GPU_IDS[@]}"; do
    PORT=$((START_PORT + GPU_ID))
    check_ready $PORT || exit 1
done

TOTAL=${#GPU_IDS[@]}
for IDX in "${!GPU_IDS[@]}"; do
    GPU_ID="${GPU_IDS[$IDX]}"
    PORT=$((START_PORT + GPU_ID))
    echo "🧠 Running inference on GPU $GPU_ID, PORT $PORT"

    conda run -n habitat CUDA_VISIBLE_DEVICES=$GPU_ID python infer.py \
        --exp-config VLN_CE/vlnce_baselines/config/r2r_baselines/aux_think_r2r.yaml \
        --split-num $TOTAL \
        --split-id $IDX \
        --visualize \
        --use-api \
        --api-port $PORT \
        --result-path "$RESULT_PATH" \
        > logs/infer_gpu${GPU_ID}.log 2>&1 &
done

wait
echo "🎉 All inference tasks completed."
