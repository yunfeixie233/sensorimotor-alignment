#!/bin/bash

# 接收参数
export MODEL=$1
export IDM=$2
export WORKER_PORT=$3
export CUDA_VISIBLE_DEVICES=$4

# 设置单卡运行必要的环境变量
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

conda activate vidar

echo "Starting Standalone Worker on Port $WORKER_PORT Device $CUDA_VISIBLE_DEVICES"
exec uvicorn server.stand_worker:api --host localhost --port $WORKER_PORT --workers 1

