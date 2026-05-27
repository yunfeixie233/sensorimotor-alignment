#!/bin/bash

policy_name=MM-ACT
task_name="adjust_bottle"
task_config="demo_randomized"
ckpt_setting="MM-ACT-text-image-action" #add ckpt name before perallel_test
seed=1
gpu_id=0
test_num=100
parallel_num=1
index=0
timestamp=$(date +"%Y-%m-%d-%H-%M-%S")
echo "start time: ${timestamp}"

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

PYTHONWARNINGS=ignore::UserWarning python script/eval_policy.py \
    --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --test_num ${test_num} \
    --parallel_num ${parallel_num} \
    --index ${index} \
    --timestamp ${timestamp}