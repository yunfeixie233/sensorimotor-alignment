#!/bin/bash

policy_name=pick_place_agent
task_name=place_empty_cup
task_config=agent_config
ckpt_setting=none
seed=0
gpu_id=0

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

export PYTHONPATH=$(pwd):$PYTHONPATH
cd ../.. # move to root
PYTHONWARNINGS=ignore::UserWarning
python3 script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} 
