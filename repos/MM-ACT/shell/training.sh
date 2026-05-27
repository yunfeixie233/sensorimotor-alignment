echo $MASTER_ADDR
echo $MASTER_PORT
echo $NODE_RANK
accelerate launch \
  --config_file accelerate_configs/2_node_8_gpus_deepspeed_zero2.yaml \
  --machine_rank "${NODE_RANK}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  training/train_mmact_robotwin_mix.py \
  config=configs/mmact_robotwin_mix.yaml