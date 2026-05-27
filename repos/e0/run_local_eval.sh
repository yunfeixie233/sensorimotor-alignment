export MUJOCO_GL="egl"
export CUDA_VISIBLE_DEVICES=0

#  libero_spatial, libero_object, libero_goal, libero_10, libero_90, all_wo_90, all
TASK_SUITE_NAME="all_wo_90"
MODEL_PREFIX="e0_diff_hybrid_libero"
PORT=8000
REPLAN_STEPS=10

python examples/libero/main.py  --args.model_name="${MODEL_PREFIX}" --args.task_suite_name="${TASK_SUITE_NAME}" --args.port=${PORT} --args.replan_steps=${REPLAN_STEPS}
