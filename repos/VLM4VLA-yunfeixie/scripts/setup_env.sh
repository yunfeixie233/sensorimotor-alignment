#!/usr/bin/env bash
# One-shot env setup for VLM4VLA training (Qwen2.5-VL-3B / SimplerBridge).
# Creates a python 3.10 conda env, installs all training dependencies,
# and verifies the imports.
#
# Assumes /opt/miniforge3 is installed.
# Usage:  bash scripts/setup_env.sh

set -euo pipefail

ENV_NAME="${ENV_NAME:-vlm4vla}"
DLIMP_DIR="${DLIMP_DIR:-/workspace/repos/dlimp_openvla}"

source /opt/miniforge3/etc/profile.d/conda.sh

if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "[setup] creating conda env $ENV_NAME (python 3.10)"
    conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "[setup] repo root: $REPO_ROOT"

echo "[setup] installing vlm4vla (pyproject deps)"
pip install -e "$REPO_ROOT"

echo "[setup] installing openvla fork (bundled in repo)"
pip install -e "$REPO_ROOT/openvla"

echo "[setup] cloning + installing dlimp_openvla (separate repo)"
mkdir -p "$(dirname "$DLIMP_DIR")"
if [ ! -d "$DLIMP_DIR/.git" ]; then
    git clone https://github.com/moojink/dlimp_openvla.git "$DLIMP_DIR"
fi
pip install -e "$DLIMP_DIR"

echo "[setup] runtime extras (hydra, bitsandbytes, deepspeed, qwen-vl-utils, decord, accelerate)"
pip install -U hydra-core
pip install bitsandbytes pretty_errors deepspeed qwen-vl-utils decord accelerate

echo "[setup] downgrading huggingface_hub to satisfy transformers 4.57"
pip install 'huggingface_hub<1.0,>=0.34.0' 'huggingface_hub[hf_transfer]' 'huggingface_hub[cli]'

echo "[setup] flash-attn (prebuilt wheel for torch 2.6 / cu12 / py310)"
WHL='https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
pip install "$WHL"

echo "[setup] verify imports"
python - <<'PY'
import torch, transformers, deepspeed, dlimp, flash_attn, accelerate
from vlm4vla.train.base_trainer import BaseTrainer
print('torch       :', torch.__version__, 'cuda', torch.version.cuda, 'gpus=', torch.cuda.device_count())
print('transformers:', transformers.__version__)
print('deepspeed   :', deepspeed.__version__)
print('flash_attn  :', flash_attn.__version__)
print('accelerate  :', accelerate.__version__)
print('imports OK')
PY

echo "[setup] DONE"
