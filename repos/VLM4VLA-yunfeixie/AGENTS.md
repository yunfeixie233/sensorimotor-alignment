# AGENTS.md — operational guide for an AI agent

This file is for an AI coding agent (e.g. Cursor, Claude Code) that lands on a fresh
machine and is asked to "launch experiment B" (or A) for this fork of VLM4VLA.
Read this first. Everything you need to launch and monitor is here.

## What this repo is

A fork of [CladernyJorn/VLM4VLA](https://github.com/CladernyJorn/VLM4VLA) configured
to reproduce two specific rows of paper Table 3:

| Run | Vision encoder | `train_text_embedding` | Target SimplerBridge SR (paper) |
|-----|----------------|------------------------|---------------------------------|
| (a) full FT     | trained  | true | **48.00** |
| (b) freeze vis  | **frozen** | true | **23.95** (a -24.05 drop) |

Backbone: Qwen2.5-VL-3B-Instruct. Dataset: BridgeData V2 (RLDS). 8x A100 80GB.
Lightning + DeepSpeed stage-2, bf16. Global batch 512 (8 per GPU x 8 accumulate x 8 GPUs).

The freeze-vision flag flip is the only difference between the two configs:

```
configs/oxe_training/bridge/finetune_qwen25vl-3b_bridge_LOCAL.json           train_vision: true   # Run (a)
configs/oxe_training/bridge/finetune_qwen25vl-3b_bridge_LOCAL_freezevis.json train_vision: false  # Run (b)
```

Behavior in code: `vlm4vla/model/backbone/base_backbone.py:385-406` calls
`self.vision_tower.requires_grad_(False)` when `train_vision` is false; `main.py:89-92`
appends `-freeze_vision` to the experiment name.

For the full background story (paper context, all design decisions, gotchas we hit),
read the companion note (in the agent's workspace if vla_idea is mounted, otherwise skip):
`vla_idea/notes/tabs/2026-04-18_vlm4vla-qwen25vl3b-simplerbridge-stepbystep.md`.

## Pre-flight checks (do these before doing anything else)

```bash
# 1. GPUs: must see 8 with >=80 GB each (A100-80GB or H100)
nvidia-smi --query-gpu=name,memory.total --format=csv

# 2. conda: must exist
ls /opt/miniforge3/etc/profile.d/conda.sh

# 3. tmux: required for the launchers
which tmux

# 4. disk: where will you put 124 GB of data + 7 GB model + ~250 GB checkpoints?
df -h /
df -h /dev/shm
findmnt -D | head
```

If `df -h /` shows the rootfs is small (<200 GB) but `df -h /dev/shm` shows >300 GB,
this is a constrained container - the default scripts target /dev/shm. **Tmpfs
is wiped on container restart.** If you have a real persistent volume mounted,
override the default by exporting `DATA_ROOT=/your/persistent/path` before running
`scripts/download_assets.sh` and edit the local configs to point `data_root_dir`,
`output_root`, `log_root`, `cache_root` at the same root.

If GPUs < 8, do NOT run as-is. The configs assume 8 GPUs to hit global batch 512.
Either find more GPUs or scale `accumulate_grad_batches` proportionally
(`batch_size * accumulate * gpus = 512`), AND set `--gpus N` and `GPUS_PER_NODE=N`
in `scripts/run.sh`.

## End-to-end sequence to launch Run (b)

```bash
# 1. Install env (~3 min on a warm pip cache, ~15 min cold)
bash scripts/setup_env.sh

# 2. Download dataset + model (~1 hour for Bridge at 50 MB/s)
HF_TOKEN=hf_xxxx bash scripts/download_assets.sh

# 3. (STRONGLY recommended) 6-min smoke validates training, ckpt save, FP32 conversion
bash scripts/launch_run.sh smoke \
    configs/oxe_training/bridge/finetune_qwen25vl-3b_bridge_LOCAL_smoketest.json \
    bridge_-bs512-lr5e-05-ws1-FCDecoder-latent1
# wait for the smoke tmux to finish; check the FP32 file landed:
#   /dev/shm/vlm4vla/ckpts/.../*.fp32.pt should be ~15 GB

# 4. Launch Run (b) - spawns 2 tmux sessions: run_b_train + run_b_sidecar
bash scripts/launch_run_b.sh

# 5. Verify it started
tmux ls
nvidia-smi
```

## Monitoring (during the ~2.4-day run)

```bash
tmux ls                              # should see run_b_train, run_b_sidecar
tmux attach -t run_b_train           # live training output (Ctrl-b d to detach)
tmux attach -t run_b_sidecar         # ckpt conversion log

tail -f /dev/shm/vlm4vla/logs/run_b.log               # full training log
tail -f /dev/shm/vlm4vla/tmp/sidecar_run_b.log        # convert/rotate activity

# quick numerical status (latest progress + GPU)
sed -r 's/\x1B\[[0-9;]*[a-zA-Z]//g' /dev/shm/vlm4vla/logs/run_b.log \
  | grep -oE 'Epoch [0-9]+/[0-9]+ +[0-9]+/[0-9]+' | tail -1
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
```

Healthy signals (after the first ~2 min of dataset-statistics warmup):
- All 8 GPUs at 80-100% utilization, 50-62 GB VRAM each, 250-410 W power.
- Steady ~1.2 micro-batches/s (which is ~7 s per optimizer step).
- `train_loss` decreasing from ~0.5 toward ~0.1 within the first few hundred steps.
- Sidecar prints `convert OK, fp32 size=15G` after each save (every 5000 opt steps).

Unhealthy signals:
- `0` training procs and a `Traceback` in the log -> see "Common failures" below.
- `/dev/shm` close to 503 GB -> sidecar isn't pruning; `ls /dev/shm/vlm4vla/ckpts/qwen25vl/bridge_finetune/*/*/`.
- `df -h /` close to 32 GB -> overlay almost full; safe to `pip cache purge`.
- All 8 GPUs at 0% util but memory still allocated -> deadlock; kill the run, check NCCL env.

## ETA math

- Throughput is reproducible at ~1.21-1.23 micro-batches/s on 8x A100-80GB.
- 1 optimizer step = `accumulate_grad_batches=8` micro-batches = ~7 s.
- With shipped `max_epochs=5` AND `max_steps=50000`, Lightning stops at the earlier
  limit. At our throughput one epoch is ~6,673 opt steps, so 5 epochs cap at ~33,366
  opt steps **before** max_steps=50000 fires. -> ~65 hours wall-clock per run (~2.7 days).
- For a strict paper-spec 50k steps (~3.9 days), edit the LOCAL config:
  `"max_epochs": -1` (then max_steps=50000 governs).

## Common failures and fixes

1. **`ModuleNotFoundError: No module named 'qwen_vl_utils' / 'decord' / 'accelerate'`**
   `setup_env.sh` should install all three; if it didn't, `pip install qwen-vl-utils decord accelerate`.

2. **`ImportError: ... flash_attn ... not installed`**
   The Qwen2.5-VL forward pass requires it. Use the prebuilt wheel:
   `pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`

3. **`huggingface_hub` version conflict / `auth login` fails / `cached_download` import errors**
   `pip install 'huggingface_hub<1.0,>=0.34.0'` (transformers 4.57 requires this; 1.x breaks it).

4. **`from robovlms.utils.zero_to_fp32 import ...` ImportError when running upstream `transform_ckpt.py`**
   Don't use it. Use `scripts/convert_ckpt_standalone.py <ds_dir> <fp32.pt>` instead.

5. **wget downloads ~133 GB extra `demos_8_17.zip`**
   You missed `--no-parent`. `scripts/download_assets.sh` has it; if you wgetted manually, kill it and use the script.

6. **`dlimp_openvla` directory inside `openvla/` is empty**
   It's a separately maintained repo. `setup_env.sh` clones `https://github.com/moojink/dlimp_openvla.git` and pip-installs it.

7. **Disk fills mid-training**
   Sidecar should be pruning. Verify it's alive: `pgrep -af ckpt_convert_sidecar.sh`. If dead, restart with:
   ```
   tmux new-session -d -s run_b_sidecar 'bash scripts/ckpt_convert_sidecar.sh \
     "$(echo /dev/shm/vlm4vla/ckpts/qwen25vl/bridge_finetune/*/bridge_*-freeze_vision)" \
     /workspace/ckpts_archive/run_b 2 /dev/shm/vlm4vla/tmp/sidecar_run_b.log'
   ```

8. **Container restart wipes /dev/shm**
   You lose the dataset, model, and all in-flight ckpts. Recovery:
   - Re-run `download_assets.sh`.
   - If `/workspace/ckpts_archive/run_b/*.fp32.pt` survived, you can warm-start (load
     model weights only, fresh optimizer, restart from step 0). True resume requires
     the raw DeepSpeed stage-2 directory (~50 GB) which only lives on /dev/shm.
   - Pragmatic: just re-launch from scratch; ~2.4 days is shorter than productionizing
     persistent ckpt sync.

9. **NCCL hang (all GPUs at 0% util after init)**
   Often network-related. Inspect `scripts/run.sh` env: `NCCL_SOCKET_IFNAME=eth0` is
   our default; if your interface differs (`ip -br link`), edit accordingly. Also try
   `NCCL_IB_DISABLE=1` for single-node setups without InfiniBand.

10. **`exitcode: 1` with `<NO_OTHER_FAILURES>`**
    The traceback is in the log somewhere; grep for the actual exception:
    `grep -E 'Error|Traceback|raise' /dev/shm/vlm4vla/logs/run_b.log | grep -v 'CropAndResize\|TensorRT\|PredictCost\|cuDNN\|cuBLAS\|cuFFT\|Running replicate' | head`.
    The TF "CostEstimator/CropAndResize" warnings are noise, ignore them.

## After training: producing FP32 checkpoints for eval

The sidecar already does this every 5000 steps. The output FP32 files live at
`/dev/shm/vlm4vla/ckpts/qwen25vl/bridge_finetune/<DATE>/bridge_-bs512-lr5e-05-ws1-FCDecoder-latent1-freeze_vision/*.fp32.pt`
plus the matching `*-project.json` in the corresponding `logs/` dir. The eval harness
under `eval/simpler/` consumes both. If you need to convert manually:

```bash
python scripts/convert_ckpt_standalone.py \
    /dev/shm/vlm4vla/ckpts/.../stepstep=0010000.ckpt \
    /dev/shm/vlm4vla/ckpts/.../stepstep=0010000.fp32.pt
```

## Decision rules for the agent

- If the user says "launch experiment B" with no other context, do steps 1-2-(3)-4
  above. Confirm GPU count and disk before downloading.
- If a previous run is already in tmux (`tmux ls` shows `run_a_train` or `run_b_train`),
  do NOT launch another - it will OOM or fail to acquire CUDA. Tell the user.
- If the dataset is already at `/dev/shm/vlm4vla/data/bridge_orig` and Qwen at
  `/dev/shm/vlm4vla/models/Qwen2.5-VL-3B-Instruct`, skip the download step.
- Always run the smoke test first on a fresh box. If it errors, do not start the full
  run - debug the smoke first. Once it produces a 15 GB FP32 file, the full pipeline is
  validated end-to-end.
- For a multi-day run, set up the monitor tmux from the upstream sequence so the user
  can poke head and see steady GPU util without re-running nvidia-smi.

## File map (key paths)

```
configs/oxe_training/bridge/
  finetune_qwen25vl-3b_bridge_LOCAL.json            # Run (a) config
  finetune_qwen25vl-3b_bridge_LOCAL_freezevis.json  # Run (b) config
  finetune_qwen25vl-3b_bridge_LOCAL_smoketest.json  # 45-step smoke
scripts/
  setup_env.sh              # one-shot env install
  download_assets.sh        # qwen + bridge download (with --no-parent fix)
  launch_run_a.sh           # entrypoint: Run (a)
  launch_run_b.sh           # entrypoint: Run (b)
  launch_run.sh             # generic two-tmux launcher
  queue_run_b.sh            # auto-launch (b) when (a) exits (same box only)
  ckpt_convert_sidecar.sh   # watch + convert + prune
  convert_ckpt_standalone.py # clean DS-stage-2 -> FP32 .pt
  run.sh                    # upstream torchrun wrapper (8 GPUs)
main.py                     # patched: HF_HOME removed, ModelCheckpoint reads cadence
vlm4vla/model/backbone/base_backbone.py  # vision freeze logic at L385-406
vlm4vla/utils/zero_to_fp32.py            # DeepSpeed checkpoint merging
```
