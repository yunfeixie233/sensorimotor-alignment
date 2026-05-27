"""Parallel sweep launcher for SimplerBridge evaluation.

For each (checkpoint, execute_step) cell this spawns a `scripts/bridge.bash`
process on a free GPU. Up to NGPU cells run concurrently.  Intended to
reproduce Table-3 numbers by selecting the best (ckpt, exec_step) pair.


===========================================================================
PORTING TO ANOTHER MACHINE / ANOTHER CHECKPOINT GROUP
===========================================================================

Everything is controlled by CLI flags (or env vars with the same name).
Copy this file as-is; only the *paths* change.

Prerequisites on the new machine:
  1. conda env `vlm4vla_eval` created and with these pip-installed:
       - this repo (`pip install -e .`)
       - `openvla/` subdir (`pip install -e ./openvla`)
       - dlimp_openvla (`pip install -e /path/to/dlimp_openvla`)
       - `bitsandbytes pretty_errors deepspeed qwen-vl-utils decord accelerate`
       - `'huggingface_hub<1.0,>=0.34.0'`
       - flash-attn wheel matching your torch/CUDA
       - Then add SimplerEnv stack (see README §Installation):
           `simpler_env`, `ManiSkill2_real2sim`, `mediapy`, `numpy==1.24.4`,
           `opencv-python<4.11`, `setuptools<81` (sapien needs pkg_resources)
  2. `<REPO>/real_inpainting` is a (sym)link to
       `<SimplerEnv>/ManiSkill2_real2sim/data/real_inpainting`
     because scripts/bridge.bash hard-codes that relative path.
  3. `configs/data/oxe_dataset_stats/dataset_statistics_bridge.json` is present.
  4. Each ckpt directory contains:
       - one `*.pt` per training step (an FP32 state dict, saved by the
         sidecar or by `scripts/convert_ckpt_standalone.py`)
       - exactly one `*-project.json` (the config snapshot from `main.py`)

Directory layout expected by this launcher:

    BASE_PATH/
        some-run-name-project.json           # 1 config JSON
        stepstep=0005000.fp32.pt             # any number of .pt files
        stepstep=0010000.fp32.pt
        ...
        stepstep=0050000.fp32.pt

What gets written (safe to run in parallel, no collisions between cells):

    BASE_PATH/
        execute_step_<N>/<step>-eval/...     # videos + action pngs + JSON
        sweep_logs/
            ckpt=<stem>__exec=<N>__gpu=<g>.log

Usage examples:

    # default: sweep all .pt in /workspace/ckpts_archive/run_b_all over
    # execute_step in {4,2,1} across 8 GPUs:
    python eval/simpler/sweep_parallel_bridge.py

    # explicit args for a different ckpt group on another box:
    python eval/simpler/sweep_parallel_bridge.py \
        --base-path /data/ckpts/run_c_all \
        --ngpu 8 --exec-steps 4,2,1

    # quick sanity sweep: only one ckpt, only exec=1, 1 GPU:
    python eval/simpler/sweep_parallel_bridge.py \
        --base-path /data/ckpts/foo \
        --step-filter stepstep=0050000 --exec-steps 1 --ngpu 1

    # only evaluate late checkpoints (often best):
    python eval/simpler/sweep_parallel_bridge.py \
        --base-path /data/ckpts/foo \
        --step-filter 004 --exec-steps 4,2,1    # matches 40k/45k

Overhead / wall time:
    - Cold start per cell: ~30 s to load Qwen2.5-VL-3B bf16 from /workspace.
    - 4 SimplerBridge tasks per cell: ~17 min on 1 A100 when GPU is the
      sole occupant, slightly more (~22-25 min) under 8x concurrency due to
      shared memory bandwidth and CPU-side sim contention.
    - Total for 10 ckpts x 3 exec_steps on 8x A100: ~90-100 min.
    - VRAM per process: ~9 GB; RAM per process: ~20 GB transient during load.

Run this under tmux so the orchestration survives disconnects:
    tmux new-session -d -s sweep "... python eval/simpler/sweep_parallel_bridge.py ..."
"""

import argparse
import itertools
import os
import subprocess
import sys
import time


def discover_cells(base_path: str, exec_steps, step_filter: str):
    json_candidates = [f for f in os.listdir(base_path) if f.endswith(".json")]
    if not json_candidates:
        raise FileNotFoundError(
            f"No *.json config found in {base_path}. "
            "Copy the training run's '*-project.json' into this directory."
        )
    if len(json_candidates) > 1:
        raise RuntimeError(
            f"Multiple .json files in {base_path}: {json_candidates}. "
            "Leave exactly one project config."
        )
    cfg_path = os.path.join(base_path, json_candidates[0])

    ckpts = sorted(
        os.path.join(base_path, f)
        for f in os.listdir(base_path)
        if f.endswith(".pt") and (not step_filter or step_filter in f)
    )
    if not ckpts:
        raise FileNotFoundError(
            f"No .pt files in {base_path} matching STEP_FILTER='{step_filter}'."
        )

    cells = [(c, es) for c, es in itertools.product(ckpts, exec_steps)]
    return cfg_path, ckpts, cells


def main():
    parser = argparse.ArgumentParser(
        description="Parallel SimplerBridge sweep over (ckpt, execute_step)."
    )
    parser.add_argument(
        "--base-path",
        default=os.environ.get("BASE_PATH", "/workspace/ckpts_archive/run_b_all"),
        help="Directory containing *.pt checkpoints and one *-project.json",
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        default=int(os.environ.get("NGPU", "8")),
        help="Max concurrent GPUs to use (one cell per GPU).",
    )
    parser.add_argument(
        "--exec-steps",
        default=os.environ.get("EXEC_STEPS", "4,2,1"),
        help="Comma-separated execute_step values to sweep.",
    )
    parser.add_argument(
        "--step-filter",
        default=os.environ.get("STEP_FILTER", ""),
        help="Optional substring required in .pt filenames (e.g. 'stepstep=0050000').",
    )
    parser.add_argument(
        "--repo-root",
        default=os.environ.get("REPO_ROOT", "/workspace/VLM4VLA"),
        help="VLM4VLA repo root (must contain scripts/bridge.bash).",
    )
    parser.add_argument(
        "--poll-secs",
        type=int,
        default=5,
        help="Seconds between scheduler wake-ups.",
    )
    args = parser.parse_args()

    exec_steps = [int(s) for s in args.exec_steps.split(",") if s.strip()]

    bridge_script = os.path.join(args.repo_root, "scripts", "bridge.bash")
    if not os.path.isfile(bridge_script):
        raise FileNotFoundError(f"scripts/bridge.bash not found at {bridge_script}")

    cfg_path, ckpts, cells = discover_cells(
        args.base_path, exec_steps, args.step_filter
    )
    log_root = os.path.join(args.base_path, "sweep_logs")
    os.makedirs(log_root, exist_ok=True)

    print(f"[sweep] repo_root   = {args.repo_root}")
    print(f"[sweep] base_path   = {args.base_path}")
    print(f"[sweep] config      = {cfg_path}")
    print(f"[sweep] ngpu        = {args.ngpu}")
    print(f"[sweep] exec_steps  = {exec_steps}")
    print(f"[sweep] step_filter = '{args.step_filter}'")
    print(f"[sweep] ckpts       = {len(ckpts)}, cells = {len(cells)}")
    for c in ckpts:
        print(f"[sweep]   ckpt: {os.path.basename(c)}")
    sys.stdout.flush()

    # gpu_id -> (Popen, log_fp, t_start, ckpt_stem, exec_step)
    running: dict = {}
    queue = list(cells)
    done = 0
    t_start = time.time()

    def free_gpu():
        busy = set(running.keys())
        for g in range(args.ngpu):
            if g not in busy:
                return g
        return None

    def spawn(gpu: int, ckpt: str, es: int):
        stem = os.path.splitext(os.path.basename(ckpt))[0]
        log_file = os.path.join(
            log_root, f"ckpt={stem}__exec={es}__gpu={gpu}.log"
        )
        lf = open(log_file, "w", buffering=1)
        cmd = ["bash", bridge_script, ckpt, cfg_path, str(es), str(gpu)]
        env = os.environ.copy()
        env["TF_CPP_MIN_LOG_LEVEL"] = env.get("TF_CPP_MIN_LOG_LEVEL", "2")
        p = subprocess.Popen(
            cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=args.repo_root
        )
        running[gpu] = (p, lf, time.time(), stem, es)
        elapsed = (time.time() - t_start) / 60
        print(
            f"[sweep t={elapsed:6.1f}m] start gpu={gpu} ckpt={stem} exec={es} "
            f"-> {log_file}",
            flush=True,
        )

    while queue or running:
        while queue and free_gpu() is not None:
            ckpt, es = queue.pop(0)
            spawn(free_gpu(), ckpt, es)

        time.sleep(args.poll_secs)

        for gpu, (p, lf, tstart, stem, es) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                lf.close()
                dur = (time.time() - tstart) / 60
                elapsed = (time.time() - t_start) / 60
                done += 1
                status = "ok" if rc == 0 else f"FAIL(rc={rc})"
                print(
                    f"[sweep t={elapsed:6.1f}m] done gpu={gpu} ckpt={stem} "
                    f"exec={es} dur={dur:5.1f}m {status}   ({done}/{len(cells)})",
                    flush=True,
                )
                del running[gpu]

    total = (time.time() - t_start) / 60
    print(f"[sweep] ALL DONE. {len(cells)} cells in {total:.1f} min")


if __name__ == "__main__":
    main()
