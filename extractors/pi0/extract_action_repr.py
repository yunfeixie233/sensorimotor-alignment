"""
Phase 2b: Extract action representations (feats_action) from Pi0 lerobot for CKNNA.

Runs the full denoising loop (select_action), hooks the expert stream's final
norm (gemma_expert.model.norm) to capture the post-norm output at each step,
takes the LAST denoising step's output, slices the action portion
(last n_action_steps tokens), and mean-pools.

Architecture: PaliGemma (SigLIP vision + Gemma LM, hidden_size=2048)
              + Gemma expert (hidden_size=1024)
Extraction point: Hook on gemma_expert.model.norm at final denoising step
Pooling: mean-pool across action tokens in the expert suffix output

Requires: conda env pi0fast_env (transformers==4.53.3 custom fork)

Usage:
    python extract_action_repr_pi0_lerobot.py \
        --ckpt_path HaomingSong/lerobot-pi0-bridge \
        --data_dir ./cknna_data \
        --output_dir ./cknna_data/pi0-lerobot-bridge \
        --seed 42
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

STATE_DIM_RAW = 8
PAD_INDEX = 6


def reconstruct_8d_state(feats_B_7d):
    """Insert pad=0 at index 6 to recover 8D raw state from 7D feats_B."""
    n = feats_B_7d.shape[0]
    state_8d = np.zeros((n, STATE_DIM_RAW), dtype=np.float32)
    state_8d[:, :PAD_INDEX] = feats_B_7d[:, :PAD_INDEX]
    state_8d[:, PAD_INDEX + 1:] = feats_B_7d[:, PAD_INDEX:]
    return state_8d


def resolve_expert_norm_module(model):
    """Resolve the expert norm module for the forward hook.

    The expert norm is at the end of the expert GemmaModel, capturing
    the expert's POST-norm output during denoising steps.
    """
    candidates = [
        "paligemma_with_expert.gemma_expert.model.norm",
    ]
    for path in candidates:
        obj = model
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            print(f"  Expert hook path resolved: {path}")
            return obj
    raise AttributeError(
        f"Cannot resolve expert norm module. Tried: {candidates}. "
        f"Check the checkpoint's model structure."
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 2b: Extract Pi0 action representations.")
    parser.add_argument("--ckpt_path", type=str, default="HaomingSong/lerobot-pi0-bridge",
                        help="HuggingFace model ID or local path to Pi0 checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Phase 1 data directory (images/, metadata.json, feats_B.pt)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save feats_action.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    feats_B = torch.load(os.path.join(args.data_dir, "feats_B.pt"), weights_only=True)
    state_8d_all = reconstruct_8d_state(feats_B.numpy())

    # ---- Load Pi0 via checkpoint's bundled code + compat shim ----
    print(f"Loading Pi0 from: {args.ckpt_path}")

    helpers_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_helpers")
    sys.path.insert(0, helpers_dir)
    from lerobot_common_shim import install_shim
    install_shim()

    from huggingface_hub import snapshot_download
    local_ckpt = snapshot_download(args.ckpt_path)

    sys.path.insert(0, local_ckpt)
    from modeling_pi0 import PI0Policy

    if "predict_action_chunk" in getattr(PI0Policy, "__abstractmethods__", set()):
        PI0Policy.predict_action_chunk = lambda self, batch, **kw: None
        PI0Policy.__abstractmethods__ = PI0Policy.__abstractmethods__ - {"predict_action_chunk"}

    policy = PI0Policy.from_pretrained(local_ckpt)
    policy.to(args.device)
    policy.eval()
    model = policy.model

    # ---- Resolve expert hidden size ----
    expert_norm = resolve_expert_norm_module(model)
    expert_hidden_size = expert_norm.weight.shape[0]
    print(f"  Expert hidden_size: {expert_hidden_size}")

    n_action_steps = policy.config.n_action_steps
    num_denoise_steps = policy.config.num_steps
    print(f"  n_action_steps: {n_action_steps}")
    print(f"  num_denoise_steps: {num_denoise_steps}")
    print(f"  Samples to process: {num_samples}")

    # ---- Register forward hook on expert norm ----
    hook_outputs = []

    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            hook_outputs.append(out[0].detach())
        else:
            hook_outputs.append(out.detach())

    handle = expert_norm.register_forward_hook(hook_fn)

    # ---- Detect image key from config ----
    img_keys = list(policy.config.image_features.keys())
    img_key = img_keys[0]
    print(f"  Image key from config: {img_key}")

    # ---- Config values ----
    OBS_ROBOT = "observation.state"
    state_dim = policy.config.robot_state_feature.shape[0]
    print(f"  Expected state_dim: {state_dim}")

    # ---- Resume support ----
    partial_path = os.path.join(args.output_dir, "feats_action_partial.pt")
    if args.resume_from > 0 and os.path.exists(partial_path):
        partial_tensor = torch.load(partial_path, weights_only=True)
        feats_list = list(partial_tensor[:args.resume_from])
        print(f"  Resuming from sample {args.resume_from} ({len(feats_list)} loaded)")
    else:
        feats_list = []
        args.resume_from = 0

    t0 = time.time()
    for i in range(args.resume_from, num_samples):
        img_np = np.array(Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB"))
        task = task_descriptions[i]

        image_tensor = torch.from_numpy(img_np / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(args.device)

        state_8d = state_8d_all[i]
        state_padded = np.zeros(state_dim, dtype=np.float32)
        copy_len = min(len(state_8d), state_dim)
        state_padded[:copy_len] = state_8d[:copy_len]
        state_tensor = torch.from_numpy(state_padded).unsqueeze(0).to(args.device)

        batch = {
            img_key: image_tensor,
            OBS_ROBOT: state_tensor,
            "task": [task],
        }

        # Fix seed for reproducible noise
        torch.manual_seed(args.seed + i)

        hook_outputs.clear()

        with torch.inference_mode():
            policy.select_action(batch)

        # hook_outputs has one entry per denoising step where expert norm fires
        # During prefill, only VLM norm fires (expert input is None), so no expert output
        # During denoising, expert norm fires once per step
        # Take the LAST denoising step's output
        final_expert_out = hook_outputs[-1]  # (B, suffix_len, expert_hidden_size)
        action_part = final_expert_out[:, -n_action_steps:, :]  # (B, n_action_steps, D)
        feat_action = action_part.float().mean(dim=1).squeeze(0).cpu()  # (D,)
        feats_list.append(feat_action)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            done = i + 1 - args.resume_from
            rate = done / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{num_samples}]  hook_calls={len(hook_outputs)}  "
                  f"shape={tuple(feat_action.shape)}  rate={rate:.1f}/s  ETA={eta/60:.1f}min")

        if (i + 1) % args.save_every == 0:
            torch.save(torch.stack(feats_list), partial_path)

    handle.remove()

    feats_action = torch.stack(feats_list)
    feats_action_path = os.path.join(args.output_dir, "feats_action.pt")
    torch.save(feats_action, feats_action_path)

    if os.path.exists(partial_path):
        os.remove(partial_path)

    extraction_meta = {
        "model": args.ckpt_path,
        "extraction_point": "hook on gemma_expert.model.norm at final denoising step",
        "pooling": "mean-pool across last n_action_steps tokens of expert suffix",
        "expert_hidden_size": expert_hidden_size,
        "n_action_steps": n_action_steps,
        "num_denoise_steps": num_denoise_steps,
        "feats_action_shape": list(feats_action.shape),
        "num_samples": num_samples,
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "action_repr_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== Pi0 Lerobot Phase 2b Complete ===")
    print(f"  feats_action: {feats_action_path}  shape={tuple(feats_action.shape)}")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()
