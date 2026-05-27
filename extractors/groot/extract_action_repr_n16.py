"""
Extract action representations (feats_action) from GR00T N1.6 for CKNNA.

Runs the full model (backbone + action head denoising loop) and hooks on
action_head.action_decoder (pre-hook) to capture the DiT output at every
denoising step. Takes the final step's representation and mean-pools across
the action horizon.

Architecture:
    Gr00tN1d6ActionHead:
        backbone_features -> vlln + AlternateVLDiT (32 layers, 4 denoising steps)
        -> action_decoder (CategorySpecificMLP, hidden_size=1024 -> action_dim)

    The pre-hook captures the DiT output (input to action_decoder),
    shape (B, state_horizon + action_horizon, 1024). We slice the last
    action_horizon tokens and mean-pool -> (1024,).

Requires:
    PYTHONPATH must point to gr00t_1p6/Isaac-GR00T

Usage:
    PYTHONPATH=$WORK/gr00t_1p6/Isaac-GR00T python extract_action_repr_groot_n16.py \
        --ckpt $WORK/GR00T-N1.6-bridge \
        --data_dir ./cknna_data \
        --output_dir ./cknna_data/groot-n16-bridge
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_GROOT_N16 = os.environ.get("ISAAC_GROOT_N16_DIR", os.path.join(_PROJECT_ROOT, "repos", "Isaac-GR00T-N16"))
if _GROOT_N16 not in sys.path:
    sys.path.insert(0, _GROOT_N16)

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype


def build_bridge_obs(image_np, task_description, state_8d):
    """Build N1.6-format observation dict using real state from feats_B."""
    return {
        "video": {
            "image_0": image_np[np.newaxis, np.newaxis],
        },
        "state": {
            "x": state_8d[0:1].reshape(1, 1, 1).astype(np.float32),
            "y": state_8d[1:2].reshape(1, 1, 1).astype(np.float32),
            "z": state_8d[2:3].reshape(1, 1, 1).astype(np.float32),
            "roll": state_8d[3:4].reshape(1, 1, 1).astype(np.float32),
            "pitch": state_8d[4:5].reshape(1, 1, 1).astype(np.float32),
            "yaw": state_8d[5:6].reshape(1, 1, 1).astype(np.float32),
            "pad": state_8d[6:7].reshape(1, 1, 1).astype(np.float32),
            "gripper": state_8d[7:8].reshape(1, 1, 1).astype(np.float32),
        },
        "language": {
            "annotation.human.action.task_description": [[task_description]],
        },
    }


def feats_b_7d_to_state_8d(feats_B):
    """Convert 7D feats_B (pad removed) back to 8D state (pad=0 at index 6)."""
    n = feats_B.shape[0]
    state_8d = np.zeros((n, 8), dtype=np.float32)
    state_8d[:, :6] = feats_B[:, :6]
    state_8d[:, 6] = 0.0
    state_8d[:, 7] = feats_B[:, 6]
    return state_8d


def process_single_obs(policy, obs):
    """Run one observation through the N1.6 processor pipeline."""
    unbatched = policy._unbatch_observation(obs)
    vla_step = policy._to_vla_step_data(unbatched[0])
    messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step}]
    processed = policy.processor(messages)
    collated = policy.collate_fn([processed])
    collated = _rec_to_dtype(collated, dtype=torch.bfloat16)
    return collated["inputs"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to GR00T N1.6 checkpoint directory (e.g. nvidia/GR00T-N1.6-bridge)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
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

    feats_B = torch.load(
        os.path.join(args.data_dir, "feats_B.pt"), weights_only=True
    ).numpy()
    state_all_8d = feats_b_7d_to_state_8d(feats_B)

    print(f"Loading GR00T N1.6: {args.ckpt}")
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.OXE_WIDOWX,
        model_path=args.ckpt,
        device=args.device,
    )
    model = policy.model

    action_hidden_size = model.action_head.hidden_size
    action_horizon = model.action_head.action_horizon
    num_denoising_steps = model.action_head.num_inference_timesteps
    print(f"  action_hidden_size: {action_hidden_size}")
    print(f"  action_horizon: {action_horizon}")
    print(f"  num_denoising_steps: {num_denoising_steps}")
    print(f"  Samples to process: {num_samples}")

    partial_path = os.path.join(args.output_dir, "feats_action_partial.pt")
    if args.resume_from > 0 and os.path.exists(partial_path):
        partial_tensor = torch.load(partial_path, weights_only=True)
        feats_list = list(partial_tensor[:args.resume_from])
        print(f"  Resuming from {args.resume_from} ({len(feats_list)} loaded)")
    else:
        feats_list = []
        args.resume_from = 0

    t0 = time.time()
    for i in range(args.resume_from, num_samples):
        img = np.array(
            Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
        )
        task = task_descriptions[i]
        state_8d = state_all_8d[i]

        obs = build_bridge_obs(img, task, state_8d)
        inputs = process_single_obs(policy, obs)

        torch.manual_seed(args.seed + i)

        action_reprs = []

        def pre_hook_fn(module, hook_inputs):
            action_reprs.append(hook_inputs[0].detach())

        handle = model.action_head.action_decoder.register_forward_pre_hook(
            pre_hook_fn
        )

        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            model.get_action(inputs)

        handle.remove()

        final_repr = action_reprs[-1]
        action_part = final_repr[:, -action_horizon:, :]
        feat = action_part.float().mean(dim=1).squeeze(0).cpu()
        feats_list.append(feat)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            done = i + 1 - args.resume_from
            rate = done / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1}/{num_samples}]  shape=({action_hidden_size},)  "
                f"rate={rate:.1f}/s  ETA={eta/60:.1f}min"
            )

        if (i + 1) % args.save_every == 0:
            torch.save(torch.stack(feats_list), partial_path)

    feats_action = torch.stack(feats_list)
    feats_action_path = os.path.join(args.output_dir, "feats_action.pt")
    torch.save(feats_action, feats_action_path)

    if os.path.exists(partial_path):
        os.remove(partial_path)

    extraction_meta = {
        "model": "GR00T-N1.6-bridge",
        "checkpoint_path": args.ckpt,
        "extraction_point": (
            "action_head.action_decoder pre-hook (final denoising step): "
            f"AlternateVLDiT output, shape (B, {action_horizon}, {action_hidden_size})"
        ),
        "pooling": "mean-pool across action horizon",
        "feats_action_shape": list(feats_action.shape),
        "num_samples": num_samples,
        "action_hidden_size": action_hidden_size,
        "action_horizon": action_horizon,
        "num_denoising_steps": num_denoising_steps,
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "action_repr_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== GR00T N1.6 Action Extraction Complete ===")
    print(f"  feats_action: {feats_action_path}  shape={tuple(feats_action.shape)}")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()
