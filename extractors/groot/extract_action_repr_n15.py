"""
Phase 2b: Extract action representations (feats_action) from GR00T N1.5 for CKNNA.

Runs the full model (backbone + action head denoising loop) and hooks on
action_head.action_decoder (pre-hook) to capture the DiT output at every
denoising step. Takes the final step's representation and mean-pools across
the action horizon.

Architecture:
    FlowmatchingActionHead:
        backbone_features -> vlln + vl_self_attention -> DiT (4 denoising steps)
        -> action_decoder (CategorySpecificMLP, hidden_size=1024 -> action_dim=32)

    The pre-hook captures the DiT output (input to action_decoder),
    shape (B, state_horizon + action_horizon, 1024). We slice the last
    action_horizon tokens and mean-pool -> (1024,).

    This extraction is aligned with StarVLA's extract_action_repr_groot(),
    which uses the same hook strategy on the analogous action_decoder.

Usage:
    PYTHONPATH=$WORK/Isaac-GR00T python extract_action_repr_groot_n15.py \
        --ckpt $WORK/GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2 \
        --data_dir ./cknna_data \
        --output_dir ./cknna_data/groot-n15-bridge
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
_GROOT_N15 = os.environ.get("ISAAC_GROOT_DIR", os.path.join(_PROJECT_ROOT, "repos", "Isaac-GR00T-N15"))
if _GROOT_N15 not in sys.path:
    sys.path.insert(0, _GROOT_N15)

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy, unsqueeze_dict_values


def build_bridge_batch(image_np, task_description, state_8d):
    """Construct an observation dict matching GR00T's Bridge format.

    Args:
        image_np: (H, W, 3) uint8 numpy array
        task_description: str
        state_8d: (8,) float32 array [x,y,z,roll,pitch,yaw,pad,gripper]

    Returns:
        batch dict ready for unsqueezing + transforms
    """
    img = image_np[None]  # (1, H, W, C) -- T=1
    return {
        "video.image_0": img,
        "state.x": state_8d[0:1][None],
        "state.y": state_8d[1:2][None],
        "state.z": state_8d[2:3][None],
        "state.roll": state_8d[3:4][None],
        "state.pitch": state_8d[4:5][None],
        "state.yaw": state_8d[5:6][None],
        "state.pad": state_8d[6:7][None],
        "state.gripper": state_8d[7:8][None],
        "annotation.human.action.task_description": [task_description],
    }


def feats_b_7d_to_state_8d(feats_B):
    """Convert 7D feats_B (pad removed) back to 8D state (pad=0 at index 6).

    feats_B columns: [x, y, z, roll, pitch, yaw, gripper]
    state_8d columns: [x, y, z, roll, pitch, yaw, pad, gripper]
    """
    n = feats_B.shape[0]
    state_8d = np.zeros((n, 8), dtype=np.float32)
    state_8d[:, :6] = feats_B[:, :6]   # x,y,z,roll,pitch,yaw
    state_8d[:, 6] = 0.0               # pad
    state_8d[:, 7] = feats_B[:, 6]     # gripper
    return state_8d


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2b: Extract GR00T N1.5 action representations for CKNNA."
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to GR00T N1.5 checkpoint directory (e.g. ShuaiYang03/GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2)",
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Phase 1 data dir (images/, metadata.json, feats_B.pt)")
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

    feats_B = torch.load(
        os.path.join(args.data_dir, "feats_B.pt"), weights_only=True
    ).numpy()
    state_all_8d = feats_b_7d_to_state_8d(feats_B)

    print(f"Loading GR00T N1.5: {args.ckpt}")

    data_config = DATA_CONFIG_MAP["bridge"]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()

    policy = Gr00tPolicy(
        model_path=args.ckpt,
        modality_config=modality_config,
        modality_transform=transforms,
        embodiment_tag="oxe",
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

        batch = build_bridge_batch(img, task, state_8d)
        batch_unsqueezed = unsqueeze_dict_values(batch)
        normalized_input = policy.apply_transforms(batch_unsqueezed)

        torch.manual_seed(args.seed + i)

        action_reprs = []

        def pre_hook_fn(module, inputs):
            action_reprs.append(inputs[0].detach())

        handle = model.action_head.action_decoder.register_forward_pre_hook(
            pre_hook_fn
        )

        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            model.get_action(normalized_input)

        handle.remove()

        # action_reprs has one entry per denoising step; take the last
        final_repr = action_reprs[-1]  # (B, state_horizon + action_horizon, D)
        action_part = final_repr[:, -action_horizon:, :]  # (B, action_horizon, D)
        feat = action_part.float().mean(dim=1).squeeze(0).cpu()  # (D,)
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
        "model": "GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2",
        "checkpoint_path": args.ckpt,
        "extraction_point": (
            "action_head.action_decoder pre-hook (final denoising step): "
            f"DiT output, shape (B, {action_horizon}, {action_hidden_size})"
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
    print(f"\n=== GR00T N1.5 Phase 2b Complete ===")
    print(f"  feats_action: {feats_action_path}  shape={tuple(feats_action.shape)}")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()
