"""
Extract CogACT action representation (feats_action) for CKNNA.

Extraction principle: feats_action = the action decoder's internal
representation at its final denoising step, right before projection
to raw action dimensions. Consistent with all other models:
  GR00T N1.5/N1.6: pre-hook on action_head.action_decoder
  Pi0:             hook on gemma_expert.model.norm
  OpenVLA:         hidden_states at generated action tokens

For CogACT, the action decoder is the DiT. The correct extraction point
is a pre-hook on DiT.final_layer (the last FinalLayer = LN+Linear that
projects from D_dit to action_dim=7). The hook fires once per denoising
step; we take the LAST step's output.

From DiT.forward():
    ...
    for block in self.blocks:
        x = block(x)                     # [B, 17, D_dit]
    ~~~ pre-hook fires HERE ~~~          # x[:, 0] = condition token
                                         # x[:, 1:] = 16 action positions
    x = self.final_layer(x)             # projects to [B, 17, 7]
    return x[:, 1:, :]                  # [B, 16, 7]

feats_action = mean(x[:, 1:, :], dim=1)  => [D_dit]
  DiT-S: D_dit=384
  DiT-B: D_dit=768
  DiT-L: D_dit=1024

Inference: DDIM with 10 steps, cfg_scale=1.5, do_sample=False.
Matches the exact SIMPLER eval forward path (sim_cogact/cogact_policy.py).
Hook fires 10 times per sample; we take the LAST (t=0) step.

Requires: cogact conda env

Usage:
    python extract_action_repr_cogact.py \
        --ckpt CogACT/CogACT-Base \
        --action_model_type DiT-B \
        --data_dir ./cknna_data \
        --output_dir ./cknna_data/cogact-base-bridge
"""

import argparse
import json
import os
import time

import sys
import numpy as np
import torch
from PIL import Image
from transformers import LlamaTokenizerFast

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_COGACT_ROOT = os.environ.get("COGACT_ROOT", os.path.join(_PROJECT_ROOT, "repos", "CogACT"))
if _COGACT_ROOT not in sys.path:
    sys.path.insert(0, _COGACT_ROOT)

from vla import load_vla


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="CogACT/CogACT-Base")
    parser.add_argument("--action_model_type", type=str, default="DiT-B",
                        help="DiT variant: DiT-S, DiT-B, DiT-L")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ddim_steps", type=int, default=10,
                        help="DDIM steps (default 10, matches SIMPLER eval)")
    parser.add_argument("--cfg_scale", type=float, default=1.5,
                        help="Classifier-free guidance scale (default 1.5, matches SIMPLER eval)")
    parser.add_argument("--unnorm_key", type=str, default=None,
                        help="Dataset key for action un-normalization. "
                             "Required when model was trained on multiple datasets "
                             "(e.g. 'bridge_orig' for OXE-pretrained CogACT).")
    parser.add_argument("--resume_from", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    print(f"Loading CogACT: {args.ckpt} (action_model_type={args.action_model_type})")
    vla = load_vla(
        args.ckpt,
        load_for_training=False,
        action_model_type=args.action_model_type,
    )
    vla.vlm = vla.vlm.to(torch.bfloat16)
    vla = vla.to(args.device).eval()

    # DiT hidden size (D_dit): 384/768/1024 for S/B/L
    dit_hidden_size = vla.action_model.net.final_layer.linear.in_features
    print(f"  DiT hidden_size (D_dit): {dit_hidden_size}")
    print(f"  DDIM steps: {args.ddim_steps}")
    print(f"  Samples to process: {num_samples}")

    # Determine unnorm_key
    unnorm_key = args.unnorm_key
    if unnorm_key is None and len(vla.norm_stats) > 1:
        # Auto-detect: prefer bridge_orig for multi-dataset OXE-pretrained models
        if "bridge_orig" in vla.norm_stats:
            unnorm_key = "bridge_orig"
            print(f"  Auto-selected unnorm_key='bridge_orig' from {list(vla.norm_stats.keys())[:3]}...")
        else:
            unnorm_key = next(iter(vla.norm_stats.keys()))
            print(f"  Auto-selected unnorm_key='{unnorm_key}' (first key)")
    elif unnorm_key is None:
        print(f"  Single dataset: unnorm_key will be auto-resolved by model")
    else:
        print(f"  Using unnorm_key='{unnorm_key}'")

    print(f"  cfg_scale: {args.cfg_scale}")

    partial_path = os.path.join(args.output_dir, "feats_action_partial.pt")
    if args.resume_from > 0 and os.path.exists(partial_path):
        partial_tensor = torch.load(partial_path, map_location="cpu", weights_only=True)
        # validate dimension matches expected D_dit
        if partial_tensor.shape[1] == dit_hidden_size:
            feats_list = list(partial_tensor[:args.resume_from])
            print(f"  Resuming from sample {args.resume_from} ({len(feats_list)} loaded)")
        else:
            print(f"  Partial has wrong dim {partial_tensor.shape[1]} vs {dit_hidden_size}, restarting")
            feats_list = []
            args.resume_from = 0
    else:
        feats_list = []
        args.resume_from = 0

    autocast_dtype = vla.vlm.llm_backbone.half_precision_dtype
    image_transform = vla.vlm.vision_backbone.image_transform
    tokenizer = vla.vlm.llm_backbone.tokenizer

    t0 = time.time()
    for i in range(args.resume_from, num_samples):
        img = Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
        task = task_descriptions[i]

        # ---- Register pre-hook on DiT.final_layer ----
        # With CFG (cfg_scale>1), forward_with_cfg calls forward() once per
        # DDIM step with a doubled batch: x shape [2*B, T+1, D_dit].
        # Index [0] = conditional (real cognition feature), [1] = unconditional.
        hook_inputs = []

        def pre_hook_fn(module, inputs):
            hook_inputs.append(inputs[0].detach().float().cpu())

        handle = vla.action_model.net.final_layer.register_forward_pre_hook(pre_hook_fn)

        # ---- Run predict_action with DDIM (matches SIMPLER eval exactly) ----
        with torch.autocast("cuda", dtype=autocast_dtype):
            vla.predict_action(
                img,
                task.lower() if task else "",
                unnorm_key=unnorm_key,
                cfg_scale=args.cfg_scale,
                use_ddim=True,
                num_ddim_steps=args.ddim_steps,
                do_sample=False,
            )

        handle.remove()

        # hook fires once per DDIM step; take the LAST step (t=0, cleanest)
        # With CFG: hook_inputs[-1] shape = [2, 17, D_dit]
        #   [0] = conditional pass, [1] = unconditional pass
        # Without CFG: shape = [1, 17, D_dit]
        # Position 0 = condition token, 1..16 = 16 action steps
        final_x = hook_inputs[-1]           # [2*B, 17, D_dit] or [B, 17, D_dit]
        action_x = final_x[0, 1:, :]        # [16, D_dit] (conditional sample)
        feat_action = action_x.mean(dim=0)   # [D_dit]
        feats_list.append(feat_action)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            done = i + 1 - args.resume_from
            rate = done / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{num_samples}]  "
                  f"shape=({dit_hidden_size},)  hook_calls={len(hook_inputs)}  "
                  f"rate={rate:.1f}/s  ETA={eta/60:.1f}min")

        hook_inputs.clear()

        if (i + 1) % args.save_every == 0:
            torch.save(torch.stack(feats_list), partial_path)

    feats_action = torch.stack(feats_list)
    feats_action_path = os.path.join(args.output_dir, "feats_action.pt")
    torch.save(feats_action, feats_action_path)

    if os.path.exists(partial_path):
        os.remove(partial_path)

    extraction_meta = {
        "model": args.ckpt,
        "action_model_type": args.action_model_type,
        "extraction_point": (
            f"pre-hook on DiT.final_layer at final DDIM step ({args.ddim_steps} steps); "
            f"mean-pool over 16 action positions; shape [D_dit={dit_hidden_size}]"
        ),
        "dit_hidden_size": dit_hidden_size,
        "ddim_steps": args.ddim_steps,
        "cfg_scale": args.cfg_scale,
        "feats_action_shape": list(feats_action.shape),
        "num_samples": num_samples,
    }
    with open(os.path.join(args.output_dir, "action_repr_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== CogACT Action Repr Extraction Complete ===")
    print(f"  feats_action: {feats_action_path}  shape={tuple(feats_action.shape)}")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()
