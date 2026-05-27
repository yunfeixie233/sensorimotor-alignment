"""
Phase 2b: Extract action representations (feats_action) from SpatialVLA for CKNNA.

Runs generate() with output_hidden_states=True, return_dict_in_generate=True.
Extracts last-layer hidden states at each generated action token position,
then mean-pools across all generated tokens.

Architecture: PaLiGemma2 (Gemma2 LM, hidden_size=2304)
Action tokens: 7 tokens (one per action dimension)

Usage:
    python extract_action_repr_spatialvla.py \
        --ckpt IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge \
        --data_dir ./cknna_data \
        --output_dir ./cknna_data/spatialvla-sft-bridge \
        --seed 42
"""

import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


def main():
    parser = argparse.ArgumentParser(description="Phase 2b: Extract SpatialVLA action representations.")
    parser.add_argument("--ckpt", type=str, default="IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Phase 1 data directory (images/, metadata.json)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save feats_action.pt")
    parser.add_argument("--unnorm_key", type=str, default="bridge_orig/1.0.0")
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

    print(f"Loading SpatialVLA: {args.ckpt}")
    processor = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.ckpt,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval().to(args.device)

    hidden_size = model.config.text_config.hidden_size
    print(f"  hidden_size: {hidden_size}")
    print(f"  Samples to process: {num_samples}")

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
        img = Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
        task = task_descriptions[i]

        inputs = processor(
            images=[img],
            text=task,
            unnorm_key=args.unnorm_key,
            return_tensors="pt",
            do_normalize=False,
        )
        inputs = inputs.to(dtype=torch.bfloat16, device=args.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        # outputs.hidden_states:
        #   [0] = prefill step: tuple of (num_layers+1) tensors, each (B, prefill_len, D)
        #   [1:] = each generated token: tuple of (num_layers+1) tensors, each (B, 1, D)
        num_gen_steps = len(outputs.hidden_states) - 1
        if num_gen_steps == 0:
            feats_list.append(torch.zeros(hidden_size, dtype=torch.float32))
            continue

        action_hidden = []
        for t in range(1, len(outputs.hidden_states)):
            last_layer = outputs.hidden_states[t][-1]  # (B, 1, D)
            action_hidden.append(last_layer)

        action_repr = torch.cat(action_hidden, dim=1)  # (B, num_gen, D)
        feat_action = action_repr.float().mean(dim=1).squeeze(0).cpu()  # (D,)
        feats_list.append(feat_action)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            done = i + 1 - args.resume_from
            rate = done / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{num_samples}]  gen_tokens={num_gen_steps}  "
                  f"shape={tuple(feat_action.shape)}  rate={rate:.1f}/s  ETA={eta/60:.1f}min")

        if (i + 1) % args.save_every == 0:
            torch.save(torch.stack(feats_list), partial_path)

    feats_action = torch.stack(feats_list)
    feats_action_path = os.path.join(args.output_dir, "feats_action.pt")
    torch.save(feats_action, feats_action_path)

    if os.path.exists(partial_path):
        os.remove(partial_path)

    extraction_meta = {
        "model": args.ckpt,
        "extraction_point": "generate() hidden_states[-1] at each gen step",
        "pooling": "mean-pool across generated action tokens",
        "hidden_size": hidden_size,
        "feats_action_shape": list(feats_action.shape),
        "num_samples": num_samples,
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "action_repr_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== SpatialVLA Phase 2b Complete ===")
    print(f"  feats_action: {feats_action_path}  shape={tuple(feats_action.shape)}")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()
