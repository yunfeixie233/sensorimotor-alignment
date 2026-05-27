"""
Phase 2: Extract features from Qwen2-VL-2B-Instruct for CKNNA.

This is the base VLM backbone used by DexVLA. DexVLA builds on Qwen2-VL
with a plug-in diffusion expert for action generation, but the VLM backbone
is Qwen2-VL.

Architecture:
  Image → Qwen2 Vision Transformer → visual tokens
  Text  → Qwen2 LM tokenizer → text embeddings
  Both  → Qwen2 decoder (joint attention) → hidden_states

Extraction point:
  Last hidden state from the full Qwen2-VL forward pass (output_hidden_states=True).
  Mean-pool over all valid tokens (image + text).
  Also saves img-only and txt-only pooled variants.

DexVLA does NOT have officially reported RoboTwin numbers in its paper.
It is pending on the RoboTwin leaderboard. This extraction uses the
base Qwen2-VL-2B-Instruct model.

Usage:
    python extract_features.py \
        --data_dir /path/to/cknna_data_exp1v2_full_L \
        --output_dir /path/to/cknna_data_exp1v2_full_L/qwen2vl-2b-instruct
"""

import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Extract Qwen2-VL features for CKNNA.")
    parser.add_argument("--ckpt", type=str,
                        default="Qwen/Qwen2-VL-2B-Instruct",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume_from", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    print(f"Loading Qwen2-VL: {args.ckpt}")
    processor = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.ckpt,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval().to(args.device)

    hidden_size = model.config.text_config.hidden_size
    print(f"  hidden_size: {hidden_size}")
    print(f"  Samples to process: {num_samples}")

    # Prepare storage
    feats_path = os.path.join(args.output_dir, "feats_A.pt")
    feats_img_path = os.path.join(args.output_dir, "feats_A_img.pt")
    feats_txt_path = os.path.join(args.output_dir, "feats_A_txt.pt")

    if args.resume_from > 0 and os.path.exists(feats_path):
        all_feats = torch.load(feats_path, weights_only=True)
        all_feats_img = torch.load(feats_img_path, weights_only=True)
        all_feats_txt = torch.load(feats_txt_path, weights_only=True)
        print(f"  Resuming from sample {args.resume_from}")
    else:
        all_feats = torch.zeros(num_samples, hidden_size, dtype=torch.float32)
        all_feats_img = torch.zeros(num_samples, hidden_size, dtype=torch.float32)
        all_feats_txt = torch.zeros(num_samples, hidden_size, dtype=torch.float32)
        args.resume_from = 0

    t0 = time.time()

    with torch.no_grad():
        for i in range(args.resume_from, num_samples):
            img_path = os.path.join(images_dir, f"{i:06d}.png")
            img = Image.open(img_path).convert("RGB")
            task = task_descriptions[i] if i < len(task_descriptions) else ""

            # Build Qwen2-VL chat input
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": task if task else "Describe this image."},
                    ],
                }
            ]
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text_input], images=[img],
                return_tensors="pt", padding=True
            ).to(args.device)

            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

            last_hs = outputs.hidden_states[-1][0].float()  # (seq_len, D)
            attn_mask = inputs["attention_mask"][0].float()  # (seq_len,)

            # Identify image vs text tokens via input_ids
            input_ids = inputs["input_ids"][0]
            # Qwen2-VL image tokens have special IDs
            # Vision start: 151652, Vision end: 151653, Image pad: 151655
            img_token_ids = {151652, 151653, 151655}
            is_img = torch.tensor(
                [int(tid.item() in img_token_ids) for tid in input_ids],
                device=last_hs.device, dtype=torch.float32
            )
            is_txt = (attn_mask - is_img).clamp(min=0)

            # imgtext: all valid tokens
            feat_imgtext = (last_hs * attn_mask.unsqueeze(-1)).sum(0) / attn_mask.sum().clamp(min=1)
            # img only
            feat_img = (last_hs * is_img.unsqueeze(-1)).sum(0) / is_img.sum().clamp(min=1)
            # txt only
            feat_txt = (last_hs * is_txt.unsqueeze(-1)).sum(0) / is_txt.sum().clamp(min=1)

            all_feats[i] = feat_imgtext.cpu()
            all_feats_img[i] = feat_img.cpu()
            all_feats_txt[i] = feat_txt.cpu()

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1 - args.resume_from) / elapsed
                print(f"  [{i+1}/{num_samples}]  {rate:.1f} samples/s  "
                      f"elapsed: {elapsed:.0f}s")

            if (i + 1) % args.save_every == 0:
                torch.save(all_feats, feats_path)
                torch.save(all_feats_img, feats_img_path)
                torch.save(all_feats_txt, feats_txt_path)

    torch.save(all_feats, feats_path)
    torch.save(all_feats_img, feats_img_path)
    torch.save(all_feats_txt, feats_txt_path)

    elapsed = time.time() - t0
    print(f"\nDone. {num_samples} samples in {elapsed:.1f}s")
    print(f"  feats_A shape: {all_feats.shape}")
    print(f"  feats_A stats: mean={all_feats.mean():.4f}, std={all_feats.std():.4f}")

    meta = {
        "model": "qwen2vl-2b-instruct",
        "base_for": "DexVLA",
        "extraction_point": "last_hidden_state mean-pooled (imgtext/img/txt)",
        "hidden_size": hidden_size,
        "checkpoint": args.ckpt,
        "note": "Base VLM for DexVLA. DexVLA is pending on RoboTwin leaderboard (no official numbers).",
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
