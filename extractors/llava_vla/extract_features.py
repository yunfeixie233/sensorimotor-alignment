"""
Phase 2: Extract features from LLaVA-OneVision-0.5B for CKNNA.

This is the base VLM backbone used by LLaVA-VLA. Architecture:
  SigLIP vision encoder + Qwen2-0.5B language model.

Extraction point:
  Last hidden state from the full forward pass, mean-pooled over
  valid tokens (image + text). Also saves img-only and txt-only variants.

LLaVA-VLA self-reports RoboTwin numbers: Seen 40.3%, DR 28.6% (8 tasks only).

Usage:
    python extract_features.py \
        --data_dir /path/to/cknna_data_exp1v2_full_L \
        --output_dir /path/to/cknna_data_exp1v2_full_L/llava-onevision-0.5b
"""

import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Extract LLaVA-OneVision features for CKNNA.")
    parser.add_argument("--ckpt", type=str,
                        default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
                        help="HuggingFace model ID")
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

    print(f"Loading LLaVA-OneVision: {args.ckpt}")
    processor = AutoProcessor.from_pretrained(args.ckpt)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.ckpt,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval().to(args.device)

    hidden_size = model.config.text_config.hidden_size
    print(f"  hidden_size: {hidden_size}")
    print(f"  Samples to process: {num_samples}")

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

    # Get image token ID for separating img vs txt
    image_token_id = model.config.image_token_index  # typically 151646

    t0 = time.time()

    with torch.no_grad():
        for i in range(args.resume_from, num_samples):
            img_path = os.path.join(images_dir, f"{i:06d}.png")
            img = Image.open(img_path).convert("RGB")
            task = task_descriptions[i] if i < len(task_descriptions) else ""

            # Build conversation input
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": task if task else "Describe this image."},
                    ],
                },
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=img, text=prompt, return_tensors="pt").to(args.device)

            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

            last_hs = outputs.hidden_states[-1][0].float()  # (seq_len, D)
            input_ids = inputs["input_ids"][0]

            # Separate image vs text tokens
            is_img = (input_ids == image_token_id).float()
            attn_mask_raw = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))
            # Remove batch dim if present
            attn_mask = attn_mask_raw[0].float() if attn_mask_raw.ndim > 1 else attn_mask_raw.float()
            is_txt = (attn_mask - is_img).clamp(min=0)

            feat_imgtext = (last_hs * attn_mask.unsqueeze(-1)).sum(0) / attn_mask.sum().clamp(min=1)
            feat_img = (last_hs * is_img.unsqueeze(-1)).sum(0) / is_img.sum().clamp(min=1)
            feat_txt = (last_hs * is_txt.unsqueeze(-1)).sum(0) / is_txt.sum().clamp(min=1)

            all_feats[i] = feat_imgtext.cpu()
            all_feats_img[i] = feat_img.cpu()
            all_feats_txt[i] = feat_txt.cpu()

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1 - args.resume_from) / elapsed
                print(f"  [{i+1}/{num_samples}]  {rate:.1f} samples/s")

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

    meta = {
        "model": "llava-onevision-qwen2-0.5b",
        "base_for": "LLaVA-VLA",
        "extraction_point": "last_hidden_state mean-pooled (imgtext/img/txt)",
        "hidden_size": hidden_size,
        "checkpoint": args.ckpt,
        "note": "Base VLM for LLaVA-VLA. Self-reported RoboTwin: Seen 40.3%, DR 28.6% (8 tasks).",
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
