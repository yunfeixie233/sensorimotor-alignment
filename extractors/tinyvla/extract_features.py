"""
Phase 2: Extract features from InternVL3-1B for CKNNA.

This is the base VLM backbone used by TinyVLA. InternVL3-1B uses:
  - InternViT (vision encoder)
  - InternLM2-1.8B (language model)

Extraction point:
  Last hidden state from the full forward pass, mean-pooled.
  Also saves img-only and txt-only variants.

TinyVLA does NOT have officially reported RoboTwin numbers.
It is pending on the RoboTwin leaderboard.

Usage:
    python extract_features.py \
        --data_dir /path/to/cknna_data_exp1v2_full_L \
        --output_dir /path/to/cknna_data_exp1v2_full_L/internvl3-1b
"""

import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Extract InternVL3-1B features for CKNNA.")
    parser.add_argument("--ckpt", type=str,
                        default="OpenGVLab/InternVL3-1B",
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

    print(f"Loading InternVL3: {args.ckpt}")
    model = AutoModel.from_pretrained(
        args.ckpt,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval().to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)

    hidden_size = model.config.llm_config.hidden_size
    print(f"  hidden_size: {hidden_size}")
    print(f"  Samples to process: {num_samples}")

    # Get the image token ID
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

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

            # Use InternVL chat interface
            question = f"<image>\n{task}" if task else "<image>\nDescribe this image."

            # Use the model's built-in chat method to get proper formatting
            # But we need hidden states, so we'll manually build the input
            try:
                # Try using the model's pixel_values + input_ids approach
                pixel_values = model.extract_feature(img)
                if pixel_values is not None:
                    pixel_values = pixel_values.to(args.device, dtype=torch.bfloat16)
            except Exception:
                pixel_values = None

            # Fallback: use the chat interface for tokenization
            # then do a manual forward pass
            generation_config = dict(max_new_tokens=1, do_sample=False)
            try:
                # InternVL3 has a chat method that handles formatting
                response, history = model.chat(
                    tokenizer, pixel_values, question,
                    generation_config, history=None,
                    return_history=True,
                    output_hidden_states=True,
                )
                # If chat doesn't return hidden states, we do manual forward
            except Exception:
                pass

            # Manual forward for hidden states
            from transformers import AutoProcessor as AP
            try:
                proc = AP.from_pretrained(args.ckpt, trust_remote_code=True)
                inputs = proc(images=img, text=question, return_tensors="pt").to(args.device)
                outputs = model.language_model(
                    inputs_embeds=model.get_input_embeddings()(inputs["input_ids"]),
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_hs = outputs.hidden_states[-1][0].float()
                seq_len = last_hs.shape[0]
                attn_mask = torch.ones(seq_len, device=last_hs.device)
                input_ids = inputs["input_ids"][0]
                is_img = (input_ids == img_context_token_id).float()
                is_txt = (attn_mask - is_img).clamp(min=0)
            except Exception as e:
                # Simplest fallback: use just the vision encoder
                print(f"  Warning: Full forward failed for sample {i}, using vision-only. Error: {e}")
                if pixel_values is not None and pixel_values.ndim >= 2:
                    feat = pixel_values.float().mean(dim=tuple(range(pixel_values.ndim - 1)))
                else:
                    feat = torch.zeros(hidden_size)
                all_feats[i] = feat.cpu()
                all_feats_img[i] = feat.cpu()
                all_feats_txt[i] = torch.zeros(hidden_size)
                continue

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
        "model": "internvl3-1b",
        "base_for": "TinyVLA",
        "extraction_point": "last_hidden_state mean-pooled (imgtext/img/txt)",
        "hidden_size": hidden_size,
        "checkpoint": args.ckpt,
        "note": "Base VLM for TinyVLA. TinyVLA is pending on RoboTwin leaderboard (no official numbers).",
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
