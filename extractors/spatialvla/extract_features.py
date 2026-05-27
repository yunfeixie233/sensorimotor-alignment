"""
Phase 2: Extract VLM features (feats_A) from SpatialVLA for CKNNA.

Loads the SpatialVLA checkpoint, runs a forward pass with
output_hidden_states=True on each (image, task) pair from Phase 1 data,
and saves the masked-mean-pooled last hidden state as feats_A.

Architecture: PaLiGemma2 (SigLIP vision + ZoeDepth + Gemma2 LM, hidden_size=2304)
Extraction point: hidden_states[-1] (POST-norm, after final Gemma2 RMSNorm)
Pooling: masked mean-pool over non-padding tokens (matches StarVLA pipeline)

The extraction is purely read-only: no hooks, no model mutation, no side effects.
output_hidden_states=True is a standard HuggingFace parameter that includes
intermediate hidden states in the return value without affecting the computation.

Requires: conda env with transformers>=4.47.0, torch>=2.1

Usage:
    python extract_features_spatialvla.py \
        --ckpt IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge \
        --data_dir ./cknna_data \
        --output_dir ./cknna_data/spatialvla-sft-bridge
"""

import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


IMAGE_TOKEN_INDEX = 257152


def masked_mean_pool(hidden_states, attention_mask):
    """Mean-pool hidden states over valid (non-padding) tokens."""
    h = hidden_states.float()
    m = attention_mask.unsqueeze(-1).float()
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def find_subsequence(seq, subseq):
    n, m = len(seq), len(subseq)
    if m == 0:
        return -1
    for i in range(n - m + 1):
        if seq[i:i + m] == subseq:
            return i
    return -1


def build_task_mask(input_ids_1d, tokenizer, task):
    """Build mask for task-instruction tokens in the input sequence.

    Tries bare encoding first (handles SentencePiece/BPE after non-space chars
    like <bos> or newline), then falls back to space-prefixed encoding (handles
    cases where the task follows a space in the prompt).
    """
    mask = torch.zeros(len(input_ids_1d), dtype=torch.long, device=input_ids_1d.device)
    if not task:
        return mask
    ids_list = input_ids_1d.tolist()
    for prefix in ["", " "]:
        task_ids = tokenizer.encode(prefix + task, add_special_tokens=False)
        start = find_subsequence(ids_list, task_ids)
        if start >= 0:
            mask[start:start + len(task_ids)] = 1
            return mask
    return mask


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Extract SpatialVLA features for CKNNA.")
    parser.add_argument("--ckpt", type=str, default="IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Phase 1 data directory (images/, metadata.json)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save feats_A.pt")
    parser.add_argument("--unnorm_key", type=str, default="bridge_orig/1.0.0",
                        help="Dataset key for intrinsic camera params (ZoeDepth)")
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

    print(f"Loading SpatialVLA: {args.ckpt}")
    processor = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.ckpt,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval().to(args.device)

    hidden_size = model.config.text_config.hidden_size
    print(f"  hidden_size: {hidden_size}")
    print(f"  use_vision_zoe: {getattr(model.config, 'use_vision_zoe', 'N/A')}")
    print(f"  Samples to process: {num_samples}")

    tokenizer = processor.tokenizer

    feats_imgtext_list = []
    feats_img_list = []
    feats_txt_list = []

    t0 = time.time()
    for i in range(num_samples):
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
            outputs = model(**inputs, output_hidden_states=True)

        last_hidden = outputs.hidden_states[-1]
        mask = inputs["attention_mask"]
        input_ids = inputs["input_ids"]

        feat_imgtext = masked_mean_pool(last_hidden, mask).squeeze(0).cpu()

        image_mask = (input_ids == IMAGE_TOKEN_INDEX).to(mask.dtype)
        feat_img = masked_mean_pool(last_hidden, image_mask).squeeze(0).cpu()

        task_mask = build_task_mask(input_ids[0], tokenizer, task).unsqueeze(0)
        feat_txt = masked_mean_pool(last_hidden, task_mask).squeeze(0).cpu()

        feats_imgtext_list.append(feat_imgtext)
        feats_img_list.append(feat_img)
        feats_txt_list.append(feat_txt)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{num_samples}]  shape=({hidden_size},)  "
                  f"rate={rate:.1f}/s  ETA={eta/60:.1f}min")

    for suffix, flist in [("feats_A", feats_imgtext_list),
                          ("feats_A_img", feats_img_list),
                          ("feats_A_txt", feats_txt_list)]:
        t = torch.stack(flist)
        p = os.path.join(args.output_dir, f"{suffix}.pt")
        torch.save(t, p)
        print(f"  Saved {p}  shape={tuple(t.shape)}")

    extraction_meta = {
        "model": args.ckpt,
        "hidden_size": hidden_size,
        "num_samples": num_samples,
        "outputs": ["feats_A.pt", "feats_A_img.pt", "feats_A_txt.pt"],
        "image_token_index": IMAGE_TOKEN_INDEX,
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== SpatialVLA Phase 2 Complete ===")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()
