"""
Phase 2: Extract VLM features (feats_A) from StarVLA models for CKNNA.

Loads a StarVLA checkpoint, runs a VLM-only forward pass on each image
from Phase 1 data, and saves the mean-pooled last hidden state as feats_A.

Works identically for all 4 frameworks (QwenFast, QwenOFT, QwenGR00T, QwenPI)
because they all share the same qwen_vl_interface.

The extraction is purely read-only: no hooks, no model mutation, no side effects.

Usage:
  python extract_features_starvla.py \
      --ckpt_path playground/Pretrained_models/Qwen-GR00T-Bridge/checkpoints/steps_45000_pytorch_model.pt \
      --data_dir ./cknna_data \
      --output_dir ./cknna_data/Qwen-GR00T-Bridge
"""

import argparse
import json
import os
import sys
import time

import torch
from PIL import Image

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STARVLA_ROOT = os.environ.get("STARVLA_ROOT", os.path.join(_PROJECT_ROOT, "repos", "starVLA"))
if STARVLA_ROOT not in sys.path:
    sys.path.insert(0, STARVLA_ROOT)


IMAGE_TOKEN_INDEX = 151655


def masked_mean_pool(hidden_states, attention_mask):
    """Mean-pool hidden states over valid (non-padding) tokens.

    Args:
        hidden_states: (B, seq_len, D) bfloat16 or float32
        attention_mask: (B, seq_len) int or bool

    Returns:
        pooled: (B, D) float32
    """
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

    Fast path: exact BPE subsequence match with "" and " " prefixes.
    Fallback: character-level alignment via progressive prefix decoding.
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
    full_text = tokenizer.decode(ids_list, skip_special_tokens=True)
    char_pos = full_text.lower().find(task.lower().strip())
    if char_pos < 0:
        return mask
    task_end_char = char_pos + len(task.strip())
    start_tok = None
    end_tok = None
    for k in range(len(ids_list)):
        prefix_len = len(tokenizer.decode(ids_list[:k + 1], skip_special_tokens=True))
        if start_tok is None and prefix_len > char_pos:
            start_tok = k
        if prefix_len >= task_end_char:
            end_tok = k + 1
            break
    if start_tok is not None and end_tok is not None:
        mask[start_tok:end_tok] = 1
    return mask


def extract_feat_a(model, images_pil, instruction, tokenizer):
    """Run VLM prefill and extract 3 mean-pooled hidden state variants.

    Returns:
        (feat_imgtext, feat_img, feat_txt) -- each (D,) float32 on CPU.
    """
    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
        images=[images_pil],
        instructions=[instruction],
    )

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.qwen_vl_interface(
            **qwen_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    last_hidden = outputs.hidden_states[-1]
    attention_mask = qwen_inputs["attention_mask"]
    input_ids = qwen_inputs["input_ids"]

    feat_imgtext = masked_mean_pool(last_hidden, attention_mask).squeeze(0).cpu()

    image_mask = (input_ids == IMAGE_TOKEN_INDEX).to(attention_mask.dtype)
    feat_img = masked_mean_pool(last_hidden, image_mask).squeeze(0).cpu()

    task_mask = build_task_mask(input_ids[0], tokenizer, instruction).unsqueeze(0)
    feat_txt = masked_mean_pool(last_hidden, task_mask).squeeze(0).cpu()

    return feat_imgtext, feat_img, feat_txt


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Extract StarVLA features.")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Phase 1 output directory (contains images/, metadata.json)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save feats_A.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume_from", type=int, default=0,
                        help="Resume from this sample index")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    metadata_path = os.path.join(args.data_dir, "metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    print(f"Loading checkpoint: {args.ckpt_path}")
    from starVLA.model.framework.base_framework import baseframework
    model = baseframework.from_pretrained(args.ckpt_path)
    model = model.to(args.device).eval()

    framework_class = type(model).__name__
    vlm_hidden_size = model.qwen_vl_interface.model.config.hidden_size
    tokenizer = model.qwen_vl_interface.processor.tokenizer
    print(f"  Framework: {framework_class}")
    print(f"  VLM hidden size: {vlm_hidden_size}")
    print(f"  Samples to process: {num_samples}")

    feats_imgtext_list = []
    feats_img_list = []
    feats_txt_list = []

    t0 = time.time()
    for i in range(num_samples):
        img_path = os.path.join(images_dir, f"{i:06d}.png")
        img = Image.open(img_path).convert("RGB")

        instruction = task_descriptions[i]

        f_imgtext, f_img, f_txt = extract_feat_a(model, [img], instruction, tokenizer)
        feats_imgtext_list.append(f_imgtext)
        feats_img_list.append(f_img)
        feats_txt_list.append(f_txt)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{num_samples}]  shape=({vlm_hidden_size},)  "
                  f"rate={rate:.1f}/s  ETA={eta/60:.1f}min")

    for suffix, flist in [("feats_A", feats_imgtext_list),
                          ("feats_A_img", feats_img_list),
                          ("feats_A_txt", feats_txt_list)]:
        t = torch.stack(flist)
        p = os.path.join(args.output_dir, f"{suffix}.pt")
        torch.save(t, p)
        print(f"  Saved {p}  shape={tuple(t.shape)}")

    extraction_meta = {
        "checkpoint": args.ckpt_path,
        "framework": framework_class,
        "vlm_hidden_size": vlm_hidden_size,
        "num_samples": num_samples,
        "data_dir": args.data_dir,
        "outputs": ["feats_A.pt", "feats_A_img.pt", "feats_A_txt.pt"],
        "image_token_index": IMAGE_TOKEN_INDEX,
    }
    meta_path = os.path.join(args.output_dir, "extraction_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== Phase 2 Complete ===")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()
