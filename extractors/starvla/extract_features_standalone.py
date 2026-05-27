"""
Phase 2: Extract VLM features from StarVLA models for CKNNA (standalone version).

This version bypasses the StarVLA framework's build_qwenvl_inputs() method
and directly accesses the Qwen3-VL backbone for feature extraction. This avoids
compatibility issues between StarVLA and newer transformers versions.

Architecture: StarVLA OFT = Qwen3-VL backbone + action head (DiT-B).
  Only the Qwen3-VL backbone is used for CKNNA feature extraction.

Extraction point: Last hidden state (post-norm) of the Qwen3-VL backbone.
  The VLM backbone is accessed via model.qwen_vl_interface.model

Pooling:
  feats_A:     mean-pool over all valid (non-padding) tokens
  feats_A_img: mean-pool over image tokens only (token_id == 151655)
  feats_A_txt: mean-pool over non-padding text tokens only

Usage:
    cd /home/ubuntu/verl/starVLA && \\
    python extract_features_standalone.py \\
        --ckpt_path /path/to/steps_XXXXX_pytorch_model.pt \\
        --data_dir /path/to/cknna_data \\
        --output_dir /path/to/cknna_data/starvla-qwen3-oft-rt2
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
    """Mean-pool hidden states over valid (non-padding) tokens."""
    h = hidden_states.float()
    m = attention_mask.unsqueeze(-1).float()
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def find_subsequence(seq, subseq):
    """Find the starting position of a subsequence in a sequence."""
    n, m = len(seq), len(subseq)
    if m == 0:
        return -1
    for i in range(n - m + 1):
        if seq[i:i + m] == subseq:
            return i
    return -1


def build_task_mask(input_ids_1d, tokenizer, task):
    """Build mask for task-instruction tokens in the input sequence."""
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
    # Fallback: character-level alignment
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


def build_simple_inputs(image_pil, instruction, processor, device):
    """Build Qwen3-VL inputs directly using the processor's image_processor + tokenizer.

    Avoids apply_chat_template() for compatibility across transformers versions.
    Matches the token layout used during StarVLA training.
    """
    # Process image
    img_inputs = processor.image_processor(
        image_pil,
        return_tensors="pt",
    )
    pixel_values = img_inputs["pixel_values"].to(device)
    image_grid_thw = img_inputs["image_grid_thw"].to(device)

    # Compute number of image tokens
    spatial_merge_size = getattr(processor.image_processor, 'merge_size', 2)
    num_img_tokens = int(torch.prod(image_grid_thw).item()) // (spatial_merge_size ** 2)

    # Build input_ids matching Qwen3-VL chat format
    # <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|>...<|vision_end|>instruction<|im_end|>\n<|im_start|>assistant\n
    tokenizer = processor.tokenizer

    # Get special token IDs
    vision_start_id = getattr(processor, 'vision_start_token_id', 151652)
    vision_end_id = getattr(processor, 'vision_end_token_id', 151653)
    image_token_id = getattr(processor, 'image_token_id', 151655)

    # Simple format: just vision tokens + instruction (no chat template overhead)
    input_ids = (
        [vision_start_id]
        + [image_token_id] * num_img_tokens
        + [vision_end_id]
    )

    # Tokenize instruction
    text_ids = tokenizer.encode(instruction, add_special_tokens=False)
    input_ids += text_ids

    attention_mask = [1] * len(input_ids)

    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Extract StarVLA features (standalone).")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Phase 1 output directory (contains images/, metadata.json)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save feats_A.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples to process (0 = all)")
    parser.add_argument("--save_every", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    metadata_path = os.path.join(args.data_dir, "metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    if args.max_samples > 0:
        num_samples = min(num_samples, args.max_samples)

    # Load the full StarVLA model using the framework loader
    print(f"Loading checkpoint: {args.ckpt_path}")
    from starVLA.model.framework.base_framework import baseframework
    model = baseframework.from_pretrained(args.ckpt_path)
    model = model.to(args.device).eval()

    # Access the VLM backbone directly
    vlm_model = model.qwen_vl_interface.model  # Qwen3VLForConditionalGeneration
    processor = model.qwen_vl_interface.processor
    tokenizer = processor.tokenizer

    framework_class = type(model).__name__
    vlm_hidden_size = vlm_model.config.text_config.hidden_size
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

        inputs = build_simple_inputs(img, instruction, processor, args.device)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = vlm_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                output_hidden_states=True,
                return_dict=True,
            )

        last_hidden = outputs.hidden_states[-1]
        attention_mask = inputs["attention_mask"]
        input_ids = inputs["input_ids"]

        # feats_A: mean-pool all valid tokens
        f_imgtext = masked_mean_pool(last_hidden, attention_mask).squeeze(0).cpu()

        # feats_A_img: mean-pool image tokens only
        image_mask = (input_ids == IMAGE_TOKEN_INDEX).to(attention_mask.dtype)
        f_img = masked_mean_pool(last_hidden, image_mask).squeeze(0).cpu()

        # feats_A_txt: mean-pool text tokens only
        task_mask = build_task_mask(input_ids[0], tokenizer, instruction).unsqueeze(0)
        f_txt = masked_mean_pool(last_hidden, task_mask).squeeze(0).cpu()

        feats_imgtext_list.append(f_imgtext)
        feats_img_list.append(f_img)
        feats_txt_list.append(f_txt)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{num_samples}]  shape=({vlm_hidden_size},)  "
                  f"rate={rate:.1f}/s  ETA={eta/60:.1f}min")

        if args.save_every > 0 and (i + 1) % args.save_every == 0:
            torch.save(torch.stack(feats_imgtext_list),
                       os.path.join(args.output_dir, "feats_A_partial.pt"))
            torch.save(torch.stack(feats_img_list),
                       os.path.join(args.output_dir, "feats_A_img_partial.pt"))
            torch.save(torch.stack(feats_txt_list),
                       os.path.join(args.output_dir, "feats_A_txt_partial.pt"))
            print(f"    (partial save at {i+1})")

    # Save final results
    for suffix, flist in [("feats_A", feats_imgtext_list),
                          ("feats_A_img", feats_img_list),
                          ("feats_A_txt", feats_txt_list)]:
        t = torch.stack(flist)
        p = os.path.join(args.output_dir, f"{suffix}.pt")
        torch.save(t, p)
        print(f"  Saved {p}  shape={tuple(t.shape)}")

    # Clean up partials
    for partial in ["feats_A_partial.pt", "feats_A_img_partial.pt", "feats_A_txt_partial.pt"]:
        p = os.path.join(args.output_dir, partial)
        if os.path.exists(p):
            os.remove(p)

    # Save metadata
    extraction_meta = {
        "checkpoint": args.ckpt_path,
        "framework": framework_class,
        "model": "StarVLA Qwen3-VL-OFT (Qwen3-VL-4B backbone, 2560D)",
        "extraction_point": "Last hidden state (post-norm) of Qwen3-VL-4B backbone",
        "vlm_hidden_size": vlm_hidden_size,
        "num_samples": num_samples,
        "data_dir": args.data_dir,
        "outputs": ["feats_A.pt", "feats_A_img.pt", "feats_A_txt.pt"],
        "image_token_index": IMAGE_TOKEN_INDEX,
        "pooling": {
            "feats_A": "mean-pool over all valid (non-padding) tokens",
            "feats_A_img": "mean-pool over image tokens only (token_id == 151655)",
            "feats_A_txt": "mean-pool over task instruction tokens only",
        },
    }
    meta_path = os.path.join(args.output_dir, "extraction_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== Phase 2 Complete ===")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()
