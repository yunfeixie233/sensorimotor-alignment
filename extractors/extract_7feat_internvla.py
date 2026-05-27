"""
7-feature extraction for InternVLA-A1-3B-Base (Understanding Expert = Qwen3-VL-2B style).

Model: Qwen3VLForConditionalGeneration with custom dims:
  - vision_config: 24 blocks, hidden 1024, out_hidden 2048 (post-merger)
  - text_config:   28 layers, hidden 2048

Extracts all 7 features in a SINGLE forward pass per sample:
  P1-V   vit_raw:      visual.merger.norm INPUT (1024D, last block output before merger)
  P2-V   pre_llm:      visual.merger OUTPUT (2048D, post-projection image embeds)
  P2-T   pre_llm_txt:  language_model.layers[0] forward_pre_hook, text-only mask
  P2-VT  pre_llm_vt:   language_model.layers[0] forward_pre_hook, all valid tokens
  P3-V   img:          hidden_states[-1] (post-norm, 2048D), image_token mask
  P3-T   txt:          hidden_states[-1], text mask (excludes vision_start, image, vision_end)
  P3-VT  imgtext:      hidden_states[-1], attention_mask

Token layout (matches existing internvla/extract_features.py):
  [vision_start | image_token x N_img | vision_end | text_tokens (max_len=48, right-padded)]

Image token id: 151655
Vision-start token id: 151652
Vision-end token id: 151653

Usage:
  python extractors/extract_7feat_internvla.py \\
      --ckpt /lambda/nfs/vla/cknna_project/checkpoints/internvla-a1-3b \\
      --data_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid \\
      --output_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid_7feat/internvla-a1-3b-base \\
      [--max_samples 5]
"""

import argparse
import json
import os
import sys
import time

import torch
from PIL import Image

torch.backends.cudnn.enabled = False

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Reuse the existing single-feature extractor's loader and input builder
from extractors.internvla.extract_features import (
    load_und_expert_from_checkpoint,
    build_inputs_for_sample,
    VISION_START_TOKEN_ID,
    VISION_END_TOKEN_ID,
    IMAGE_TOKEN_ID,
)


def masked_mean_pool(hidden_states, mask):
    h = hidden_states.float()
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def extract_all_7(model, processor, img, instruction, device):
    """Single forward pass with hooks. Returns dict {key: [D] tensor}."""
    captures = {}

    inputs = build_inputs_for_sample(img, instruction, processor, device)
    input_ids = inputs["input_ids"]      # [1, seq_len]
    attn_mask = inputs["attention_mask"] # [1, seq_len]

    visual = model.model.visual
    merger = visual.merger
    norm_module = merger.norm  # LayerNorm at start of merger
    layer0 = model.model.language_model.layers[0]

    # ── Hook 1: vit_raw — input to merger.norm ──────────────────────────────
    # The norm receives the concatenated last-block output (still 1024D).
    def hook_norm_pre(module, inputs_tup):
        x = inputs_tup[0]
        # Qwen3VL packs all image tokens in dim 0 (no batch dim for vision)
        # x shape: [num_image_tokens, hidden]
        if x.dim() == 2:
            captures["p1_v"] = x.detach().float().mean(dim=0).cpu()
        else:
            captures["p1_v"] = x.detach().float().mean(dim=(0, 1)).cpu()

    h_norm = norm_module.register_forward_pre_hook(hook_norm_pre)

    # ── Hook 2: pre_llm — merger OUTPUT (post-projection) ───────────────────
    def hook_merger(module, inputs_tup, output):
        # output shape: [num_image_tokens // (merge^2), out_hidden=2048]
        if output.dim() == 2:
            captures["p2_v"] = output.detach().float().mean(dim=0).cpu()
        else:
            captures["p2_v"] = output.detach().float().mean(dim=(0, 1)).cpu()

    h_merger = merger.register_forward_hook(hook_merger)

    # ── Hook 3: layers[0] forward_pre_hook (text/imgtext at LLM input) ──────
    def hook_layer0_pre(module, args, kwargs):
        if "hidden_states" in kwargs:
            hidden = kwargs["hidden_states"]
        else:
            hidden = args[0]
        h = hidden.detach().float()  # [1, seq_len, D]

        # P2-VT: mean over all valid (non-padding) tokens
        captures["p2_vt"] = masked_mean_pool(h, attn_mask).squeeze(0).cpu()

        # P2-T: text tokens only (exclude vision_start, image, vision_end)
        is_vision = (
            (input_ids == VISION_START_TOKEN_ID)
            | (input_ids == IMAGE_TOKEN_ID)
            | (input_ids == VISION_END_TOKEN_ID)
        )
        text_mask = attn_mask * (~is_vision).to(attn_mask.dtype)
        captures["p2_t"] = masked_mean_pool(h, text_mask).squeeze(0).cpu()

    h_layer0 = layer0.register_forward_pre_hook(hook_layer0_pre, with_kwargs=True)

    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                output_hidden_states=True,
                return_dict=True,
            )
    finally:
        h_norm.remove()
        h_merger.remove()
        h_layer0.remove()

    # P3-V/T/VT from final hidden state
    last_hidden = outputs.hidden_states[-1]  # [1, seq_len, D]

    captures["p3_vt"] = masked_mean_pool(last_hidden, attn_mask).squeeze(0).cpu()

    image_mask = (input_ids == IMAGE_TOKEN_ID).to(attn_mask.dtype)
    captures["p3_v"] = masked_mean_pool(last_hidden, image_mask).squeeze(0).cpu()

    is_vision = (
        (input_ids == VISION_START_TOKEN_ID)
        | (input_ids == IMAGE_TOKEN_ID)
        | (input_ids == VISION_END_TOKEN_ID)
    )
    text_mask = attn_mask * (~is_vision).to(attn_mask.dtype)
    captures["p3_t"] = masked_mean_pool(last_hidden, text_mask).squeeze(0).cpu()

    return captures


def main():
    parser = argparse.ArgumentParser(
        description="7-feature extraction for InternVLA-A1-3B-Base.")
    parser.add_argument("--ckpt", required=True, help="Local InternVLA-A1 ckpt dir")
    parser.add_argument("--data_dir", required=True, help="CKNNA data dir")
    parser.add_argument("--output_dir", required=True, help="Where to save 7 .pt files")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    if args.max_samples:
        num_samples = min(num_samples, args.max_samples)
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    model, processor = load_und_expert_from_checkpoint(args.ckpt, args.device)
    print(f"  Samples: {num_samples}")
    print(f"  Vision hidden: {model.config.vision_config.hidden_size}")
    print(f"  Text hidden:   {model.config.text_config.hidden_size}")

    keys = ["p1_v", "p2_v", "p2_t", "p2_vt", "p3_v", "p3_t", "p3_vt"]
    accum = {k: [] for k in keys}

    t0 = time.time()
    for i in range(num_samples):
        img = Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
        instruction = task_descriptions[i]

        feats = extract_all_7(model, processor, img, instruction, args.device)

        for k in keys:
            if k not in feats:
                raise RuntimeError(f"Feature {k} missing at sample {i}")
            accum[k].append(feats[k])

        if (i + 1) % 500 == 0:
            torch.cuda.empty_cache()

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (num_samples - i - 1) / rate
            shapes = "  ".join(f"{k}={tuple(feats[k].shape)}" for k in keys)
            print(
                f"  [{i+1}/{num_samples}]  {shapes}  rate={rate:.1f}/s  ETA={eta/60:.1f}min",
                flush=True,
            )

    file_map = {
        "p1_v": "feats_A_vit.pt",
        "p2_v": "feats_A_pre_llm.pt",
        "p2_t": "feats_A_pre_llm_txt.pt",
        "p2_vt": "feats_A_pre_llm_vt.pt",
        "p3_v": "feats_A_img.pt",
        "p3_t": "feats_A_txt.pt",
        "p3_vt": "feats_A.pt",
    }

    print(f"\nSaving {len(keys)} feature tensors:")
    for k in keys:
        tensor = torch.stack(accum[k])
        out_path = os.path.join(args.output_dir, file_map[k])
        torch.save(tensor, out_path)
        print(f"  {file_map[k]:<28} shape={tuple(tensor.shape)}")

    meta = {
        "model": args.ckpt,
        "family": "InternVLA-A1-3B-Base (Qwen3-VL und_expert)",
        "type": "all_7_features",
        "num_samples": num_samples,
        "feature_dims": {k: int(accum[k][0].shape[-1]) for k in keys},
        "file_map": file_map,
    }
    with open(os.path.join(args.output_dir, "extraction_metadata_all7.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone. {elapsed/60:.1f} min total, {num_samples/elapsed:.2f} samples/s")


if __name__ == "__main__":
    main()
