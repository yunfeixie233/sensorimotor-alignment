"""
Extract vision-encoder intermediate features (L1 + L2) from finetuned StarVLA models
(Qwen-GR00T, Qwen-OFT, Qwen3VL-GR00T, Qwen3VL-OFT).

These models wrap Qwen2.5-VL or Qwen3-VL as:
    model.qwen_vl_interface.model  →  Qwen{2.5,3}VLForConditionalGeneration

The ViT hooks are identical to extract_vit_levels_qwen.py but accessed through
the StarVLA wrapper. ViT is NOT frozen during finetuning (freeze_modules=''),
so L1/L2 features differ from the raw base VLM.

L1 vit_raw:  merger.ln_q (Qwen2.5) / merger.norm (Qwen3)  → mean-pool → (D_vit,)
L2 pre_llm:  merger output → mean-pool → (D_llm,)

Usage:
    STARVLA_ROOT=/home/ubuntu/vla/cknna_project/repos/starVLA \\
    python extractors/extract_vit_levels_starvla.py \\
        --ckpt_path /path/to/steps_XXXXX_pytorch_model.pt \\
        --data_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid \\
        --output_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid/Qwen-GR00T-Bridge \\
        --device cuda
"""

import argparse
import json
import os
import sys
import time

import torch
from PIL import Image

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STARVLA_ROOT = os.environ.get("STARVLA_ROOT", os.path.join(_PROJECT_ROOT, "repos", "starVLA"))
if STARVLA_ROOT not in sys.path:
    sys.path.insert(0, STARVLA_ROOT)


def build_simple_inputs(image_pil, instruction, processor, device):
    """Build Qwen-VL inputs directly (avoids apply_chat_template compatibility issues)."""
    img_inputs = processor.image_processor(image_pil, return_tensors="pt")
    pixel_values = img_inputs["pixel_values"].to(device)
    image_grid_thw = img_inputs["image_grid_thw"].to(device)

    spatial_merge_size = getattr(processor.image_processor, 'merge_size', 2)
    num_img_tokens = int(torch.prod(image_grid_thw).item()) // (spatial_merge_size ** 2)

    tokenizer = processor.tokenizer
    vision_start_id = getattr(processor, 'vision_start_token_id', 151652)
    vision_end_id = getattr(processor, 'vision_end_token_id', 151653)
    image_token_id = getattr(processor, 'image_token_id', 151655)

    input_ids = (
        [vision_start_id]
        + [image_token_id] * num_img_tokens
        + [vision_end_id]
    )
    text_ids = tokenizer.encode(instruction, add_special_tokens=False)
    input_ids += text_ids

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long, device=device),
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }


def extract_vit_levels(vlm_model, batch, device):
    """Return (feat_vit_raw, feat_pre_llm) via forward hooks on the Qwen VLM backbone.

    vlm_model is model.qwen_vl_interface.model (Qwen{2.5,3}VLForConditionalGeneration).
    Hooks target model.visual.merger (one less .model level than standalone Qwen scripts
    because we already have the VLM backbone directly).
    """
    captures = {}

    def hook_ln(module, input, output):
        t = output[0] if isinstance(output, tuple) else output
        captures["vit_raw"] = t.detach().float().mean(dim=0).cpu()

    def hook_merger(module, input, output):
        t = output[0] if isinstance(output, tuple) else output
        captures["pre_llm"] = t.detach().float().mean(dim=0).cpu()

    # vlm_model is Qwen{2.5,3}VLForConditionalGeneration
    # Access: vlm_model.model.visual.merger  (same as model.model.visual.merger in standalone)
    merger = vlm_model.model.visual.merger
    if hasattr(merger, "ln_q"):
        ln_module = merger.ln_q        # Qwen2.5VL
        ln_name = "merger.ln_q"
    else:
        ln_module = merger.norm         # Qwen3VL
        ln_name = "merger.norm"

    h1 = ln_module.register_forward_hook(hook_ln)
    h2 = merger.register_forward_hook(hook_merger)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        vlm_model(**batch, output_hidden_states=False, return_dict=True)

    h1.remove()
    h2.remove()

    return captures.get("vit_raw"), captures.get("pre_llm"), ln_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True, help="StarVLA .pt checkpoint")
    parser.add_argument("--data_dir", required=True, help="CKNNA data dir (images/, metadata.json)")
    parser.add_argument("--output_dir", required=True, help="Where to save feats_A_vit.pt, feats_A_pre_llm.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    # Load StarVLA model
    print(f"Loading StarVLA checkpoint: {args.ckpt_path}")
    from starVLA.model.framework.base_framework import baseframework
    starvla_model = baseframework.from_pretrained(args.ckpt_path)
    starvla_model = starvla_model.to(args.device).eval()

    # Access VLM backbone
    vlm_model = starvla_model.qwen_vl_interface.model
    processor = starvla_model.qwen_vl_interface.processor

    framework_class = type(starvla_model).__name__
    vlm_hidden_size = vlm_model.config.text_config.hidden_size
    merger = vlm_model.model.visual.merger
    vit_dim = merger.ln_q.weight.shape[0] if hasattr(merger, "ln_q") else merger.norm.weight.shape[0]

    print(f"  Framework: {framework_class}")
    print(f"  VLM hidden size: {vlm_hidden_size}")
    print(f"  ViT dim: {vit_dim}")
    print(f"  Samples: {num_samples}", flush=True)

    feats_vit = []
    feats_pre_llm = []

    t0 = time.time()
    for i in range(num_samples):
        img = Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
        instruction = task_descriptions[i]
        batch = build_simple_inputs(img, instruction, processor, args.device)
        f_vit, f_pre, ln_name = extract_vit_levels(vlm_model, batch, args.device)
        if f_vit is None or f_pre is None:
            raise RuntimeError(f"Hook failed at sample {i}")
        feats_vit.append(f_vit)
        feats_pre_llm.append(f_pre)

        if (i + 1) % 500 == 0:
            torch.cuda.empty_cache()

        if (i + 1) % 200 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (num_samples - i - 1) / rate
            print(f"  [{i+1}/{num_samples}]  vit={tuple(f_vit.shape)}  "
                  f"pre_llm={tuple(f_pre.shape)}  rate={rate:.1f}/s  ETA={eta/60:.1f}min",
                  flush=True)

    vit_t = torch.stack(feats_vit)
    pre_t = torch.stack(feats_pre_llm)
    torch.save(vit_t, os.path.join(args.output_dir, "feats_A_vit.pt"))
    torch.save(pre_t, os.path.join(args.output_dir, "feats_A_pre_llm.pt"))
    print(f"\nSaved feats_A_vit.pt     shape={tuple(vit_t.shape)}")
    print(f"Saved feats_A_pre_llm.pt shape={tuple(pre_t.shape)}")

    meta = {
        "checkpoint": args.ckpt_path,
        "framework": framework_class,
        "type": "vit_level_features",
        "vit_dim": vit_t.shape[-1], "pre_llm_dim": pre_t.shape[-1],
        "num_samples": num_samples,
    }
    with open(os.path.join(args.output_dir, "extraction_metadata_vit.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
