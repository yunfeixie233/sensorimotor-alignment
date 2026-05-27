"""
Extract vision-encoder intermediate features (L1 + L2) from raw Qwen2.5-VL / Qwen3-VL models.

Intervention levels:
  L1 vit_raw  — final ViT LayerNorm output, BEFORE spatial merge + MLP projection
                Qwen2.5VL: merger.ln_q (RMSNorm on blocks[-1] output)
                Qwen3VL:   merger.norm (LayerNorm on blocks[-1] output)
                This is the conceptual equivalent of SigLIP/CLIP post_layernorm.
                Shape: (N_patches, D_vit) → mean-pool → (D_vit,)
                Dims: Qwen2.5=1280, Qwen3-2B/4B=1024, Qwen3-8B=1152
  L2 pre_llm  — full merger output, BEFORE LLM  (after spatial merge + MLP projection)
                Shape: (N_merged, D_llm) → mean-pool → (D_llm,)
                Dims: 2B→2048, 3B→2048, 4B→2560, 7B→3584, 8B→4096

The existing feats_A.pt (L3 post_llm) is produced by extract_raw_qwen.py.

Outputs (saved in --output_dir):
  feats_A_vit.pt    — (N, 1280) L1 vit_raw
  feats_A_pre_llm.pt — (N, D_merger)  L2 pre_llm

Usage:
  python extractors/extract_vit_levels_qwen.py \\
      --model_path Qwen/Qwen2.5-VL-3B-Instruct \\
      --data_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_exp1v2_full_L \\
      --output_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_exp1v2_full_L/qwen25vl-3b-raw \\
      --model_family qwen2.5
"""

import argparse
import json
import os
import time

import torch
from PIL import Image


def load_qwen25(model_path):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, attn_implementation="sdpa", torch_dtype="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"
    return model, processor


def load_qwen3(model_path):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, attn_implementation="sdpa", torch_dtype="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"
    return model, processor


def build_inputs_qwen25(processor, img, instruction):
    import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                                  '../Qwen3-VL/qwen-vl-utils/src/'))
    from qwen_vl_utils import process_vision_info
    messages = [[{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": instruction},
    ]}]]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
             for m in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(text=texts, images=image_inputs, videos=video_inputs,
                     padding=True, return_tensors="pt")


def build_inputs_qwen3(processor, img, instruction):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": instruction},
    ]}]
    return processor.apply_chat_template(
        [messages], tokenize=True, padding=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )


def extract_vit_levels(model, batch, device):
    """Return (feat_vit_raw, feat_pre_llm) via forward hooks.

    feat_vit_raw: mean-pool over patches from last ViT block output  → (D_vit,)
    feat_pre_llm: mean-pool over merged visual tokens (merger output) → (D_merger,)

    L1 hook: merger.ln_q (Qwen2.5VL) or merger.norm (Qwen3VL)
             — final LayerNorm on ViT output BEFORE spatial merge+projection
             — conceptually equivalent to SigLIP/CLIP post_layernorm
             — output shape: (N_patches, D_vit); mean-pool → (D_vit,)
    L2 hook: merger (full module output)
             — shape: (N_merged, D_llm); mean-pool → (D_llm,)
    """
    captures = {}

    def hook_ln(module, input, output):
        # output: (N_patches, D_vit) — NaViT packed format, no batch dim
        t = output[0] if isinstance(output, tuple) else output
        # t is 2D: (N_patches, D_vit); mean over patches → (D_vit,)
        captures["vit_raw"] = t.detach().float().mean(dim=0).cpu()

    def hook_merger(module, input, output):
        # output: (N_merged, D_llm) — after spatial merge + MLP projection
        t = output[0] if isinstance(output, tuple) else output
        captures["pre_llm"] = t.detach().float().mean(dim=0).cpu()

    # L1: hook the final LayerNorm BEFORE spatial merge
    #   Qwen2.5VL: merger.ln_q  (RMSNorm, applied as: self.mlp(self.ln_q(x).view(...)))
    #   Qwen3VL:   merger.norm  (LayerNorm, applied as: self.norm(x).view(...) before linear)
    if hasattr(model.model.visual.merger, "ln_q"):
        ln_module = model.model.visual.merger.ln_q      # Qwen2.5VL
    else:
        ln_module = model.model.visual.merger.norm       # Qwen3VL
    merger = model.model.visual.merger
    h1 = ln_module.register_forward_hook(hook_ln)
    h2 = merger.register_forward_hook(hook_merger)

    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(**batch, output_hidden_states=False, return_dict=True)

    h1.remove()
    h2.remove()

    return captures.get("vit_raw"), captures.get("pre_llm")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_family", required=True, choices=["qwen2.5", "qwen3"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    print(f"Loading {args.model_family} model: {args.model_path}")
    if args.model_family == "qwen2.5":
        model, processor = load_qwen25(args.model_path)
        build_inputs = build_inputs_qwen25
    else:
        model, processor = load_qwen3(args.model_path)
        build_inputs = build_inputs_qwen3

    model = model.to(args.device).eval()
    merger = model.model.visual.merger
    ln_name = "merger.ln_q" if hasattr(merger, "ln_q") else "merger.norm"
    vit_dim = merger.ln_q.weight.shape[0] if hasattr(merger, "ln_q") else merger.norm.weight.shape[0]
    print(f"  ViT blocks: {len(model.model.visual.blocks)},  vit_dim: {vit_dim},  L1 hook: {ln_name}", flush=True)
    print(f"  Samples: {num_samples}", flush=True)

    feats_vit = []
    feats_pre_llm = []

    t0 = time.time()
    for i in range(num_samples):
        img = Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
        instruction = task_descriptions[i]
        batch = build_inputs(processor, img, instruction)
        f_vit, f_pre = extract_vit_levels(model, batch, args.device)
        if f_vit is None or f_pre is None:
            raise RuntimeError(f"Hook failed at sample {i} — check model structure")
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
        "model": args.model_path, "model_family": args.model_family,
        "type": "vit_level_features",
        "vit_dim": vit_t.shape[-1], "pre_llm_dim": pre_t.shape[-1],
        "num_samples": num_samples,
    }
    with open(os.path.join(args.output_dir, "extraction_metadata_vit.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
