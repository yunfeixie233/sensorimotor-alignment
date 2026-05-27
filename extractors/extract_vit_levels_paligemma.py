"""
Extract vision-encoder intermediate features (L1 + L2) from raw PaliGemma-1 / PaliGemma-2.

Intervention levels:
  L1 vit_raw  — SigLIP ViT post_layernorm output, BEFORE projector
                shape: (256, 1152)  → mean-pool → (1152,)
  L2 pre_llm  — after PaliGemmaMultiModalProjector Linear(1152→2048)
                shape: (256, 2048)  → mean-pool → (2048,)

PaliGemma-1: google/paligemma-3b-pt-224   → SigLIP So400m, 1152-dim, 256 patches
PaliGemma-2: google/paligemma2-3b-pt-224  → SigLIP So400m (same), 1152-dim, 256 patches

The existing feats_A.pt (L3 post_llm) is produced by extract_raw_paligemma.py.

Usage:
  python extractors/extract_vit_levels_paligemma.py \\
      --model_path google/paligemma-3b-pt-224 \\
      --data_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_exp1v2_full_L \\
      --output_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_exp1v2_full_L/paligemma1-raw
"""

import argparse
import json
import os
import time

import torch
from PIL import Image


def load_paligemma(model_path):
    from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, token=os.environ.get("HF_TOKEN")
    )
    processor = AutoProcessor.from_pretrained(
        model_path, token=os.environ.get("HF_TOKEN")
    )
    return model, processor


def build_inputs_paligemma(processor, img, instruction):
    return processor(
        text=instruction,
        images=img,
        return_tensors="pt",
        padding=True,
    )


def extract_vit_levels(model, batch, device):
    """Return (feat_vit_raw, feat_pre_llm) via forward hooks.

    PaliGemma pipeline:
      SigLIP ViT → post_layernorm → (1, 256, 1152)   [L1]
                → projector (Linear) → (1, 256, 2048) [L2]
                → concat with text → Gemma LLM        [L3, existing]
    """
    captures = {}

    def hook_post_ln(module, input, output):
        t = output[0] if isinstance(output, tuple) else output
        captures["vit_raw"] = t.detach().float().mean(dim=1).squeeze(0).cpu()

    def hook_projector(module, input, output):
        t = output[0] if isinstance(output, tuple) else output
        if t.dim() == 2:
            t = t.unsqueeze(0)
        captures["pre_llm"] = t.detach().float().mean(dim=1).squeeze(0).cpu()

    post_ln = model.model.vision_tower.vision_model.post_layernorm
    projector = model.model.multi_modal_projector
    h1 = post_ln.register_forward_hook(hook_post_ln)
    h2 = projector.register_forward_hook(hook_projector)

    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        model(**batch, return_dict=True)

    h1.remove()
    h2.remove()

    return captures.get("vit_raw"), captures.get("pre_llm")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    print(f"Loading PaliGemma: {args.model_path}")
    model, processor = load_paligemma(args.model_path)
    model = model.to(args.device).eval()

    # Probe dimensions
    post_ln = model.model.vision_tower.vision_model.post_layernorm
    proj = model.model.multi_modal_projector
    print(f"  post_layernorm: {post_ln}")
    print(f"  projector:      {proj}")
    print(f"  Samples: {num_samples}")

    feats_vit = []
    feats_pre_llm = []

    t0 = time.time()
    for i in range(num_samples):
        img = Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
        instruction = task_descriptions[i]
        batch = build_inputs_paligemma(processor, img, instruction)
        f_vit, f_pre = extract_vit_levels(model, batch, args.device)
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
        "model": args.model_path,
        "type": "vit_level_features",
        "vit_dim": vit_t.shape[-1], "pre_llm_dim": pre_t.shape[-1],
        "num_samples": num_samples,
    }
    with open(os.path.join(args.output_dir, "extraction_metadata_vit.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
