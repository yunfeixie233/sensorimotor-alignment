"""
Extract vision-encoder intermediate features (L1 + L2) from raw KosMos-2.

KosMos-2 visual pipeline:
  Image (224px) → CLIP ViT-L/14 encoder (1024-dim, 256 patches + CLS)
                → post_layernorm (LayerNorm on all tokens) + normalize     [L1 vit_raw]
                → image_to_text_projection resampler (64 tokens, 2048-dim) [L2 pre_llm]
                → MAGNETO LM (2048-dim)                                    [L3 post_llm, existing]

Internal path (confirmed from transformers source):
  model.vision_model.model.post_layernorm → applied to last_hidden_state before projection
  model.image_to_text_projection          → Kosmos2ImageToTextProjection (resampler, 64 tokens)

L1: hook model.vision_model.model.post_layernorm
    output: (1, N_patches+1, 1024) — skip CLS (idx 0), mean over spatial patches → (1024,)
    Consistent with PaliGemma-1 which also hooks post_layernorm on SigLIP ViT
L2: hook model.image_to_text_projection
    output: (1, 64, 2048) → mean over 64 tokens → (2048,)

Checkpoint: microsoft/kosmos-2-patch14-224

Usage:
  python extractors/extract_vit_levels_kosmos2.py \\
      --model_path microsoft/kosmos-2-patch14-224 \\
      --data_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_exp1v2_full_L \\
      --output_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_exp1v2_full_L/kosmos2-raw
"""

import argparse
import json
import os
import time

import torch
from PIL import Image


def load_kosmos2(model_path):
    from transformers import Kosmos2ForConditionalGeneration, Kosmos2Processor
    model = Kosmos2ForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    )
    processor = Kosmos2Processor.from_pretrained(model_path)
    return model, processor


def build_inputs(processor, img, instruction):
    prompt = f"<grounding>{instruction}"
    return processor(text=prompt, images=img, return_tensors="pt")


def probe_structure(model):
    """Print top-level submodules and confirm hook targets."""
    print("  KosMos-2 top-level submodules:")
    for name, module in model.named_children():
        print(f"    .{name}: {type(module).__name__}")
    print("  Hook targets:")
    print(f"    L1: model.vision_model.model.post_layernorm "
          f"({type(model.vision_model.model.post_layernorm).__name__})")
    print(f"    L2: model.image_to_text_projection "
          f"({type(model.image_to_text_projection).__name__})")


def extract_vit_levels(model, batch, device):
    """Return (feat_vit_raw, feat_pre_llm) via forward hooks.

    Hook targets (confirmed from transformers source):
      L1 vit_raw:   model.vision_model.model.post_layernorm
                    → applied to (1, N_patches+1, 1024) before normalization + projection
                    → skip CLS (idx 0), mean over spatial patches → (1024,)
                    (consistent with PaliGemma's post_layernorm hook)

      L2 pre_llm:   model.image_to_text_projection
                    → Kosmos2ImageToTextProjection (resampler to 64 tokens)
                    → output: (1, 64, 2048) → mean → (2048,)
    """
    captures = {}

    def hook_post_layernorm(module, input, output):
        # post_layernorm fires TWICE per forward pass in Kosmos2ForConditionalGeneration:
        #   1st call (inside Kosmos2VisionTransformer.forward): pooled CLS only → shape (1, 1024) [2D]
        #   2nd call (in Kosmos2ForConditionalGeneration.forward): all patches → shape (1, 257, 1024) [3D]
        # We want the 2nd call (all patches for spatial mean-pool), so skip 2D outputs.
        t = output[0] if isinstance(output, tuple) else output
        if t.dim() != 3:
            return  # Skip the CLS-only pooled call
        # t: (1, N_patches+1, 1024) — skip CLS token at position 0
        patches = t[:, 1:, :].detach().float()  # (1, N_patches, 1024)
        captures["vit_raw"] = patches.mean(dim=1).squeeze(0).cpu()  # (1024,)

    def hook_projector(module, input, output):
        # output: (1, 64, 2048) projected image tokens
        t = output[0] if isinstance(output, tuple) else output
        if t.dim() == 2:
            t = t.unsqueeze(0)
        captures["pre_llm"] = t.detach().float().mean(dim=1).squeeze(0).cpu()  # (2048,)

    # Kosmos2ForConditionalGeneration top-level attributes (NOT under .model):
    #   model.vision_model                        → Kosmos2VisionModel
    #   model.vision_model.model.post_layernorm   → final ViT LayerNorm (L1 hook)
    #   model.image_to_text_projection            → resampler to LM space (L2 hook)
    h1 = model.vision_model.model.post_layernorm.register_forward_hook(hook_post_layernorm)
    h2 = model.image_to_text_projection.register_forward_hook(hook_projector)

    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            image_embeds_position_mask=batch["image_embeds_position_mask"],
            return_dict=True,
        )

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

    print(f"Loading KosMos-2: {args.model_path}")
    model, processor = load_kosmos2(args.model_path)
    model = model.to(args.device).eval()

    probe_structure(model)
    print(f"  Samples: {num_samples}")

    feats_vit = []
    feats_pre_llm = []

    t0 = time.time()
    for i in range(num_samples):
        img = Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
        instruction = task_descriptions[i]
        batch = build_inputs(processor, img, instruction)
        f_vit, f_pre = extract_vit_levels(model, batch, args.device)
        if f_vit is None or f_pre is None:
            raise RuntimeError(
                f"Hook failed at sample {i}. "
                "Expected: model.vision_model (L1), model.image_to_text_projection (L2). "
                "Check probe_structure() output above."
            )
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
