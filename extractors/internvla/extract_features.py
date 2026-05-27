"""
Phase 2: Extract VLM features from InternVLA-A1 Understanding Expert for CKNNA.

Architecture: InternVLA-A1 = Mixture-of-Transformers (MoT) with 3 experts:
  Understanding Expert (und_expert): Qwen3-VL-2B (28 layers, 2048D)
  Generation Expert (gen_expert):    Cosmos VAE + Qwen3-600M (1024D)
  Action Expert (act_expert):        Qwen3-600M (1024D)

Only the Understanding Expert is used for CKNNA feature extraction.
Its weights are extracted from the full MoT checkpoint and loaded into
a standalone Qwen3VLForConditionalGeneration.

Extraction point: Last hidden state (post-norm) of the Understanding Expert.
  Token layout (matching InternVLA-A1 training format):
    [vision_start(151652) | image_tokens(151655 x N_img) | vision_end(151653) | text_tokens(max_length=48, right-padded)]

Pooling:
  feats_A:     mean-pool over all valid (non-padding) tokens
  feats_A_img: mean-pool over image tokens only (token_id == 151655)
  feats_A_txt: mean-pool over non-padding text tokens only

Usage:
    python extract_features.py \
        --ckpt /path/to/InternVLA-A1-3B-RoboTwin \
        --data_dir /path/to/cknna_data \
        --output_dir /path/to/cknna_data/internvla-a1-robotwin
"""

import argparse
import glob
import json
import os
import time

import torch
from PIL import Image

# Token IDs from Qwen3-VL tokenizer (same as InternVLA-A1 transform)
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653
IMAGE_TOKEN_ID = 151655

# InternVLA-A1 training config
TEXT_MAX_LENGTH = 48
SPATIAL_MERGE_SIZE = 2


def masked_mean_pool(hidden_states, mask):
    """Mean-pool hidden states over valid tokens indicated by mask."""
    h = hidden_states.float()
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def _import_qwen3vl():
    """Import Qwen3VL classes, with fallback for older transformers."""
    try:
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
        return Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    except ImportError:
        # transformers < 4.52 doesn't have Qwen3-VL;
        # try importing from InternVLA-A1 repo's custom code
        import sys
        repo_replace = os.path.join(
            os.environ.get("INTERNVLA_ROOT",
                           os.path.join(os.path.dirname(__file__), "..", "..", "repos", "InternVLA-A1")),
            "src", "lerobot", "policies", "InternVLA_A1_3B", "transformers_replace",
        )
        # Register qwen3_vl as a transformers model type
        import transformers
        qwen3_vl_src = os.path.join(repo_replace, "models", "qwen3_vl")
        qwen3_vl_dst = os.path.join(os.path.dirname(transformers.__file__), "models", "qwen3_vl")
        if not os.path.exists(qwen3_vl_dst):
            os.symlink(qwen3_vl_src, qwen3_vl_dst)
            # Create __init__.py to make it importable
            init_path = os.path.join(qwen3_vl_dst, "__init__.py")
            if not os.path.exists(init_path):
                with open(init_path, "w") as f:
                    f.write("from .modeling_qwen3_vl import *\n")
            # Register in CONFIG_MAPPING and MODEL_MAPPING
            from transformers import AutoConfig, AutoModel
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLModel
            )
            if "qwen3_vl" not in transformers.models.auto.configuration_auto.CONFIG_MAPPING_NAMES:
                AutoConfig.register("qwen3_vl", Qwen3VLConfig)
            print("  Registered Qwen3-VL from InternVLA-A1 repo's transformers_replace/")
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        # Build a Qwen3VLProcessor manually from Qwen2.5-VL components
        # (Qwen3-VL processor is functionally identical to Qwen2.5-VL)
        try:
            from transformers import Qwen3VLProcessor
        except ImportError:
            from transformers import Qwen2_5_VLProcessor as Qwen3VLProcessor
        return Qwen3VLForConditionalGeneration, Qwen3VLProcessor


def load_und_expert_from_checkpoint(ckpt_path, device="cuda"):
    """Load the Understanding Expert from an InternVLA-A1 checkpoint.

    The checkpoint stores the full MoT model. We extract only the
    und_expert (Qwen3VLForConditionalGeneration) weights and load them
    into a standalone model.

    Args:
        ckpt_path: Path to checkpoint directory (with safetensors files)
                   or a single safetensors file.
        device: Target device.

    Returns:
        (model, processor) tuple.
    """
    from safetensors.torch import load_file
    Qwen3VLForConditionalGeneration, Qwen3VLProcessor = _import_qwen3vl()

    # 1. Load checkpoint state dict
    if os.path.isdir(ckpt_path):
        st_files = sorted(glob.glob(os.path.join(ckpt_path, "*.safetensors")))
        if not st_files:
            raise FileNotFoundError(f"No safetensors files found in {ckpt_path}")
        state_dict = {}
        for f in st_files:
            state_dict.update(load_file(f, device="cpu"))
        print(f"  Loaded {len(st_files)} safetensors file(s), {len(state_dict)} keys total")
    else:
        state_dict = load_file(ckpt_path, device="cpu")
        print(f"  Loaded single safetensors file, {len(state_dict)} keys total")

    # 2. Filter for und_expert keys
    # The full model hierarchy is:
    #   QwenA1Policy.model (QwenA1) .qwen3_vl_with_expert.und_expert (Qwen3VLForConditionalGeneration)
    # Checkpoint keys start with "model.qwen3_vl_with_expert.und_expert."
    prefix = "model.qwen3_vl_with_expert.und_expert."
    und_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            und_state_dict[k[len(prefix):]] = v

    if not und_state_dict:
        # Try without the outer "model." prefix (some checkpoints save differently)
        prefix = "qwen3_vl_with_expert.und_expert."
        for k, v in state_dict.items():
            if k.startswith(prefix):
                und_state_dict[k[len(prefix):]] = v

    if not und_state_dict:
        # Maybe the checkpoint IS the und_expert already
        # Check if it has typical Qwen3VL keys
        if any(k.startswith("model.language_model.layers.") or k.startswith("model.visual.") for k in state_dict):
            und_state_dict = state_dict
            print("  Checkpoint appears to be a standalone Qwen3VL model")
        else:
            available_prefixes = set()
            for k in list(state_dict.keys())[:50]:
                parts = k.split(".")
                if len(parts) >= 3:
                    available_prefixes.add(".".join(parts[:3]))
            raise ValueError(
                f"Could not find und_expert weights in checkpoint. "
                f"Sample key prefixes: {sorted(available_prefixes)[:10]}"
            )

    print(f"  Extracted {len(und_state_dict)} und_expert keys")

    # 3. Create a Qwen3VLForConditionalGeneration with matching config
    # The und_expert uses the Qwen3-VL-2B architecture (28 layers, 2048D)
    print("  Initializing Qwen3VLForConditionalGeneration...")
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", trust_remote_code=True)
    except Exception:
        # Fallback: construct config from scratch matching Qwen3-VL-2B
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLConfig
        config = Qwen3VLConfig()

    # Match InternVLA-A1's architecture overrides
    if hasattr(config, "text_config"):
        config.text_config.hidden_size = 2048
        config.text_config.intermediate_size = 6144
        config.text_config.num_attention_heads = 16
        config.text_config.head_dim = 128
        config.text_config.num_hidden_layers = 28
        config.text_config.num_key_value_heads = 8
    if hasattr(config, "vision_config"):
        config.vision_config.depth = 24
        config.vision_config.hidden_size = 1024
        config.vision_config.intermediate_size = 4096
        config.vision_config.out_hidden_size = 2048

    model = Qwen3VLForConditionalGeneration(config)

    # 4. Load und_expert weights
    missing, unexpected = model.load_state_dict(und_state_dict, strict=False)
    if missing:
        # Filter out expected missing keys (e.g., lm_head if tied)
        real_missing = [k for k in missing if "lm_head" not in k]
        if real_missing:
            print(f"  WARNING: {len(real_missing)} missing keys (first 5): {real_missing[:5]}")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")

    model = model.eval().to(device)
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # 5. Load processor
    processor = Qwen3VLProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

    return model, processor


def build_inputs_for_sample(image_pil, instruction, processor, device):
    """Build model inputs matching InternVLA-A1's training format.

    Token layout: [vision_start | image_tokens x N_img | vision_end | text_tokens (max_length=48)]

    Args:
        image_pil: PIL Image.
        instruction: Task description string.
        processor: Qwen3VLProcessor instance.
        device: Target device.

    Returns:
        dict with input_ids, attention_mask, pixel_values, image_grid_thw.
    """
    # Process image through Qwen3-VL image processor
    img_inputs = processor.image_processor(
        image_pil,
        do_rescale=True,  # PIL images are [0,255], need rescaling
        return_tensors="pt",
    )
    pixel_values = img_inputs["pixel_values"].to(device)  # [N_patches, patch_dim]
    image_grid_thw = img_inputs["image_grid_thw"].to(device)  # [1, 3]

    # Compute number of image tokens after spatial merge
    num_img_tokens = int(torch.prod(image_grid_thw).item()) // (SPATIAL_MERGE_SIZE ** 2)

    # Build input_ids: [vision_start, image_token x N, vision_end, text_tokens]
    input_ids = (
        [VISION_START_TOKEN_ID]
        + [IMAGE_TOKEN_ID] * num_img_tokens
        + [VISION_END_TOKEN_ID]
    )
    attention_mask = [1] * (num_img_tokens + 2)  # All image tokens are valid

    # Tokenize text instruction
    lang_inputs = processor.tokenizer(
        instruction,
        max_length=TEXT_MAX_LENGTH,
        padding="max_length",
        padding_side="right",
        truncation=True,
        return_attention_mask=True,
    )
    input_ids += lang_inputs["input_ids"]
    attention_mask += lang_inputs["attention_mask"]

    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Extract InternVLA-A1 features for CKNNA.")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to InternVLA-A1 checkpoint directory or safetensors file")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Phase 1 data directory (images/, metadata.json)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save feats_A.pt etc.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples to process (0 = all)")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--resume_from", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load metadata ---
    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    if args.max_samples > 0:
        num_samples = min(num_samples, args.max_samples)

    # --- Load model ---
    print(f"Loading InternVLA-A1 Understanding Expert from: {args.ckpt}")
    model, processor = load_und_expert_from_checkpoint(args.ckpt, args.device)

    D = model.config.text_config.hidden_size
    print(f"  Hidden size: {D}")
    print(f"  Samples to process: {num_samples}")

    # --- Resume support ---
    feats_imgtext_list = []
    feats_img_list = []
    feats_txt_list = []
    start_idx = 0

    if args.resume_from > 0:
        partial_path = os.path.join(args.output_dir, "feats_A_partial.pt")
        if os.path.exists(partial_path):
            feats_imgtext_list = list(torch.load(partial_path, weights_only=True))
            start_idx = len(feats_imgtext_list)
            print(f"  Resumed {start_idx} samples from partial save")
            for suffix, flist in [("feats_A_img_partial.pt", feats_img_list),
                                  ("feats_A_txt_partial.pt", feats_txt_list)]:
                p = os.path.join(args.output_dir, suffix)
                if os.path.exists(p):
                    flist.extend(list(torch.load(p, weights_only=True)))

    # --- Main extraction loop ---
    t0 = time.time()

    for i in range(start_idx, num_samples):
        img = Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
        task = task_descriptions[i]

        inputs = build_inputs_for_sample(img, task, processor, args.device)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                output_hidden_states=True,
                return_dict=True,
            )

        # Last hidden state (post-norm, 2048D)
        last_hidden = outputs.hidden_states[-1]  # [1, seq_len, D]
        attn_mask = inputs["attention_mask"]      # [1, seq_len]
        input_ids = inputs["input_ids"]           # [1, seq_len]

        # feats_A: mean-pool over all valid tokens
        feat_it = masked_mean_pool(last_hidden, attn_mask).squeeze(0).cpu()  # [D]

        # feats_A_img: mean-pool over image tokens only
        image_mask = (input_ids == IMAGE_TOKEN_ID).to(attn_mask.dtype)
        feat_i = masked_mean_pool(last_hidden, image_mask).squeeze(0).cpu()  # [D]

        # feats_A_txt: mean-pool over non-padding text tokens only
        # Text tokens are after vision_end; exclude vision_start, image, vision_end tokens
        is_vision = (
            (input_ids == VISION_START_TOKEN_ID)
            | (input_ids == IMAGE_TOKEN_ID)
            | (input_ids == VISION_END_TOKEN_ID)
        )
        text_mask = attn_mask * (~is_vision).to(attn_mask.dtype)
        feat_t = masked_mean_pool(last_hidden, text_mask).squeeze(0).cpu()  # [D]

        feats_imgtext_list.append(feat_it)
        feats_img_list.append(feat_i)
        feats_txt_list.append(feat_t)

        if (i + 1) % 100 == 0 or i == start_idx:
            elapsed = time.time() - t0
            processed = i - start_idx + 1
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{num_samples}]  D={D}  rate={rate:.1f}/s  ETA={eta/60:.1f}min")

        if args.save_every > 0 and (i + 1) % args.save_every == 0:
            torch.save(torch.stack(feats_imgtext_list),
                       os.path.join(args.output_dir, "feats_A_partial.pt"))
            torch.save(torch.stack(feats_img_list),
                       os.path.join(args.output_dir, "feats_A_img_partial.pt"))
            torch.save(torch.stack(feats_txt_list),
                       os.path.join(args.output_dir, "feats_A_txt_partial.pt"))
            print(f"    (partial save at {i+1})")

    # --- Save final results ---
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

    # --- Save metadata ---
    extraction_meta = {
        "model": "InternVLA-A1 Understanding Expert (Qwen3-VL-2B, fine-tuned)",
        "checkpoint": args.ckpt,
        "extraction_point": "Last hidden state (post-norm) of Qwen3-VL-2B Understanding Expert",
        "architecture": "MoT: und_expert=Qwen3-VL-2B(28L,2048D), gen_expert=Qwen3-600M, act_expert=Qwen3-600M",
        "hidden_size": D,
        "token_layout": f"[vision_start | image_tokens(151655) | vision_end | text(max_len={TEXT_MAX_LENGTH})]",
        "pooling": {
            "feats_A": "mean-pool over all valid (non-padding) tokens",
            "feats_A_img": "mean-pool over image tokens only (token_id == 151655)",
            "feats_A_txt": "mean-pool over non-padding text tokens only",
        },
        "num_samples": num_samples,
        "outputs": ["feats_A.pt", "feats_A_img.pt", "feats_A_txt.pt"],
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== InternVLA-A1 Phase 2 Complete ===")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()
