"""
7-feature extraction for CogACT models (Prismatic VLM backbone: DINOv2+SigLIP + Llama-2-7b).

Extracts all 7 VLM features in a SINGLE forward pass per sample:
  P1-V   vit_raw:      vision_backbone forward hook -> DINOv2+SigLIP concat 2176D
  P2-V   pre_llm:      projector (FusedMLPProjector) forward hook -> 4096D
  P2-T   pre_llm_txt:  llm_backbone.llm.model.layers[0] pre-hook, text-only mask
  P2-VT  pre_llm_vt:   llm_backbone.llm.model.layers[0] pre-hook, all tokens mean-pooled
  P3-V   img:           hidden_states[-1], image mask
  P3-T   txt:           hidden_states[-1], task mask
  P3-VT  imgtext:       hidden_states[-1], attention mask

Module paths (runtime-verified, differ from OpenVLA!):
  - vlm.vision_backbone: DINOv2 + SigLIP concat (2176D)
  - vlm.projector: FusedMLPProjector (4096D)
  - vlm.llm_backbone.llm.model.layers[0]: first Llama decoder layer
  - featurizer is None on DinoSigLIPViTBackbone (use vision_backbone directly)

3 model variants:
  CogACT/CogACT-Small  --action_model_type DiT-S
  CogACT/CogACT-Base   --action_model_type DiT-B
  CogACT/CogACT-Large  --action_model_type DiT-L

Env: cogact conda env, PYTHONNOUSERSITE=1, COGACT_ROOT env var

Usage:
  COGACT_ROOT=/home/ubuntu/vla/cknna_project/repos/CogACT \
  PYTHONNOUSERSITE=1 \
  python extractors/extract_7feat_cogact.py \
      --ckpt CogACT/CogACT-Base \
      --action_model_type DiT-B \
      --data_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid \
      --output_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid_7feat/CogACT-Base \
      [--max_samples 10]
"""

import argparse
import json
import os
import sys
import time

import torch
from PIL import Image
from transformers import LlamaTokenizerFast

torch.backends.cudnn.enabled = False

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_COGACT_ROOT = os.environ.get("COGACT_ROOT", os.path.join(_PROJECT_ROOT, "repos", "CogACT"))
if _COGACT_ROOT not in sys.path:
    sys.path.insert(0, _COGACT_ROOT)


# ---------------------------------------------------------------------------
# Redirect meta-llama/Llama-2-7b-hf to local cache (must happen before import)
# ---------------------------------------------------------------------------
def _redirect_llama_to_local_cache():
    """Redirect meta-llama/Llama-2-7b-hf to a pre-populated local cache directory.

    Prismatic's load() calls AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
    which fails without gated-model access. This patches from_pretrained to use
    a local cache created by scripts/setup_prismatic_cache.sh.
    """
    from transformers import AutoConfig, AutoTokenizer

    llama_cache = os.environ.get(
        "PRISMATIC_LLAMA_CACHE",
        os.path.expanduser("~/.cache/huggingface/llama2-7b-config"),
    )
    if not os.path.isdir(llama_cache):
        return
    if not os.path.isfile(os.path.join(llama_cache, "config.json")):
        return

    _LLAMA_HF_PATHS = {"meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-7b-chat-hf"}

    _orig_config_fp = AutoConfig.from_pretrained.__func__

    @classmethod
    def _patched_config_fp(cls, pretrained_model_name_or_path, *a, **kw):
        if pretrained_model_name_or_path in _LLAMA_HF_PATHS:
            pretrained_model_name_or_path = llama_cache
        return _orig_config_fp(cls, pretrained_model_name_or_path, *a, **kw)

    AutoConfig.from_pretrained = _patched_config_fp

    _orig_tok_fp = AutoTokenizer.from_pretrained.__func__

    @classmethod
    def _patched_tok_fp(cls, pretrained_model_name_or_path, *a, **kw):
        if pretrained_model_name_or_path in _LLAMA_HF_PATHS:
            pretrained_model_name_or_path = llama_cache
        return _orig_tok_fp(cls, pretrained_model_name_or_path, *a, **kw)

    AutoTokenizer.from_pretrained = _patched_tok_fp

    print(f"[cogact] Redirected meta-llama/Llama-2-7b-hf -> {llama_cache}")


_redirect_llama_to_local_cache()

from vla import load_vla  # noqa: E402  (must be after llama redirect + sys.path)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def masked_mean_pool(hidden_states, mask):
    """Mean-pool hidden states over positions where mask == 1."""
    h = hidden_states.float()
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def find_subsequence(seq, subseq):
    n, m = len(seq), len(subseq)
    if m == 0:
        return -1
    for i in range(n - m + 1):
        if seq[i : i + m] == subseq:
            return i
    return -1


def build_task_mask_prismatic(input_ids_1d, tokenizer, task, num_patches):
    """Build task-instruction-only mask in multimodal sequence space.

    The multimodal sequence is [BOS, vision_patches, text_tokens_after_BOS],
    so text token positions are shifted by num_patches relative to input_ids.
    """
    multimodal_len = len(input_ids_1d) + num_patches
    device = input_ids_1d.device
    mask = torch.zeros(multimodal_len, dtype=torch.long, device=device)
    if not task:
        return mask
    ids_list = input_ids_1d.tolist()
    for prefix in ["", " "]:
        task_ids = tokenizer.encode(prefix + task, add_special_tokens=False)
        start_in_text = find_subsequence(ids_list, task_ids)
        if start_in_text >= 0:
            mm_start = start_in_text + num_patches
            mask[mm_start : mm_start + len(task_ids)] = 1
            return mask
    return mask


# ---------------------------------------------------------------------------
# Core 7-feature extraction
# ---------------------------------------------------------------------------
def extract_all_7(vlm, input_ids, attention_mask, pixel_values, tokenizer,
                  task, device, autocast_dtype):
    """Extract all 7 features in one forward pass.

    Returns:
        captures: dict of (D,) tensors for 7 VLM features
    """
    captures = {}

    # --- Hook targets (runtime-verified module paths) ---
    vision_backbone = vlm.vision_backbone     # DinoSigLIPViTBackbone -> concat 2176D
    projector = vlm.projector                 # FusedMLPProjector -> 4096D
    layer0 = vlm.llm_backbone.llm.model.layers[0]  # first LlamaDecoderLayer

    # We need num_patches for masking; compute it inside the forward pass
    # by capturing projector output shape.

    # --- P1-V: raw ViT output (DINOv2 + SigLIP concat, 2176D) ---
    def hook_p1_v(module, input, output):
        # VisionBackbone.forward() returns concatenated features: (1, num_patches, 2176)
        t = output[0] if isinstance(output, tuple) else output
        captures["p1_v"] = t.detach().float().mean(dim=1).squeeze(0).cpu()

    # --- P2-V: projector output (4096D) ---
    def hook_p2_v(module, input, output):
        t = output[0] if isinstance(output, tuple) else output
        captures["p2_v"] = t.detach().float().mean(dim=1).squeeze(0).cpu()

    # --- P2-T and P2-VT: LLM layer 0 pre-hook ---
    def hook_layer0_pre(module, args):
        t = args[0]  # (1, seq_len, 4096)
        hidden = t.detach().float()
        # P2-VT: mean-pool over all positions
        captures["p2_vt"] = hidden.mean(dim=1).squeeze(0).cpu()

        # P2-T: text-only positions
        # At this point, the multimodal sequence is already formed.
        # Compute num_patches from the sequence length difference.
        mm_seq_len = hidden.shape[1]
        n_patches = mm_seq_len - input_ids.shape[1]

        # Build multimodal attention mask
        patch_mask = torch.ones(
            (1, n_patches), dtype=attention_mask.dtype, device=attention_mask.device
        )
        mm_mask = torch.cat(
            [attention_mask[:, :1], patch_mask, attention_mask[:, 1:]], dim=1
        )

        # Image mask: patches are at positions 1..num_patches
        img_mask = torch.zeros_like(mm_mask)
        img_mask[0, 1 : 1 + n_patches] = 1

        # Text mask = multimodal_mask - image_mask
        text_mask = (mm_mask - img_mask).clamp(min=0)
        captures["p2_t"] = masked_mean_pool(hidden, text_mask).squeeze(0).cpu()

    h1 = vision_backbone.register_forward_hook(hook_p1_v)
    h2 = projector.register_forward_hook(hook_p2_v)
    h3 = layer0.register_forward_pre_hook(hook_layer0_pre)

    with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype):
        output = vlm(
            input_ids=input_ids,
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

    h1.remove()
    h2.remove()
    h3.remove()

    # --- P3 features from LLM final hidden states ---
    last_hidden = output.hidden_states[-1]

    text_seq_len = input_ids.shape[1]
    multimodal_seq_len = last_hidden.shape[1]
    num_patches = multimodal_seq_len - text_seq_len

    # Multimodal attention mask: [BOS, patches (all 1), text_after_BOS]
    patch_mask = torch.ones(
        (1, num_patches), dtype=attention_mask.dtype, device=attention_mask.device
    )
    multimodal_mask = torch.cat(
        [attention_mask[:, :1], patch_mask, attention_mask[:, 1:]], dim=1
    )

    # P3-VT: all valid tokens
    captures["p3_vt"] = masked_mean_pool(last_hidden, multimodal_mask).squeeze(0).cpu()

    # P3-V: image patches only
    image_mask = torch.zeros_like(multimodal_mask)
    image_mask[0, 1 : 1 + num_patches] = 1
    captures["p3_v"] = masked_mean_pool(last_hidden, image_mask).squeeze(0).cpu()

    # P3-T: task instruction tokens only
    task_mask = build_task_mask_prismatic(
        input_ids[0], tokenizer, task.lower(), num_patches
    ).unsqueeze(0)
    captures["p3_t"] = masked_mean_pool(last_hidden, task_mask).squeeze(0).cpu()

    return captures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="7-feature extraction for CogACT.")
    parser.add_argument("--ckpt", type=str, default="CogACT/CogACT-Base",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--action_model_type", type=str, default="DiT-B",
                        help="DiT variant: DiT-S (Small), DiT-B (Base), DiT-L (Large)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="CKNNA data directory (images/, metadata.json)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save 7 .pt feature files")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load metadata ---
    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    if args.max_samples:
        num_samples = min(num_samples, args.max_samples)
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    # --- Load model ---
    print(f"Loading CogACT: {args.ckpt} (action_model_type={args.action_model_type})")
    vla = load_vla(
        args.ckpt,
        load_for_training=False,
        action_model_type=args.action_model_type,
    )
    vla.vlm = vla.vlm.to(torch.bfloat16)
    vla = vla.to(args.device).eval()

    vlm = vla.vlm
    tokenizer = vlm.llm_backbone.tokenizer
    image_transform = vlm.vision_backbone.image_transform
    hidden_size = vlm.llm_backbone.llm.config.hidden_size
    autocast_dtype = vlm.llm_backbone.half_precision_dtype

    assert isinstance(tokenizer, LlamaTokenizerFast), (
        f"Expected LlamaTokenizerFast, got {type(tokenizer)}"
    )

    # Determine ViT dim from vision backbone
    dino_dim = vlm.vision_backbone.dino_featurizer.embed_dim
    siglip_dim = vlm.vision_backbone.siglip_featurizer.embed_dim
    vit_dim = dino_dim + siglip_dim  # 1024 + 1152 = 2176
    projector_out_dim = hidden_size  # FusedMLPProjector outputs LLM hidden size

    print(f"  hidden_size (LLM): {hidden_size}")
    print(f"  ViT dim (DINOv2+SigLIP): {vit_dim} ({dino_dim}+{siglip_dim})")
    print(f"  projector out dim: {projector_out_dim}")
    print(f"  autocast dtype: {autocast_dtype}")
    print(f"  tokenizer type: {type(tokenizer).__name__}")
    print(f"  Samples to process: {num_samples}")

    # --- Feature accumulation ---
    keys = ["p1_v", "p2_v", "p2_t", "p2_vt", "p3_v", "p3_t", "p3_vt"]
    accum = {k: [] for k in keys}

    t0 = time.time()
    for i in range(num_samples):
        img = Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
        task = task_descriptions[i]

        # Build prompt (same format as CogACT predict_action)
        prompt_builder = vlm.get_prompt_builder()
        prompt_builder.add_turn(
            role="human",
            message=f"What action should the robot take to {task.lower()}?",
        )
        prompt_text = prompt_builder.get_prompt()

        input_ids = tokenizer(
            prompt_text, truncation=True, return_tensors="pt"
        ).input_ids.to(args.device)
        # Append cognition tokens: [29871, 2] (empty-space + cognition/EOS)
        input_ids = torch.cat(
            (input_ids, torch.tensor([[29871, 2]], dtype=torch.long, device=args.device)),
            dim=1,
        )
        attention_mask = torch.ones_like(input_ids)

        # Image transform (returns dict for DinoSigLIP: {"dino": ..., "siglip": ...})
        pixel_values = image_transform(img)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(args.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(args.device) for k, v in pixel_values.items()}

        feats = extract_all_7(
            vlm, input_ids, attention_mask, pixel_values,
            tokenizer, task, args.device, autocast_dtype,
        )

        for k in keys:
            if k not in feats or feats[k] is None:
                raise RuntimeError(f"Feature {k} missing at sample {i}")
            accum[k].append(feats[k])

        if (i + 1) % 500 == 0:
            torch.cuda.empty_cache()

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            shapes = "  ".join(f"{k}={tuple(feats[k].shape)}" for k in keys)
            print(
                f"  [{i+1}/{num_samples}]  {shapes}  rate={rate:.1f}/s  ETA={eta/60:.1f}min",
                flush=True,
            )

    # --- Save ---
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
        "action_model_type": args.action_model_type,
        "type": "all_7_features",
        "num_samples": num_samples,
        "feature_dims": {k: int(accum[k][0].shape[-1]) for k in keys},
        "file_map": file_map,
    }
    with open(os.path.join(args.output_dir, "extraction_metadata_all7.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone. {elapsed/60:.1f} min total, {num_samples/elapsed:.1f} samples/s")


if __name__ == "__main__":
    main()
