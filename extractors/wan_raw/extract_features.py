"""
Phase 2: Extract features from raw WAN 2.2-TI2V-5B (video generation model, pre-VA-tuning).

Architecture: WAN 2.2 diffusion transformer (d=3072, 30 blocks) — the SAME backbone that
              LingBot-VA is initialized from, but WITHOUT any robot action training.
              Text conditioning via cross-attention (UMT5 → PixArt MLP → cross-attn K,V).

This script uses the **diffusers-native** WanTransformer3DModel (not LingBot-VA's modified class)
to avoid config mismatches from action-specific layers (action_embedder, condition_embedder_action).

Extraction approach:
  - Single-frame forward pass at t=0 (no noise, no denoising)
  - Hook transformer.norm_out → mean-pool over spatial tokens → [1, 3072]
  - Feature semantics: img+text inseparable (text enters via cross-attention only)

Output:
  <output_dir>/feats_A.pt                 -- (N, 3072) float32, norm_out mean-pooled
  <output_dir>/extraction_metadata.json

Usage:
  python extract_features.py \
      --transformer_path /path/to/Wan2.2-TI2V-5B-Diffusers/transformer \
      --vae_path /path/to/vae \
      --text_encoder_path /path/to/text_encoder \
      --tokenizer_path /path/to/tokenizer \
      --data_dir /path/to/cknna_data_droid \
      --output_dir /path/to/cknna_data_droid/wan-raw-5b \
      --num_cameras 1 --device cuda
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# --- VAE / T5 loading from lingbot-va repo (shared components) ---
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LINGBOT_VA_ROOT = os.environ.get(
    "LINGBOT_VA_ROOT",
    os.path.join(_PROJECT_ROOT, "repos", "lingbot-va"),
)
LINGBOT_VA_ROOT = os.path.abspath(LINGBOT_VA_ROOT)
if LINGBOT_VA_ROOT not in sys.path:
    sys.path.insert(0, LINGBOT_VA_ROOT)

from wan_va.modules.utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_vae,
)

# --- WAN transformer from diffusers (native, no action layers) ---
from diffusers import WanTransformer3DModel


# --------------------------------------------------------------------------
# VAE encoding (identical to lingbot_va extractor)
# --------------------------------------------------------------------------

def encode_single_image(streaming_vae, image_tensor, dtype, vae_device):
    """Encode a single camera image through the causal VAE.

    Args:
        image_tensor: [1, 3, 1, H, W] float tensor in [-1, 1]
    Returns:
        normalized latent: [1, 48, 1, H//16, W//16]
    """
    streaming_vae.clear_cache()
    enc_out = streaming_vae.encode_chunk(image_tensor.to(vae_device).to(dtype))
    mu, logvar = torch.chunk(enc_out, 2, dim=1)

    latents_mean = torch.tensor(streaming_vae.vae.config.latents_mean).to(mu.device)
    latents_std = 1.0 / torch.tensor(streaming_vae.vae.config.latents_std).to(mu.device)
    latents_mean = latents_mean.view(1, -1, 1, 1, 1)
    latents_std = latents_std.view(1, -1, 1, 1, 1)
    mu_norm = ((mu.float() - latents_mean) * latents_std).to(mu)
    return mu_norm


# --------------------------------------------------------------------------
# Text encoding (same as lingbot_va extractor)
# --------------------------------------------------------------------------

def encode_text(text_encoder, tokenizer, prompt, max_seq_len=512, device="cuda", dtype=torch.bfloat16):
    """Encode a text prompt using T5 encoder."""
    prompt = [prompt] if isinstance(prompt, str) else prompt
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_seq_len,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    mask = text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()

    encoder_device = next(text_encoder.parameters()).device
    prompt_embeds = text_encoder(
        text_input_ids.to(encoder_device), mask.to(encoder_device)
    ).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack([
        torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
        for u in prompt_embeds
    ], dim=0)
    return prompt_embeds


# --------------------------------------------------------------------------
# Main extraction
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract raw WAN 2.2-5B features for CKNNA.")
    parser.add_argument("--transformer_path", type=str, required=True,
                        help="Path to Wan2.2-TI2V-5B-Diffusers/transformer")
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--text_encoder_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_cameras", type=int, default=1, choices=[1])
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=320)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_every", type=int, default=200)
    args = parser.parse_args()

    assert args.image_height % 32 == 0
    assert args.image_width % 32 == 0
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load metadata ---
    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    if args.max_samples > 0:
        num_samples = min(num_samples, args.max_samples)
    print(f"Samples to process: {num_samples}")

    device = args.device
    dtype = torch.bfloat16

    # --- Load diffusers-native WAN transformer ---
    print(f"Loading WAN transformer (diffusers): {args.transformer_path}")
    transformer = WanTransformer3DModel.from_pretrained(
        args.transformer_path,
        torch_dtype=dtype,
    ).to(device)
    transformer.eval().requires_grad_(False)
    print(f"  params: {sum(p.numel() for p in transformer.parameters())/1e9:.2f}B")

    # Read config
    hidden_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    num_layers = transformer.config.num_layers
    patch_size = list(transformer.config.patch_size)

    print(f"Loading VAE: {args.vae_path}")
    vae = load_vae(args.vae_path, dtype, device)
    streaming_vae = WanVAEStreamingWrapper(vae)

    print(f"Loading text encoder: {args.text_encoder_path}")
    text_encoder = load_text_encoder(args.text_encoder_path, dtype, device)
    text_encoder.eval().requires_grad_(False)

    print(f"Loading tokenizer: {args.tokenizer_path}")
    tokenizer = load_tokenizer(args.tokenizer_path)

    # --- Compute spatial dimensions ---
    latent_h = args.image_height // 16
    latent_w = args.image_width // 16
    post_patch_h = latent_h // patch_size[1]
    post_patch_w = latent_w // patch_size[2]
    num_tokens = post_patch_h * post_patch_w

    print(f"\n  Image: {args.image_height}x{args.image_width}")
    print(f"  Latent: {latent_h}x{latent_w}")
    print(f"  Patch size: {patch_size}")
    print(f"  Tokens per frame: {post_patch_h}x{post_patch_w} = {num_tokens}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num layers: {num_layers}")

    # --- Register norm_out hook ---
    features = {}

    def norm_out_hook(module, input, output):
        features["norm_out_raw"] = output.detach().float().clone()

    hook_handle = transformer.norm_out.register_forward_hook(norm_out_hook)

    # --- Main extraction loop ---
    feats_list = []
    t0 = time.time()

    for i in range(num_samples):
        # a) Load and preprocess image
        img_path = os.path.join(images_dir, f"{i:06d}.png")
        img = Image.open(img_path).convert("RGB")
        img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1)
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(args.image_height, args.image_width),
            mode="bilinear", align_corners=False,
        )
        img_tensor = img_tensor / 255.0 * 2.0 - 1.0
        img_tensor = img_tensor.unsqueeze(2)  # [1, 3, 1, H, W]

        with torch.no_grad():
            latent = encode_single_image(streaming_vae, img_tensor, dtype, device)
            # latent: [1, 48, 1, 16, 20] for 256x320 single-cam

        # b) Text encode
        task = task_descriptions[i]
        with torch.no_grad():
            text_emb = encode_text(text_encoder, tokenizer, task, device=device, dtype=dtype)
            # text_emb: [1, 512, 4096]

        # c) Forward pass through diffusers WAN
        # diffusers expects: hidden_states=[B,C,F,H,W], timestep=[B], encoder_hidden_states=[B,L,D]
        timestep = torch.zeros([1], dtype=torch.long, device=device)

        with torch.no_grad():
            transformer(
                hidden_states=latent.to(dtype),
                timestep=timestep,
                encoder_hidden_states=text_emb,
            )

        # d) Extract norm_out features (hook fires BEFORE scale/shift and proj_out)
        norm_out_feat = features["norm_out_raw"]  # [1, num_tokens, hidden_dim]
        feat_pooled = norm_out_feat.mean(dim=1).cpu()  # [1, hidden_dim]
        feats_list.append(feat_pooled.squeeze(0))

        # e) Progress
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            processed = i + 1
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = num_samples - processed
            eta = remaining / rate if rate > 0 else 0
            print(f"  [{processed}/{num_samples}]  rate={rate:.2f}/s  ETA={eta/60:.1f}min"
                  f"  feat={tuple(feat_pooled.shape)}")

        if args.save_every > 0 and (i + 1) % args.save_every == 0:
            torch.save(torch.stack(feats_list),
                       os.path.join(args.output_dir, "feats_A_partial.pt"))
            print(f"    (partial save at {i+1})")

    # --- Cleanup ---
    hook_handle.remove()

    # --- Save ---
    print("\nSaving final results...")
    feats_A = torch.stack(feats_list)
    torch.save(feats_A, os.path.join(args.output_dir, "feats_A.pt"))
    print(f"  feats_A.pt: {tuple(feats_A.shape)}")

    # Remove partial save
    partial = os.path.join(args.output_dir, "feats_A_partial.pt")
    if os.path.exists(partial):
        os.remove(partial)

    # Save metadata
    extraction_meta = {
        "model": "wan-raw-5b",
        "model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "transformer_path": args.transformer_path,
        "extraction_point": "norm_out (pre scale/shift, post LayerNorm)",
        "feature_semantics": "img+text (inseparable — text enters via cross-attention only)",
        "hidden_size": hidden_dim,
        "num_tokens_per_frame": num_tokens,
        "pooling": "mean over spatial tokens",
        "num_samples": len(feats_list),
        "num_cameras": args.num_cameras,
        "image_size": [args.image_height, args.image_width],
        "latent_size": [latent_h, latent_w],
        "patch_size": patch_size,
        "num_layers": num_layers,
        "timestep": 0,
        "notes": "Raw WAN 2.2-TI2V-5B — video generation model BEFORE any VA/robot tuning. "
                 "Same backbone as LingBot-VA but without action pretraining.",
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== WAN Raw Phase 2 Complete ===")
    print(f"  N={len(feats_list)} samples, {elapsed/60:.1f} min ({elapsed/len(feats_list):.2f} s/sample)")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
