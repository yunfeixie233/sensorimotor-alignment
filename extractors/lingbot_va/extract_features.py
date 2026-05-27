"""
Phase 2: Extract features from LingBot-VA (diffusion world-action model) for CKNNA.

Architecture: Wan2.2-5B video backbone (d=3072, 30 transformer blocks) + 350M action stream (MoT).
              Text conditioning via cross-attention (T5 encoder → PixArt MLP → cross-attn K,V).
              NOT a VLM — a diffusion transformer.

Extraction approach:
  - Single-frame forward pass at t=0 (no noise, no KV cache, no denoising, no CFG)
  - Hook transformer.norm_out → mean-pool over spatial tokens → [1, 3072]
  - Optionally hook all 30 block outputs for layer-wise CKNNA

Feature semantics:
  Text enters ONLY via cross-attention (attn2: Q=visual, K=V=text). There are NO text tokens
  in the self-attention sequence or in norm_out output. Text is deeply embedded INTO visual
  representations via 30× cross-attention layers. Therefore:
    - feats_A = "img+text" (inseparable) — NOT image-only
    - There is NO separable image-only or text-only component
    - This differs from VLAs where image and text tokens are separate in the sequence

Input handling:
  - 1 camera: DROID exterior_image_1_left (180×320) resized to 256×320
    VAE latent [1,48,1,16,20] → patch[1,2,2] → 8×10 = 80 tokens
  - 3 cameras: all at same resolution (side-by-side, matching Franka/DROID path)
    exterior_image_1_left + exterior_image_2_left + wrist_image_left
    Each 256×320 → VAE → 16×20, concat width → [1,48,1,16,60]
    After patch[1,2,2] → 8×30 = 240 tokens

Output:
  <output_dir>/feats_A.pt          -- (N, 3072) float32, norm_out mean-pooled (img+text)
  <output_dir>/feats_A_layers.pt   -- (N, 31, 3072) float32, all 30 blocks + norm_out
  <output_dir>/extraction_metadata.json

Usage:
  python extract_features.py \
      --transformer_path /path/to/transformer \
      --vae_path /path/to/vae \
      --text_encoder_path /path/to/text_encoder \
      --tokenizer_path /path/to/tokenizer \
      --data_dir /path/to/cknna_data_droid \
      --output_dir /path/to/cknna_data_droid/lingbot-va-base \
      --num_cameras 1 \
      --max_samples 200
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
    load_transformer,
    load_vae,
)
from wan_va.utils import get_mesh_id


# --------------------------------------------------------------------------
# VAE encoding
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

    # Normalize using VAE config stats (same as _encode_obs)
    # Official uses normalize_latents(mu, mean, 1.0/std), so we invert std here
    latents_mean = torch.tensor(streaming_vae.vae.config.latents_mean).to(mu.device)
    latents_std = 1.0 / torch.tensor(streaming_vae.vae.config.latents_std).to(mu.device)
    latents_mean = latents_mean.view(1, -1, 1, 1, 1)
    latents_std = latents_std.view(1, -1, 1, 1, 1)
    mu_norm = ((mu.float() - latents_mean) * latents_std).to(mu)
    return mu_norm


def encode_multicam(streaming_vae, cam_images_list, dtype, vae_device):
    """Encode 3 cameras using side-by-side concatenation (matches Franka/DROID path).

    For DROID/Franka (env_type='none'), wan_va_server._encode_obs uses:
      1. All cameras at SAME resolution
      2. VAE encode as a batch: torch.cat(videos, dim=0)
      3. Concat latents along width: torch.cat(mu_norm.split(1, dim=0), dim=-1)

    Layout (side-by-side):
        ┌──────────┬──────────┬──────────┐
        │  cam_1   │  cam_2   │  cam_3   │  each H×W → VAE → h×w
        │  (h×w)   │  (h×w)   │  (h×w)   │  concatenated along width → h×(3w)
        └──────────┴──────────┴──────────┘

    With H=256, W=320: each VAE → 16×20, concat → 16×60, patch[1,2,2] → 8×30 = 240 tokens

    Args:
        cam_images_list: list of [1, 3, 1, H, W] float tensors in [-1, 1], all same size
    Returns:
        normalized latent: [1, 48, 1, h, w*num_cams]
    """
    # Batch all cameras together: [N_cams, 3, 1, H, W]
    cam_batch = torch.cat(cam_images_list, dim=0)

    streaming_vae.clear_cache()
    enc_out = streaming_vae.encode_chunk(cam_batch.to(vae_device).to(dtype))

    mu, logvar = torch.chunk(enc_out, 2, dim=1)

    # Normalize each camera's latent
    # Official uses normalize_latents(mu, mean, 1.0/std), so we invert std here
    latents_mean = torch.tensor(streaming_vae.vae.config.latents_mean).to(mu.device)
    latents_std = 1.0 / torch.tensor(streaming_vae.vae.config.latents_std).to(mu.device)
    latents_mean = latents_mean.view(1, -1, 1, 1, 1)
    latents_std = latents_std.view(1, -1, 1, 1, 1)
    mu_norm = ((mu.float() - latents_mean) * latents_std).to(mu)

    # Concatenate along width (last dim), matching server.py:373
    # mu_norm: [N_cams, 48, 1, h, w] → split → list of [1, 48, 1, h, w] → cat along dim=-1
    video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)  # [1, 48, 1, h, w*N_cams]
    return video_latent


# --------------------------------------------------------------------------
# Text encoding (same as server._get_t5_prompt_embeds)
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

    # Truncate to actual length, then pad back
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
    parser = argparse.ArgumentParser(description="Phase 2: Extract LingBot-VA features for CKNNA.")
    # Model paths
    parser.add_argument("--transformer_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--text_encoder_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="DROID data dir with images/ and metadata.json")
    parser.add_argument("--output_dir", type=str, required=True)
    # Camera mode
    parser.add_argument("--num_cameras", type=int, default=1, choices=[1, 3],
                        help="1=exterior_image_1_left only (80 tokens), "
                             "3=all cameras spatially concatenated (120 tokens)")
    # Options
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples to process (0 = all)")
    parser.add_argument("--image_height", type=int, default=256,
                        help="Resize height for cam_high (must be divisible by 32)")
    parser.add_argument("--image_width", type=int, default=320,
                        help="Resize width for cam_high (must be divisible by 32)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_layers", action="store_true",
                        help="Save per-layer features (30 blocks + norm_out)")
    parser.add_argument("--save_every", type=int, default=200,
                        help="Save partial results every N samples")
    parser.add_argument("--resume_from", type=int, default=0,
                        help="Resume from sample index")
    args = parser.parse_args()

    assert args.image_height % 32 == 0, f"height must be divisible by 32, got {args.image_height}"
    assert args.image_width % 32 == 0, f"width must be divisible by 32, got {args.image_width}"

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

    # --- Load models ---
    device = args.device
    dtype = torch.bfloat16

    print(f"Loading transformer: {args.transformer_path}")
    # Must override attn_mode from "flex" (training) to "torch" (inference)
    # See: Q5 notes — "Must change attn_mode in config.json: flex for training, torch for inference"
    import json as _json
    _cfg_path = os.path.join(args.transformer_path, "config.json")
    with open(_cfg_path) as _f:
        _cfg = _json.load(_f)
    if _cfg.get("attn_mode") == "flex":
        print("  Overriding attn_mode: flex → torch (required for inference)")
        _cfg["attn_mode"] = "torch"
        with open(_cfg_path, "w") as _f:
            _json.dump(_cfg, _f, indent=2)
    transformer = load_transformer(args.transformer_path, dtype, device)
    transformer.eval().requires_grad_(False)
    print(f"  params: {sum(p.numel() for p in transformer.parameters())/1e9:.2f}B")

    print(f"Loading VAE: {args.vae_path}")
    vae = load_vae(args.vae_path, dtype, device)
    streaming_vae = WanVAEStreamingWrapper(vae)

    print(f"Loading text encoder: {args.text_encoder_path}")
    text_encoder = load_text_encoder(args.text_encoder_path, dtype, device)
    text_encoder.eval().requires_grad_(False)

    print(f"Loading tokenizer: {args.tokenizer_path}")
    tokenizer = load_tokenizer(args.tokenizer_path)

    # --- Compute spatial dimensions ---
    patch_size = list(transformer.config.patch_size)
    hidden_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    num_layers = transformer.config.num_layers

    # VAE 16x spatial compression
    latent_h_single = args.image_height // 16
    latent_w_single = args.image_width // 16

    if args.num_cameras == 1:
        latent_h = latent_h_single
        latent_w = latent_w_single
    else:
        # 3 cameras: side-by-side concatenation along width (Franka/DROID path)
        # All cameras at same resolution, concat latent widths
        latent_h = latent_h_single
        latent_w = latent_w_single * args.num_cameras  # e.g. 20*3 = 60

    post_patch_h = latent_h // patch_size[1]
    post_patch_w = latent_w // patch_size[2]
    num_tokens = post_patch_h * post_patch_w

    print(f"\n  Cameras: {args.num_cameras}")
    print(f"  Image (per camera): {args.image_height}×{args.image_width}")
    print(f"  Combined latent: {latent_h}×{latent_w}")
    print(f"  Patch size: {patch_size}")
    print(f"  Tokens per frame: {post_patch_h}×{post_patch_w} = {num_tokens}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num layers: {num_layers}")

    # --- Precompute grid_id (same for all samples — single frame at position 0) ---
    grid_id = get_mesh_id(
        1,             # f: 1 temporal token
        post_patch_h,  # h: spatial height after patching
        post_patch_w,  # w: spatial width after patching
        0,             # t: type indicator (0 = latent)
        1,             # f_w: frame weight
        0,             # f_shift: frame_st_id = 0
    ).to(device)
    grid_id_batched = grid_id[None]  # [1, 4, num_tokens]

    # --- Register hooks ---
    features = {}

    def make_block_hook(layer_idx):
        def hook_fn(module, input, output):
            features[f"block_{layer_idx}"] = output.detach().float().mean(dim=1).cpu()
        return hook_fn

    def norm_out_hook(module, input, output):
        features["norm_out_raw"] = output.detach().float().clone()

    hooks = []
    if args.save_layers:
        for i, block in enumerate(transformer.blocks):
            hooks.append(block.register_forward_hook(make_block_hook(i)))
    hooks.append(transformer.norm_out.register_forward_hook(norm_out_hook))

    # --- Resume support ---
    feats_list = []
    layer_feats_list = []

    if args.resume_from > 0:
        partial_path = os.path.join(args.output_dir, "feats_A_partial.pt")
        if os.path.exists(partial_path):
            feats_list = list(torch.load(partial_path, weights_only=True))
            print(f"  Resumed {len(feats_list)} samples from partial save")
        if args.save_layers:
            partial_layers = os.path.join(args.output_dir, "feats_A_layers_partial.pt")
            if os.path.exists(partial_layers):
                layer_feats_list = list(torch.load(partial_layers, weights_only=True))

    # --- Main extraction loop ---
    t0 = time.time()
    start_idx = args.resume_from if args.resume_from > 0 else len(feats_list)

    # --- Determine image directories ---
    if args.num_cameras == 3:
        cam_dirs = {
            'cam_high': os.path.join(args.data_dir, "images"),  # exterior_image_1_left
            'cam_left': os.path.join(args.data_dir, "images_cam2"),  # exterior_image_2_left
            'cam_right': os.path.join(args.data_dir, "images_cam3"),  # wrist_image_left
        }
        for name, d in cam_dirs.items():
            assert os.path.isdir(d), f"3-camera mode requires {d} ({name}). Run extract_droid_3cam.py first."

    for i in range(start_idx, num_samples):
        if args.num_cameras == 1:
            # a) Load and preprocess single image
            img_path = os.path.join(images_dir, f"{i:06d}.png")
            img = Image.open(img_path).convert("RGB")
            img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1)  # [3, H, W]
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(args.image_height, args.image_width),
                mode="bilinear",
                align_corners=False,
            )  # [1, 3, H, W]
            img_tensor = img_tensor / 255.0 * 2.0 - 1.0
            img_tensor = img_tensor.unsqueeze(2)  # [1, 3, 1, H, W]

            with torch.no_grad():
                latent = encode_single_image(streaming_vae, img_tensor, dtype, device)
        else:
            # a) Load and preprocess 3 camera images (all at same resolution)
            cam_tensors_list = []
            for cam_dir in [cam_dirs['cam_high'], cam_dirs['cam_left'], cam_dirs['cam_right']]:
                img_path = os.path.join(cam_dir, f"{i:06d}.png")
                img = Image.open(img_path).convert("RGB")
                img_t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1)
                img_t = F.interpolate(
                    img_t.unsqueeze(0),
                    size=(args.image_height, args.image_width),
                    mode="bilinear", align_corners=False,
                )
                img_t = img_t / 255.0 * 2.0 - 1.0
                img_t = img_t.unsqueeze(2)  # [1, 3, 1, H, W]
                cam_tensors_list.append(img_t)

            with torch.no_grad():
                latent = encode_multicam(
                    streaming_vae, cam_tensors_list, dtype, device,
                )

        # c) Text encode
        task = task_descriptions[i]
        with torch.no_grad():
            text_emb = encode_text(text_encoder, tokenizer, task, device=device, dtype=dtype)
            # text_emb: [1, 512, 4096]

        # d) Build input dict
        timesteps = torch.zeros([1], dtype=torch.float32, device=device)  # t=0
        timesteps_batched = timesteps[None]  # [1, 1]

        input_dict = {
            "noisy_latents": latent.to(dtype),
            "timesteps": timesteps_batched,
            "grid_id": grid_id_batched,
            "text_emb": text_emb,
        }

        # e) Forward pass (no cache, no CFG)
        with torch.no_grad():
            transformer(input_dict, update_cache=0, action_mode=False)

        # f) Extract features from hook
        norm_out_feat = features["norm_out_raw"]  # [1, num_tokens, 3072]
        feat_pooled = norm_out_feat.mean(dim=1).cpu()  # [1, 3072]
        feats_list.append(feat_pooled.squeeze(0))

        if args.save_layers:
            layer_feats = []
            for li in range(num_layers):
                layer_feats.append(features[f"block_{li}"].squeeze(0))  # [3072]
            # Add norm_out pooled as last layer
            layer_feats.append(feat_pooled.squeeze(0))
            layer_feats_list.append(torch.stack(layer_feats))  # [31, 3072]

        # g) Progress & periodic save
        if (i + 1) % 50 == 0 or i == start_idx:
            elapsed = time.time() - t0
            processed = i - start_idx + 1
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = num_samples - i - 1
            eta = remaining / rate if rate > 0 else 0
            print(f"  [{i+1}/{num_samples}]  rate={rate:.2f}/s  ETA={eta/60:.1f}min"
                  f"  feat={tuple(feat_pooled.shape)}")

        if args.save_every > 0 and (i + 1) % args.save_every == 0:
            torch.save(torch.stack(feats_list),
                       os.path.join(args.output_dir, "feats_A_partial.pt"))
            if args.save_layers and layer_feats_list:
                torch.save(torch.stack(layer_feats_list),
                           os.path.join(args.output_dir, "feats_A_layers_partial.pt"))
            print(f"    (partial save at {i+1})")

    # --- Remove hooks ---
    for h in hooks:
        h.remove()

    # --- Save final results ---
    print("\nSaving final results...")
    N = len(feats_list)

    feats_A = torch.stack(feats_list)  # [N, 3072]
    torch.save(feats_A, os.path.join(args.output_dir, "feats_A.pt"))
    print(f"  feats_A.pt: {tuple(feats_A.shape)} (img+text — inseparable in diffusion model)")

    if args.save_layers and layer_feats_list:
        feats_layers = torch.stack(layer_feats_list)  # [N, 31, 3072]
        torch.save(feats_layers, os.path.join(args.output_dir, "feats_A_layers.pt"))
        print(f"  feats_A_layers.pt: {tuple(feats_layers.shape)}")

    # Clean up partial saves
    for partial in ["feats_A_partial.pt", "feats_A_layers_partial.pt"]:
        p = os.path.join(args.output_dir, partial)
        if os.path.exists(p):
            os.remove(p)

    # --- Save metadata ---
    extraction_meta = {
        "model": "lingbot-va",
        "transformer_path": args.transformer_path,
        "extraction_point": "norm_out (pre scale/shift, post LayerNorm)",
        "feature_semantics": "img+text (inseparable — text enters via cross-attention only)",
        "hidden_size": hidden_dim,
        "num_tokens_per_frame": num_tokens,
        "pooling": "mean over spatial tokens",
        "num_samples": N,
        "num_cameras": args.num_cameras,
        "image_size_per_camera": [args.image_height, args.image_width],
        "multicam_layout": ("side-by-side (Franka/DROID path, all same resolution)"
                            if args.num_cameras == 3 else "single"),
        "latent_size": [latent_h, latent_w],
        "patch_size": patch_size,
        "post_patch_size": [post_patch_h, post_patch_w],
        "num_layers": num_layers,
        "timestep": 0.0,
        "kv_cache": "none",
        "cfg": "none (conditional only)",
        "outputs": ["feats_A.pt"]
            + (["feats_A_layers.pt"] if args.save_layers else []),
        "notes": {
            "feats_A": "norm_out mean-pooled (img+text inseparable). Text enters via "
                       "30x cross-attention; no text tokens in self-attention or output.",
            "feats_A_layers": "30 block outputs + norm_out, each mean-pooled to 3072-dim",
            "no_feats_A_img": "LingBot-VA has NO separable image-only feature. "
                              "feats_A IS the combined img+text representation.",
        },
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== LingBot-VA Phase 2 Complete ===")
    print(f"  N={N} samples, {elapsed/60:.1f} min ({elapsed/N:.2f} s/sample)")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
