"""
Multi-position feature extraction from LingBot-VA for CKNNA.

Extracts features from 7 positions within the WAN 2.2-5B backbone in a single
forward pass per sample:

  P0: input_embed           - after patch_embedding_mlp, before block 0
  P1: block14_post_selfattn  - after self-attn residual in block 14
  P2: block14_post_crossattn - after cross-attn residual in block 14
  P3: block29_post_selfattn  - after self-attn residual in block 29
  P4: block29_post_crossattn - after cross-attn residual in block 29
  P5: block29               - full block 29 output (post-FFN)
  P6: norm_out              - after final LayerNorm (current primary)

Hook strategy (code-verified against wan_va/modules/model.py):
  - P0: forward hook on transformer.patch_embedding_mlp
  - P1/P3: forward PRE-hook on blocks[N].norm2 (input = post-self-attn residual)
  - P2/P4: forward PRE-hook on blocks[N].norm3 (input = post-cross-attn residual)
  - P5: forward hook on blocks[29] (output = full block output)
  - P6: forward hook on transformer.norm_out (output = post-LN, pre-scale/shift)

Note: LingBot-VA uses cross_attn_norm=True, so norm2 is FP32LayerNorm (not Identity).

Usage:
  python extract_features_multipos.py \\
      --transformer_path /path/to/transformer \\
      --vae_path /path/to/vae \\
      --text_encoder_path /path/to/text_encoder \\
      --tokenizer_path /path/to/tokenizer \\
      --data_dir /path/to/cknna_data_droid \\
      --output_dir /path/to/cknna_data_droid/lingbot-va-base-multipos \\
      --num_cameras 1
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


# --- Position names (order matters for logging) ---
POSITION_NAMES = [
    "input_embed",
    "block14_post_selfattn",
    "block14_post_crossattn",
    "block29_post_selfattn",
    "block29_post_crossattn",
    "block29",
    "norm_out",
]


# --------------------------------------------------------------------------
# VAE encoding (unchanged from extract_features.py)
# --------------------------------------------------------------------------

def encode_single_image(streaming_vae, image_tensor, dtype, vae_device):
    """Encode a single camera image through the causal VAE."""
    streaming_vae.clear_cache()
    enc_out = streaming_vae.encode_chunk(image_tensor.to(vae_device).to(dtype))
    mu, logvar = torch.chunk(enc_out, 2, dim=1)
    latents_mean = torch.tensor(streaming_vae.vae.config.latents_mean).to(mu.device)
    latents_std = 1.0 / torch.tensor(streaming_vae.vae.config.latents_std).to(mu.device)
    latents_mean = latents_mean.view(1, -1, 1, 1, 1)
    latents_std = latents_std.view(1, -1, 1, 1, 1)
    mu_norm = ((mu.float() - latents_mean) * latents_std).to(mu)
    return mu_norm


def encode_multicam(streaming_vae, cam_images_list, dtype, vae_device):
    """Encode 3 cameras using side-by-side concatenation."""
    cam_batch = torch.cat(cam_images_list, dim=0)
    streaming_vae.clear_cache()
    enc_out = streaming_vae.encode_chunk(cam_batch.to(vae_device).to(dtype))
    mu, logvar = torch.chunk(enc_out, 2, dim=1)
    latents_mean = torch.tensor(streaming_vae.vae.config.latents_mean).to(mu.device)
    latents_std = 1.0 / torch.tensor(streaming_vae.vae.config.latents_std).to(mu.device)
    latents_mean = latents_mean.view(1, -1, 1, 1, 1)
    latents_std = latents_std.view(1, -1, 1, 1, 1)
    mu_norm = ((mu.float() - latents_mean) * latents_std).to(mu)
    video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
    return video_latent


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
    parser = argparse.ArgumentParser(description="Multi-position LingBot-VA feature extraction for CKNNA.")
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
                        help="1=exterior_image_1_left only, 3=all cameras spatially concatenated")
    # Options
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples to process (0 = all)")
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=320)
    parser.add_argument("--device", type=str, default="cuda")
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
    import json as _json
    _cfg_path = os.path.join(args.transformer_path, "config.json")
    with open(_cfg_path) as _f:
        _cfg = _json.load(_f)
    if _cfg.get("attn_mode") == "flex":
        print("  Overriding attn_mode: flex -> torch (required for inference)")
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

    latent_h_single = args.image_height // 16
    latent_w_single = args.image_width // 16

    if args.num_cameras == 1:
        latent_h = latent_h_single
        latent_w = latent_w_single
    else:
        latent_h = latent_h_single
        latent_w = latent_w_single * args.num_cameras

    post_patch_h = latent_h // patch_size[1]
    post_patch_w = latent_w // patch_size[2]
    num_tokens = post_patch_h * post_patch_w

    print(f"\n  Cameras: {args.num_cameras}")
    print(f"  Image (per camera): {args.image_height}x{args.image_width}")
    print(f"  Combined latent: {latent_h}x{latent_w}")
    print(f"  Patch size: {patch_size}")
    print(f"  Tokens per frame: {post_patch_h}x{post_patch_w} = {num_tokens}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num layers: {num_layers}")
    print(f"  Extraction positions: {POSITION_NAMES}")

    # --- Precompute grid_id ---
    grid_id = get_mesh_id(1, post_patch_h, post_patch_w, 0, 1, 0).to(device)
    grid_id_batched = grid_id[None]

    # --- Register hooks for all 7 positions ---
    features = {}
    hooks = []

    # P0: input_embed — forward hook on patch_embedding_mlp
    def input_embed_hook(module, input, output):
        features["input_embed"] = output.detach().float().mean(dim=1).cpu()
    hooks.append(transformer.patch_embedding_mlp.register_forward_hook(input_embed_hook))

    # P1: block14_post_selfattn — pre-hook on blocks[14].norm2
    # norm2 input = hidden_states.float() = post-self-attn residual
    def b14_post_selfattn_hook(module, args):
        features["block14_post_selfattn"] = args[0].detach().float().mean(dim=1).cpu()
    hooks.append(transformer.blocks[14].norm2.register_forward_pre_hook(b14_post_selfattn_hook))

    # P2: block14_post_crossattn — pre-hook on blocks[14].norm3
    # norm3 input = hidden_states.float() = post-cross-attn residual
    def b14_post_crossattn_hook(module, args):
        features["block14_post_crossattn"] = args[0].detach().float().mean(dim=1).cpu()
    hooks.append(transformer.blocks[14].norm3.register_forward_pre_hook(b14_post_crossattn_hook))

    # P3: block29_post_selfattn — pre-hook on blocks[29].norm2
    def b29_post_selfattn_hook(module, args):
        features["block29_post_selfattn"] = args[0].detach().float().mean(dim=1).cpu()
    hooks.append(transformer.blocks[29].norm2.register_forward_pre_hook(b29_post_selfattn_hook))

    # P4: block29_post_crossattn — pre-hook on blocks[29].norm3
    def b29_post_crossattn_hook(module, args):
        features["block29_post_crossattn"] = args[0].detach().float().mean(dim=1).cpu()
    hooks.append(transformer.blocks[29].norm3.register_forward_pre_hook(b29_post_crossattn_hook))

    # P5: block29 — forward hook on blocks[29] output
    def block29_hook(module, input, output):
        features["block29"] = output.detach().float().mean(dim=1).cpu()
    hooks.append(transformer.blocks[29].register_forward_hook(block29_hook))

    # P6: norm_out — forward hook on transformer.norm_out
    def norm_out_hook(module, input, output):
        features["norm_out"] = output.detach().float().mean(dim=1).cpu()
    hooks.append(transformer.norm_out.register_forward_hook(norm_out_hook))

    print(f"  Registered {len(hooks)} hooks")

    # --- Per-position accumulator lists ---
    feats_lists = {name: [] for name in POSITION_NAMES}

    # --- Resume support ---
    if args.resume_from > 0:
        partial_path = os.path.join(args.output_dir, "feats_A_norm_out_partial.pt")
        if os.path.exists(partial_path):
            # Load all partial saves
            for name in POSITION_NAMES:
                pp = os.path.join(args.output_dir, f"feats_A_{name}_partial.pt")
                if os.path.exists(pp):
                    feats_lists[name] = list(torch.load(pp, weights_only=True))
            print(f"  Resumed {len(feats_lists['norm_out'])} samples from partial save")

    # --- Main extraction loop ---
    t0 = time.time()
    start_idx = args.resume_from if args.resume_from > 0 else len(feats_lists["norm_out"])

    if args.num_cameras == 3:
        cam_dirs = {
            'cam_high': os.path.join(args.data_dir, "images"),
            'cam_left': os.path.join(args.data_dir, "images_cam2"),
            'cam_right': os.path.join(args.data_dir, "images_cam3"),
        }
        for name, d in cam_dirs.items():
            assert os.path.isdir(d), f"3-camera mode requires {d} ({name})."

    for i in range(start_idx, num_samples):
        if args.num_cameras == 1:
            img_path = os.path.join(images_dir, f"{i:06d}.png")
            img = Image.open(img_path).convert("RGB")
            img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1)
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(args.image_height, args.image_width),
                mode="bilinear", align_corners=False,
            )
            img_tensor = img_tensor / 255.0 * 2.0 - 1.0
            img_tensor = img_tensor.unsqueeze(2)

            with torch.no_grad():
                latent = encode_single_image(streaming_vae, img_tensor, dtype, device)
        else:
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
                img_t = img_t.unsqueeze(2)
                cam_tensors_list.append(img_t)

            with torch.no_grad():
                latent = encode_multicam(streaming_vae, cam_tensors_list, dtype, device)

        # Text encode
        task = task_descriptions[i]
        with torch.no_grad():
            text_emb = encode_text(text_encoder, tokenizer, task, device=device, dtype=dtype)

        # Build input dict
        timesteps = torch.zeros([1], dtype=torch.float32, device=device)
        timesteps_batched = timesteps[None]

        input_dict = {
            "noisy_latents": latent.to(dtype),
            "timesteps": timesteps_batched,
            "grid_id": grid_id_batched,
            "text_emb": text_emb,
        }

        # Forward pass — all hooks fire automatically
        with torch.no_grad():
            transformer(input_dict, update_cache=0, action_mode=False)

        # Collect features from hooks into per-position lists
        for name in POSITION_NAMES:
            feat = features[name]  # [1, hidden_dim] from mean-pool
            feats_lists[name].append(feat.squeeze(0))  # [hidden_dim]

        # Progress
        if (i + 1) % 50 == 0 or i == start_idx:
            elapsed = time.time() - t0
            processed = i - start_idx + 1
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = num_samples - i - 1
            eta = remaining / rate if rate > 0 else 0
            print(f"  [{i+1}/{num_samples}]  rate={rate:.2f}/s  ETA={eta/60:.1f}min")

        # Periodic save
        if args.save_every > 0 and (i + 1) % args.save_every == 0:
            for name in POSITION_NAMES:
                torch.save(
                    torch.stack(feats_lists[name]),
                    os.path.join(args.output_dir, f"feats_A_{name}_partial.pt"),
                )
            print(f"    (partial save at {i+1})")

    # --- Remove hooks ---
    for h in hooks:
        h.remove()

    # --- Save final results ---
    print("\nSaving final results...")
    N = len(feats_lists["norm_out"])

    for name in POSITION_NAMES:
        feats = torch.stack(feats_lists[name])  # [N, hidden_dim]
        torch.save(feats, os.path.join(args.output_dir, f"feats_A_{name}.pt"))
        print(f"  feats_A_{name}.pt: {tuple(feats.shape)}")

    # Backward compatibility: feats_A.pt = feats_A_norm_out.pt
    feats_norm_out = torch.stack(feats_lists["norm_out"])
    torch.save(feats_norm_out, os.path.join(args.output_dir, "feats_A.pt"))
    print(f"  feats_A.pt: {tuple(feats_norm_out.shape)} (= norm_out, backward compat)")

    # Clean up partial saves
    for name in POSITION_NAMES:
        p = os.path.join(args.output_dir, f"feats_A_{name}_partial.pt")
        if os.path.exists(p):
            os.remove(p)

    # --- Save metadata ---
    extraction_meta = {
        "model": "lingbot-va",
        "transformer_path": args.transformer_path,
        "extraction_mode": "multi-position (7 positions in single forward pass)",
        "positions": {
            "input_embed": "After patch_embedding_mlp, before block 0. Pure image, no text.",
            "block14_post_selfattn": "After self-attn residual in block 14. Pre-hook on blocks[14].norm2.",
            "block14_post_crossattn": "After cross-attn residual in block 14. Pre-hook on blocks[14].norm3.",
            "block29_post_selfattn": "After self-attn residual in block 29. Pre-hook on blocks[29].norm2.",
            "block29_post_crossattn": "After cross-attn residual in block 29. Pre-hook on blocks[29].norm3.",
            "block29": "Full block 29 output (post-FFN). Forward hook on blocks[29].",
            "norm_out": "After final FP32LayerNorm (pre scale/shift). Forward hook on norm_out.",
        },
        "feature_semantics": "img+text inseparable (text via cross-attention). input_embed is image-only.",
        "hidden_size": hidden_dim,
        "num_tokens_per_frame": num_tokens,
        "pooling": "mean over spatial tokens",
        "num_samples": N,
        "num_cameras": args.num_cameras,
        "image_size_per_camera": [args.image_height, args.image_width],
        "latent_size": [latent_h, latent_w],
        "patch_size": patch_size,
        "num_layers": num_layers,
        "timestep": 0.0,
        "outputs": [f"feats_A_{name}.pt" for name in POSITION_NAMES] + ["feats_A.pt"],
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== LingBot-VA Multi-Position Extraction Complete ===")
    print(f"  N={N} samples, {elapsed/60:.1f} min ({elapsed/N:.2f} s/sample)")
    print(f"  Positions: {len(POSITION_NAMES)}")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
