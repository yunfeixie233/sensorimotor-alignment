"""
Verify two things about the extraction pipeline:

1. VAE cache: clear_cache() per frame is correct (cache accumulation CRASHES)
2. Transformer: update_cache=0 (all-at-once) == update_cache=2 (incremental KV cache)
"""
import os
import sys
import json
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LINGBOT_VA_ROOT = os.path.join(_PROJECT_ROOT, "repos", "lingbot-va")
sys.path.insert(0, LINGBOT_VA_ROOT)

from wan_va.modules.utils import (
    WanVAEStreamingWrapper, load_text_encoder, load_tokenizer, load_transformer, load_vae,
)
from wan_va.utils import get_mesh_id

CKPT_ROOT = os.path.join(_PROJECT_ROOT, "checkpoints", "lingbot-va")
DATA_DIR = "/lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid_multiframe"
device = "cuda"
dtype = torch.bfloat16


def load_test_frames(sample_idx=0, num_frames=4, cam_dir="images"):
    frames = []
    m_max = 64
    start = m_max - num_frames + 1
    for offset in range(start, m_max + 1):
        path = os.path.join(DATA_DIR, cam_dir, f"{sample_idx:06d}", f"{offset:03d}.png")
        img = Image.open(path).convert("RGB")
        t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1)
        t = F.interpolate(t.unsqueeze(0), size=(256, 320), mode="bilinear", align_corners=False)
        t = t / 255.0 * 2.0 - 1.0
        t = t.unsqueeze(2)
        frames.append(t)
    return frames


def test_vae_cache_crashes():
    """Prove that NOT clearing VAE cache between T=1 frames crashes."""
    print("=" * 70)
    print("TEST 1: VAE cache accumulation crashes with T=1 frames")
    print("=" * 70)

    vae = load_vae(os.path.join(CKPT_ROOT, "lingbot-va-posttrain-robotwin", "vae"), dtype, device)
    sv = WanVAEStreamingWrapper(vae)

    # Show the architecture constraint
    for name, m in sv.encoder.named_modules():
        if hasattr(m, 'time_conv') and hasattr(m.time_conv, 'kernel_size'):
            tc = m.time_conv
            print(f"  {name}.time_conv: kernel={tc.kernel_size} padding={tc.padding}")

    frames = load_test_frames(sample_idx=0, num_frames=2)
    print(f"\n  Loaded {len(frames)} frames, each {frames[0].shape}")

    # Frame 0 with clear cache: should work
    sv.clear_cache()
    enc0 = sv.encode_chunk(frames[0].to(device).to(dtype))
    print(f"  Frame 0 (clear cache): OK, output {enc0.shape}")

    # Frame 1 WITHOUT clearing cache: should CRASH
    print(f"  Frame 1 (no clear cache): ", end="")
    try:
        enc1 = sv.encode_chunk(frames[1].to(device).to(dtype))
        print(f"OK, output {enc1.shape}")
        print("\n  UNEXPECTED: No crash. VAE cache accumulation works.")
        return False  # no crash = our assumption was wrong
    except RuntimeError as e:
        print(f"CRASHED as expected")
        print(f"    Error: {e}")
        print(f"\n  ✓ VERIFIED: VAE cache cannot accumulate across T=1 frames.")
        print(f"    Kernel (3,1,1) > padded input temporal dim (2). clear_cache() per frame is CORRECT.")
        return True  # crash confirms the constraint


def test_transformer_equivalence():
    """Prove update_cache=0 (all-at-once) ≈ update_cache=2 (incremental)."""
    print("\n" + "=" * 70)
    print("TEST 2: Transformer update_cache=0 vs update_cache=2")
    print("=" * 70)

    transformer = load_transformer(
        os.path.join(CKPT_ROOT, "lingbot-va-base", "transformer"), dtype, device)
    transformer.eval().requires_grad_(False)

    vae = load_vae(os.path.join(CKPT_ROOT, "lingbot-va-posttrain-robotwin", "vae"), dtype, device)
    sv = WanVAEStreamingWrapper(vae)

    text_encoder = load_text_encoder(
        os.path.join(CKPT_ROOT, "lingbot-va-posttrain-robotwin", "text_encoder"), dtype, device)
    text_encoder.eval().requires_grad_(False)
    tokenizer = load_tokenizer(
        os.path.join(CKPT_ROOT, "lingbot-va-posttrain-robotwin", "tokenizer"))

    with open(os.path.join(DATA_DIR, "metadata.json")) as f:
        metadata = json.load(f)

    patch_size = list(transformer.config.patch_size)
    T = 3
    post_patch_h = 256 // 16 // patch_size[1]
    post_patch_w = (320 * 3) // 16 // patch_size[2]
    tokens_per_frame = post_patch_h * post_patch_w

    # Load and VAE-encode T frames (3 cameras each)
    latents_mean = torch.tensor(vae.config.latents_mean).to(device).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).to(device).view(1, -1, 1, 1, 1)

    frames_per_cam = {}
    for cam_dir in ["images", "images_cam2", "images_cam3"]:
        frames_per_cam[cam_dir] = load_test_frames(sample_idx=0, num_frames=T, cam_dir=cam_dir)

    latent_list = []
    for t in range(T):
        cam_batch = torch.cat([
            frames_per_cam["images"][t],
            frames_per_cam["images_cam2"][t],
            frames_per_cam["images_cam3"][t],
        ], dim=0)
        sv.clear_cache()
        enc = sv.encode_chunk(cam_batch.to(device).to(dtype))
        mu, _ = torch.chunk(enc, 2, dim=1)
        mu_norm = ((mu.float() - latents_mean) * latents_std).to(mu)
        mu_norm = torch.cat(mu_norm.split(1, dim=0), dim=-1)
        latent_list.append(mu_norm)

    latent_all = torch.cat(latent_list, dim=2)  # [1, 48, T, h, w*3]

    # Encode text
    sys.path.insert(0, _PROJECT_ROOT)
    from extractors.lingbot_va.extract_features_multiframe import encode_text
    text_emb = encode_text(text_encoder, tokenizer, metadata["task_descriptions"][0],
                           device=device, dtype=dtype)

    # --- Method A: update_cache=0, all frames at once ---
    features = {}
    def hook_fn(module, input, output):
        features["out"] = output.detach().float().clone()

    h = transformer.norm_out.register_forward_hook(hook_fn)

    grid_id_all = get_mesh_id(T, post_patch_h, post_patch_w, 0, 1, 0).to(device)
    with torch.no_grad():
        transformer({
            "noisy_latents": latent_all.to(dtype),
            "timesteps": torch.zeros([1, T], dtype=torch.float32, device=device),
            "grid_id": grid_id_all[None],
            "text_emb": text_emb,
        }, update_cache=0, action_mode=False)
    out_all = features["out"].clone()
    h.remove()

    # --- Method B: update_cache=2, incremental frame-by-frame ---
    h = transformer.norm_out.register_forward_hook(hook_fn)

    # Free VAE + text encoder to make room for KV cache
    del sv, vae, text_encoder
    torch.cuda.empty_cache()

    cache_name = "test_verify"
    # attn_window in units of chunks, not tokens — match official server
    attn_window = T + 2
    transformer.create_empty_cache(cache_name, attn_window,
                                   tokens_per_frame, 1,
                                   device=device, dtype=dtype, batch_size=1)

    incremental_outs = []
    for t in range(T):
        grid_id_t = get_mesh_id(1, post_patch_h, post_patch_w, 0, 1, t).to(device)
        with torch.no_grad():
            transformer({
                "noisy_latents": latent_list[t].to(dtype),
                "timesteps": torch.zeros([1, 1], dtype=torch.float32, device=device),
                "grid_id": grid_id_t[None],
                "text_emb": text_emb,
            }, update_cache=2, cache_name=cache_name, action_mode=False)
        incremental_outs.append(features["out"].clone())

    h.remove()
    transformer.clear_cache(cache_name)

    out_incr = torch.cat(incremental_outs, dim=1)

    # Compare
    print(f"\n  All-at-once: {out_all.shape}")
    print(f"  Incremental: {out_incr.shape}")

    for t in range(T):
        s, e = t * tokens_per_frame, (t + 1) * tokens_per_frame
        diff = (out_all[:, s:e] - out_incr[:, s:e]).abs()
        print(f"  Frame {t}: max_diff={diff.max().item():.8f}  mean_diff={diff.mean().item():.8f}")

    total_diff = (out_all - out_incr).abs()
    max_diff = total_diff.max().item()
    mean_diff = total_diff.mean().item()

    if max_diff < 0.05:
        print(f"\n  ✓ EQUIVALENT: max_diff={max_diff:.8f} (within bf16 numerical precision)")
    else:
        print(f"\n  ✗ MISMATCH: max_diff={max_diff:.8f}")

    return max_diff


if __name__ == "__main__":
    vae_ok = test_vae_cache_crashes()
    transformer_diff = test_transformer_equivalence()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  VAE: clear_cache per frame correct (crash verified): {vae_ok}")
    print(f"  Transformer: update_cache=0 ≈ update_cache=2, max_diff: {transformer_diff:.8f}")
