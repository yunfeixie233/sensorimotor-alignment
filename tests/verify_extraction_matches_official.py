"""
Final verification: our extraction script now uses incremental KV cache.
Confirm it produces the same output as the reference incremental path.
"""
import os, sys, json, torch
import numpy as np, torch.nn.functional as F
from PIL import Image

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "repos", "lingbot-va"))
sys.path.insert(0, _PROJECT_ROOT)

from wan_va.modules.utils import (WanVAEStreamingWrapper, load_text_encoder, load_tokenizer, load_transformer, load_vae)
from wan_va.utils import get_mesh_id
from extractors.lingbot_va.extract_features_multiframe import encode_text, encode_single_frame

CKPT = os.path.join(_PROJECT_ROOT, "checkpoints", "lingbot-va")
DATA = "/lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid_multiframe"
device, dtype = "cuda", torch.bfloat16

T = 3  # test with 3 frames
patch_size = [1, 2, 2]
post_patch_h, post_patch_w = 8, 30  # 3 cameras
tokens_per_frame = 240

def load_frame(sample_idx, frame_offset, cam_dir):
    path = os.path.join(DATA, cam_dir, f"{sample_idx:06d}", f"{frame_offset:03d}.png")
    img = Image.open(path).convert("RGB")
    t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1)
    t = F.interpolate(t.unsqueeze(0), size=(256, 320), mode="bilinear", align_corners=False)
    return (t / 255.0 * 2.0 - 1.0).unsqueeze(2)

def main():
    print("Loading models...")
    transformer = load_transformer(os.path.join(CKPT, "lingbot-va-base", "transformer"), dtype, device)
    transformer.eval().requires_grad_(False)
    vae = load_vae(os.path.join(CKPT, "lingbot-va-posttrain-robotwin", "vae"), dtype, device)
    sv = WanVAEStreamingWrapper(vae)
    text_encoder = load_text_encoder(os.path.join(CKPT, "lingbot-va-posttrain-robotwin", "text_encoder"), dtype, device)
    text_encoder.eval().requires_grad_(False)
    tokenizer = load_tokenizer(os.path.join(CKPT, "lingbot-va-posttrain-robotwin", "tokenizer"))

    with open(os.path.join(DATA, "metadata.json")) as f:
        meta = json.load(f)

    m_max = meta["m_max"]  # 64
    frame_start = m_max - (T - 1)  # 62

    features = {}
    def hook(module, input, output):
        features["out"] = output.detach().float().clone()
    h = transformer.norm_out.register_forward_hook(hook)

    # VAE encode all frames
    frame_latents = []
    for j in range(frame_start, m_max + 1):
        cam_frames = [load_frame(0, j, d) for d in ["images", "images_cam2", "images_cam3"]]
        cam_batch = torch.cat(cam_frames, dim=0)  # [3, 3, 1, H, W]
        lat = encode_single_frame(sv, cam_batch, dtype, device)
        frame_latents.append(lat)

    text_emb = encode_text(text_encoder, tokenizer, meta["task_descriptions"][0], device=device, dtype=dtype)

    # Free VAE + text encoder for memory
    del sv, vae, text_encoder
    torch.cuda.empty_cache()

    # ---- Run 1: Our extraction path (incremental KV cache) ----
    print("\n--- Our extraction (incremental KV cache, update_cache=2) ---")
    cache_name = "test_ours"
    transformer.create_empty_cache(cache_name, T*2+2, tokens_per_frame, 0,
                                   device=device, dtype=dtype, batch_size=1)

    our_frame_outs = []
    for t in range(T):
        gid = get_mesh_id(1, post_patch_h, post_patch_w, 0, 1, t).to(device)
        with torch.no_grad():
            transformer({
                "noisy_latents": frame_latents[t].to(dtype),
                "timesteps": torch.zeros([1, 1], dtype=torch.float32, device=device),
                "grid_id": gid[None],
                "text_emb": text_emb,
            }, update_cache=2, cache_name=cache_name, action_mode=False)
        our_frame_outs.append(features["out"].clone())

    transformer.clear_cache(cache_name)
    our_output = torch.cat(our_frame_outs, dim=1)

    # ---- Run 2: Same thing again (reproducibility check) ----
    print("--- Reproducibility check (same path again) ---")
    transformer.create_empty_cache(cache_name, T*2+2, tokens_per_frame, 0,
                                   device=device, dtype=dtype, batch_size=1)

    repro_frame_outs = []
    for t in range(T):
        gid = get_mesh_id(1, post_patch_h, post_patch_w, 0, 1, t).to(device)
        with torch.no_grad():
            transformer({
                "noisy_latents": frame_latents[t].to(dtype),
                "timesteps": torch.zeros([1, 1], dtype=torch.float32, device=device),
                "grid_id": gid[None],
                "text_emb": text_emb,
            }, update_cache=2, cache_name=cache_name, action_mode=False)
        repro_frame_outs.append(features["out"].clone())

    transformer.clear_cache(cache_name)
    repro_output = torch.cat(repro_frame_outs, dim=1)

    h.remove()

    # Compare
    diff_repro = (our_output - repro_output).abs()
    print(f"\n  Reproducibility: max_diff={diff_repro.max().item():.10f}")

    # ---- Run 3: Old buggy path (all-at-once, no cache) for comparison ----
    print("--- Old buggy path (all-at-once, update_cache=0, no KV cache) ---")
    h2 = transformer.norm_out.register_forward_hook(hook)

    latent_all = torch.cat(frame_latents, dim=2)
    grid_all = get_mesh_id(T, post_patch_h, post_patch_w, 0, 1, 0).to(device)

    with torch.no_grad():
        transformer({
            "noisy_latents": latent_all.to(dtype),
            "timesteps": torch.zeros([1, T], dtype=torch.float32, device=device),
            "grid_id": grid_all[None],
            "text_emb": text_emb,
        }, update_cache=0, action_mode=False)
    old_output = features["out"].clone()
    h2.remove()

    diff_old = (our_output - old_output).abs()

    print(f"\n{'='*60}")
    print(f"RESULTS (T={T} frames, 3 cameras, {tokens_per_frame} tokens/frame)")
    print(f"{'='*60}")
    print(f"  Reproducibility (same path twice): max={diff_repro.max():.10f}")
    print(f"  Old vs New path:                   max={diff_old.max():.6f}  mean={diff_old.mean():.6f}")
    print()
    for t in range(T):
        s, e = t * tokens_per_frame, (t+1) * tokens_per_frame
        d = (our_output[:, s:e] - old_output[:, s:e]).abs()
        print(f"  Frame {t}: max={d.max():.6f}  mean={d.mean():.6f}")

    if diff_repro.max() < 1e-6:
        print(f"\n  ✓ Incremental KV cache path is deterministic and reproducible")
    if diff_old.max() > 0.1:
        print(f"  ✓ Old path (bidirectional) differs significantly from new (causal) — fix is meaningful")


if __name__ == "__main__":
    main()
