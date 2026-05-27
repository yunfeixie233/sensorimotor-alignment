"""
Phase 2 (Multi-Frame): Extract LingBot-VA features for multi-frame CKNNA experiment.

Supports two modes:
  --mode joint   (Exp A): Feed T frames jointly through causal VAE + transformer.
                 Temporal attention is active.
  --mode indep   (Exp B): Feed each frame independently (clear_cache each time,
                 f_shift=0). No temporal attention.

Output (Solution C — per-camera per-frame pooling):
  feats_A_percam.pt  — [N, T, 3, 3072] bfloat16, per-camera per-frame mean-pooled
  feats_A.pt         — [N, 3072] float32, global mean-pooled (backward compat)

Multi-frame data layout (from load_droid_multiframe.py):
  images/NNNNNN/000.png ... {m_max:03d}.png
  For --m=4 with m_max=64: load frames 060.png ... 064.png (last 5)

Usage:
  # Exp A: joint multi-frame (m=16)
  python extract_features_multiframe.py \
      --transformer_path /path/to/transformer \
      --vae_path /path/to/vae \
      --text_encoder_path /path/to/text_encoder \
      --tokenizer_path /path/to/tokenizer \
      --data_dir /path/to/cknna_data_droid_multiframe \
      --output_dir /path/to/output/lingbot-va-joint-m16 \
      --m 16 --mode joint

  # Exp B: independent per-frame (m=16)
  python extract_features_multiframe.py \
      ... same args ... \
      --output_dir /path/to/output/lingbot-va-indep-m16 \
      --m 16 --mode indep
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
# Image loading
# --------------------------------------------------------------------------

def load_image(path, height, width):
    """Load a PNG image, resize, normalize to [-1, 1]. Retries once if corrupt."""
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        # Zero-byte or corrupt file — delete and raise to skip this sample
        import os
        sz = os.path.getsize(path) if os.path.exists(path) else -1
        raise ValueError(f"Corrupt image {path} ({sz} bytes)")
    img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1)  # [3, H, W]
    img_tensor = F.interpolate(
        img_tensor.unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )  # [1, 3, H, W]
    img_tensor = img_tensor / 255.0 * 2.0 - 1.0
    return img_tensor  # [1, 3, H, W]


# --------------------------------------------------------------------------
# VAE encoding
# --------------------------------------------------------------------------

def _vae_encode_one(streaming_vae, frame_tensor, dtype, vae_device, latents_mean, latents_std):
    """Encode a single [1, 3, 1, H, W] or [N_cams, 3, 1, H, W] through VAE.

    Always clears VAE cache before encoding. This is correct because:
      - VAE temporal convs have kernel (3,1,1) with zero padding
      - Cache from T=1 frame + new T=1 = temporal dim 2 < kernel 3 → crash
      - Official _encode_obs() also always operates on cold-start cache
      - Multi-frame temporal modeling happens at TRANSFORMER level, not VAE

    For multi-camera: cameras are batched on dim=0, encoded together,
    then latents concatenated along width (matching official _encode_obs
    for non-robotwin: torch.cat(mu_norm.split(1, dim=0), dim=-1)).

    Returns: [1, 48, 1, H_lat, W_lat] (1-cam) or [1, 48, 1, H_lat, W_lat*N] (N-cam)
    """
    streaming_vae.clear_cache()
    enc_out = streaming_vae.encode_chunk(frame_tensor.to(vae_device).to(dtype))
    mu, logvar = torch.chunk(enc_out, 2, dim=1)
    # latents_std is pre-inverted (1/std), matching official normalize_latents(mu, mean, 1/std)
    mu_norm = ((mu.float() - latents_mean) * latents_std).to(mu)

    if mu_norm.shape[0] > 1:
        # Multi-camera: concat along width (last dim), matching server.py:373
        mu_norm = torch.cat(mu_norm.split(1, dim=0), dim=-1)  # [1, 48, 1, h, w*N_cams]

    return mu_norm


def encode_multiframe_joint(streaming_vae, frame_list, dtype, vae_device):
    """Encode T frames for Exp A: each frame VAE-encoded independently,
    then concatenate latents along temporal dim.

    Each frame is a cold-start VAE encoding (clear_cache per frame) because:
      - VAE causal temporal convs: kernel (3,1,1), padding (0,0,0)
      - Cache from T=1 + new T=1 = temporal 2 < kernel 3 → would crash
      - Official server also encodes each observation independently
      - All multi-frame temporal modeling happens at the transformer level
        via self-attention over T×tokens_per_frame tokens

    Args:
        frame_list: list of T tensors, each [1, 3, 1, H, W] (1-cam)
                    or [N_cams, 3, 1, H, W] (multi-cam)
    Returns:
        [1, 48, T, H_lat, W_lat] (1-cam) or [1, 48, T, H_lat, W_lat*N] (N-cam)
    """
    latents_mean = torch.tensor(streaming_vae.vae.config.latents_mean).to(vae_device)
    latents_std = 1.0 / torch.tensor(streaming_vae.vae.config.latents_std).to(vae_device)
    latents_mean = latents_mean.view(1, -1, 1, 1, 1)
    latents_std = latents_std.view(1, -1, 1, 1, 1)

    latent_list = []
    for frame in frame_list:
        lat = _vae_encode_one(streaming_vae, frame, dtype, vae_device, latents_mean, latents_std)
        latent_list.append(lat)

    return torch.cat(latent_list, dim=2)  # [1, 48, T, H_lat, W_lat(*N)]


def encode_single_frame(streaming_vae, frame_tensor, dtype, vae_device):
    """Encode a single frame independently (Exp B). Same as joint per-frame encoding."""
    latents_mean = torch.tensor(streaming_vae.vae.config.latents_mean).to(
        next(streaming_vae.vae.parameters()).device)
    latents_std = 1.0 / torch.tensor(streaming_vae.vae.config.latents_std).to(latents_mean.device)
    latents_mean = latents_mean.view(1, -1, 1, 1, 1)
    latents_std = latents_std.view(1, -1, 1, 1, 1)
    return _vae_encode_one(streaming_vae, frame_tensor, dtype,
                           latents_mean.device, latents_mean, latents_std)


# --------------------------------------------------------------------------
# Text encoding (same as single-frame extractor)
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
    parser = argparse.ArgumentParser(
        description="Phase 2 (Multi-Frame): Extract LingBot-VA features for CKNNA."
    )
    # Model paths
    parser.add_argument("--transformer_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--text_encoder_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Multi-frame data dir (from load_droid_multiframe.py)")
    parser.add_argument("--output_dir", type=str, required=True)
    # Multi-frame parameters
    parser.add_argument("--m", type=int, required=True,
                        help="Number of lookback frames (0 = single anchor frame only)")
    parser.add_argument("--mode", type=str, required=True, choices=["joint", "indep"],
                        help="joint = Exp A (temporal attn ON), indep = Exp B (temporal attn OFF)")
    # Camera mode
    parser.add_argument("--num_cameras", type=int, default=3, choices=[1, 3],
                        help="1=cam1 only, 3=all cameras side-by-side (matches official Franka/DROID config)")
    # Action history
    parser.add_argument("--with_actions", action="store_true", default=False,
                        help="Include action history tokens in KV cache (matches official _compute_kv_cache)")
    # Options
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=320)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--resume_from", type=int, default=0)
    args = parser.parse_args()

    assert args.image_height % 32 == 0 and args.image_width % 32 == 0

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load metadata ---
    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    m_max = metadata["m_max"]
    task_descriptions = metadata["task_descriptions"]

    # Camera directories
    cam_dirs = [os.path.join(args.data_dir, "images")]  # cam1 always
    if args.num_cameras == 3:
        cam_dirs.append(os.path.join(args.data_dir, "images_cam2"))
        cam_dirs.append(os.path.join(args.data_dir, "images_cam3"))
        for d in cam_dirs[1:]:
            assert os.path.isdir(d), f"3-camera mode requires {d}. Run Phase 1 with 3 cameras."

    assert args.m <= m_max, f"--m={args.m} exceeds m_max={m_max} in data"

    if args.max_samples > 0:
        num_samples = min(num_samples, args.max_samples)

    T = args.m + 1  # total frames: m lookback + 1 anchor
    print(f"=== Multi-Frame Feature Extraction ===")
    print(f"  Mode: {args.mode} ({'Exp A: joint temporal attention' if args.mode == 'joint' else 'Exp B: independent per-frame'})")
    print(f"  m = {args.m} (T = {T} frames per sample)")
    print(f"  m_max = {m_max} (stored in data)")
    print(f"  Samples: {num_samples}")
    print(f"  With action history: {args.with_actions}")

    if args.with_actions and args.mode == "indep":
        print("  WARNING: --with_actions has no effect in --mode indep (no KV cache)")

    # --- Load and preprocess action history ---
    actions_30d = None
    if args.with_actions and args.mode == "joint":
        actions_hist_path = os.path.join(args.data_dir, "actions_history.pt")
        assert os.path.exists(actions_hist_path), \
            f"--with_actions requires {actions_hist_path}. Run load_droid_multiframe.py first."
        actions_history_raw = torch.load(actions_hist_path, weights_only=True)  # [N_all, 65, 7]
        print(f"  Loaded actions_history: {tuple(actions_history_raw.shape)}")

        # Franka config: map 7D DROID → 30D (option A: dims 0-6 → channels 0-6)
        ACTION_DIM_30 = 30
        USED_ACTION_CHANNEL_IDS = (
            list(range(0, 7)) + list(range(28, 29)) +
            list(range(7, 14)) + list(range(29, 30))
        )
        action_mask_30 = torch.zeros(ACTION_DIM_30, dtype=torch.bool)
        action_mask_30[USED_ACTION_CHANNEL_IDS] = True

        q01 = torch.tensor([
            0.3051295876502991, -0.22647984325885773, 0.19957000017166138,
            -0.022680532187223434, -0.05553057789802551, -0.2693849802017212,
            -0.29341773986816405, 0.2935442328453064, -0.4431332051753998,
            0.21256473660469055, -0.7962440848350525, -0.40816226601600647,
            -0.28359392285346985, -0.44507765769958496
        ] + [0.] * 16, dtype=torch.float32)

        q99 = torch.tensor([
            0.7572150230407715, 0.47736290097236633, 0.6428080797195435,
            0.9835678935050964, 0.9927203059196472, 0.28041139245033264,
            0.47529348731040877, 0.7564866304397571, 0.04082797020673729,
            0.5355993628501885, 0.9976375699043274, 0.8973174452781656,
            0.6016915678977965, 0.5027598619461056
        ] + [0.] * 14 + [1.0, 1.0], dtype=torch.float32)

        N_act, T_act, _ = actions_history_raw.shape
        actions_30d = torch.zeros(N_act, T_act, ACTION_DIM_30, dtype=torch.float32)
        actions_30d[:, :, 0:7] = actions_history_raw  # 7D → channels 0-6
        actions_30d = (actions_30d - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        actions_30d[:, :, ~action_mask_30] = 0.0
        print(f"  Preprocessed to 30D: {tuple(actions_30d.shape)}, "
              f"range [{actions_30d.min():.2f}, {actions_30d.max():.2f}]")

    # --- Load models ---
    device = args.device
    dtype = torch.bfloat16

    print(f"\nLoading transformer: {args.transformer_path}")
    import json as _json
    _cfg_path = os.path.join(args.transformer_path, "config.json")
    with open(_cfg_path) as _f:
        _cfg = _json.load(_f)
    if _cfg.get("attn_mode") == "flex":
        print("  Overriding attn_mode: flex -> torch")
        _cfg["attn_mode"] = "torch"
        with open(_cfg_path, "w") as _f:
            _json.dump(_cfg, _f, indent=2)
    transformer = load_transformer(args.transformer_path, dtype, device)
    transformer.eval().requires_grad_(False)

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

    latent_h = args.image_height // 16
    latent_w_single = args.image_width // 16
    if args.num_cameras == 3:
        # 3 cameras side-by-side: concat along width in latent space
        latent_w = latent_w_single * args.num_cameras  # e.g. 20*3 = 60
    else:
        latent_w = latent_w_single
    post_patch_h = latent_h // patch_size[1]
    post_patch_w = latent_w // patch_size[2]
    post_patch_w_single = latent_w_single // patch_size[2]  # per-camera width (e.g. 10)
    tokens_per_frame = post_patch_h * post_patch_w
    tokens_per_cam = post_patch_h * post_patch_w_single  # 80 per camera

    print(f"\n  Cameras: {args.num_cameras}")
    print(f"  Image (per camera): {args.image_height}x{args.image_width}")
    print(f"  Combined latent: {latent_h}x{latent_w}")
    print(f"  Patch size: {patch_size}")
    print(f"  Tokens per frame: {post_patch_h}x{post_patch_w} = {tokens_per_frame}")
    print(f"  Tokens per camera per frame: {post_patch_h}x{post_patch_w_single} = {tokens_per_cam}")
    if args.mode == "joint":
        print(f"  Total tokens (T={T}): {T * tokens_per_frame}")
    print(f"  Output: feats_A_percam.pt [N, {T}, {args.num_cameras}, 3072] bf16")

    def _percam_pool(norm_out_raw, num_frames):
        """Pool norm_out per-camera per-frame.

        Args:
            norm_out_raw: [1, num_frames*tokens_per_frame, D]
            num_frames: T (number of temporal frames)
        Returns:
            [num_frames, num_cameras, D] in bfloat16
        """
        tokens = norm_out_raw.squeeze(0)  # [T*tpf, D]
        D = tokens.shape[-1]
        # Reshape to [T, h, w, D]
        tokens = tokens.view(num_frames, post_patch_h, post_patch_w, D)
        cam_feats = []
        for c in range(args.num_cameras):
            # Slice this camera's width columns
            cam_tokens = tokens[:, :, c * post_patch_w_single:(c + 1) * post_patch_w_single, :]
            # Mean-pool spatial dims → [T, D]
            cam_feats.append(cam_tokens.reshape(num_frames, -1, D).mean(dim=1))
        # Stack cameras → [T, num_cameras, D]
        return torch.stack(cam_feats, dim=1).to(torch.bfloat16).cpu()

    # --- Precompute grid_id(s) ---
    if args.mode == "joint":
        # Exp A: incremental KV cache matching official _compute_kv_cache path.
        # Each frame gets its own grid_id with f_shift=frame_index.
        # Precompute grid_ids for all possible frame positions.
        grid_ids_per_frame = []
        for t in range(T):
            gid = get_mesh_id(
                1,             # f: 1 frame at a time (incremental)
                post_patch_h,
                post_patch_w,
                0,             # t: type (0 = latent)
                1,             # f_w: frame weight
                t,             # f_shift: this frame's temporal position
            ).to(device)
            grid_ids_per_frame.append(gid[None])  # [1, 4, tokens_per_frame]

        # Create KV cache — sized for T frames of latent + action tokens
        cache_name = "cknna_extract"
        # attn_window in the official code: total_tolen = (attn_window//2) * latent_tokens + (attn_window//2) * action_tokens
        # action_per_frame=1: DROID is 15Hz camera + 15Hz action (1:1).
        # We encode each raw frame independently (no VAE temporal compression),
        # so each latent frame = 1 raw frame = 1 action.
        # NOTE: action_per_frame=4 would be correct ONLY if video were subsampled
        # at stride 4 (matching VAE temporal downsample), but we use stride=1.
        ACTION_PER_FRAME = 1
        action_token_per_chunk = ACTION_PER_FRAME if (args.with_actions and actions_30d is not None) else 0
        attn_window = T * 2 + 2  # generous: (attn_window//2) * tokens_per_frame >= T * tokens_per_frame
        transformer.create_empty_cache(
            cache_name, attn_window, tokens_per_frame, action_token_per_chunk,
            device=device, dtype=dtype, batch_size=1)
        print(f"  KV cache created: attn_window={attn_window}, tokens_per_frame={tokens_per_frame}, "
              f"action_token_per_chunk={action_token_per_chunk}")

        # Precompute action grid_ids (if using action history)
        action_grid_ids_per_frame = []
        if action_token_per_chunk > 0:
            for t in range(T):
                gid = get_mesh_id(
                    1,                 # f: 1 frame at a time
                    ACTION_PER_FRAME,  # h: 1 action per frame (DROID stride=1)
                    1,                 # w: 1
                    1,                 # t: type = 1 (action)
                    1,                 # f_w
                    t,                 # f_shift: temporal position
                    action=True,
                ).to(device)
                action_grid_ids_per_frame.append(gid[None])  # [1, 4, 1]
    else:
        # Exp B: grid_id for single frame (reused for each independent frame)
        grid_id_single = get_mesh_id(
            1,             # f: 1 frame
            post_patch_h,
            post_patch_w,
            0,
            1,
            0,             # f_shift=0: each frame treated as cold-start position 0
        ).to(device)
        grid_id_single_batched = grid_id_single[None]  # [1, 4, tokens_per_frame]
        cache_name = None

    # --- Register norm_out hook ---
    features = {}

    def norm_out_hook(module, input, output):
        features["norm_out_raw"] = output.detach().float().clone()

    hook_handle = transformer.norm_out.register_forward_hook(norm_out_hook)

    # --- Main extraction loop ---
    feats_list = []
    feats_percam_list = []

    if args.resume_from > 0:
        partial_path = os.path.join(args.output_dir, "feats_A_partial.pt")
        partial_percam_path = os.path.join(args.output_dir, "feats_A_percam_partial.pt")
        if os.path.exists(partial_path):
            feats_list = list(torch.load(partial_path, weights_only=True))
            print(f"  Resumed {len(feats_list)} samples from partial save")
        if os.path.exists(partial_percam_path):
            feats_percam_list = list(torch.load(partial_percam_path, weights_only=True))
            print(f"  Resumed {len(feats_percam_list)} percam samples from partial save")

    t0 = time.time()
    start_idx = args.resume_from if args.resume_from > 0 else len(feats_list)

    def _load_frame(sample_idx, frame_offset):
        """Load one frame from all cameras, return tensor for VAE.

        1-camera: [1, 3, 1, H, W]
        3-camera: [3, 3, 1, H, W]  (batched on dim=0, matching official _encode_obs)
        """
        cam_tensors = []
        for cam_dir in cam_dirs:
            img_path = os.path.join(cam_dir, f"{sample_idx:06d}", f"{frame_offset:03d}.png")
            img = load_image(img_path, args.image_height, args.image_width)
            img = img.unsqueeze(2)  # [1, 3, 1, H, W]
            cam_tensors.append(img)
        if len(cam_tensors) == 1:
            return cam_tensors[0]  # [1, 3, 1, H, W]
        return torch.cat(cam_tensors, dim=0)  # [N_cams, 3, 1, H, W]

    valid_indices = []  # track which sample indices were successfully processed
    skipped_indices = []

    for i in range(start_idx, num_samples):
        frame_start = m_max - args.m  # first frame offset to load

        try:
            if args.mode == "joint":
                # ---- EXP A: Incremental KV cache (matches official _compute_kv_cache) ----
                # 1. VAE-encode each frame independently (clear_cache per frame)
                frame_latents = []
                for j in range(frame_start, m_max + 1):
                    frame_tensor = _load_frame(i, j)
                    with torch.no_grad():
                        lat = encode_single_frame(streaming_vae, frame_tensor, dtype, device)
                    frame_latents.append(lat)  # each [1, 48, 1, h, w*3]

                # 2. Encode text
                task = task_descriptions[i]
                with torch.no_grad():
                    text_emb = encode_text(text_encoder, tokenizer, task, device=device, dtype=dtype)

                # 3. Clear KV cache for this sample (fresh start like _reset)
                transformer.clear_cache(cache_name)
                transformer.create_empty_cache(
                    cache_name, T * 2 + 2, tokens_per_frame, action_token_per_chunk,
                    device=device, dtype=dtype, batch_size=1)

                # 4. Feed frames incrementally with update_cache=2
                #    Matches official _compute_kv_cache: video pass then action pass per frame.
                frame_norm_outs = []
                for t in range(T):
                    # --- Video pass ---
                    input_dict = {
                        "noisy_latents": frame_latents[t].to(dtype),
                        "timesteps": torch.zeros([1, 1], dtype=torch.float32, device=device),
                        "grid_id": grid_ids_per_frame[t],
                        "text_emb": text_emb,
                    }
                    with torch.no_grad():
                        transformer(input_dict, update_cache=2,
                                    cache_name=cache_name, action_mode=False)
                    frame_norm_outs.append(features["norm_out_raw"].clone())  # [1, tpf, 3072]

                    # --- Action pass (if enabled) ---
                    if action_token_per_chunk > 0 and actions_30d is not None:
                        frame_offset = frame_start + t
                        action_vec = actions_30d[i, frame_offset]  # [30]
                        action_input = action_vec.to(dtype=dtype, device=device)
                        action_input = action_input.view(1, 30, 1, 1, 1)  # [B, C, F=1, H=1, W=1]
                        action_input_dict = {
                            "noisy_latents": action_input,
                            "timesteps": torch.zeros([1, 1], dtype=torch.float32, device=device),
                            "grid_id": action_grid_ids_per_frame[t],
                            "text_emb": text_emb,
                        }
                        with torch.no_grad():
                            transformer(action_input_dict, update_cache=2,
                                        cache_name=cache_name, action_mode=True)

                # 5. Concatenate all frames' norm_out and pool
                norm_out_feat = torch.cat(frame_norm_outs, dim=1)  # [1, T*tpf, 3072]
                feat_percam = _percam_pool(norm_out_feat, T)  # [T, 3, 3072] bf16
                feat_pooled = norm_out_feat.mean(dim=1).cpu().squeeze(0)  # [3072] f32

            else:
                # ---- EXP B: Independent per-frame encoding ----
                frame_feats = []
                frame_feats_percam = []
                task = task_descriptions[i]
                with torch.no_grad():
                    text_emb = encode_text(text_encoder, tokenizer, task, device=device, dtype=dtype)

                for j in range(frame_start, m_max + 1):
                    frame_tensor = _load_frame(i, j)

                    with torch.no_grad():
                        latent = encode_single_frame(streaming_vae, frame_tensor, dtype, device)

                    timesteps = torch.zeros([1, 1], dtype=torch.float32, device=device)

                    input_dict = {
                        "noisy_latents": latent.to(dtype),
                        "timesteps": timesteps,
                        "grid_id": grid_id_single_batched,
                        "text_emb": text_emb,
                    }

                    with torch.no_grad():
                        transformer(input_dict, update_cache=0, action_mode=False)

                    norm_out_feat = features["norm_out_raw"]  # [1, tpf, 3072]
                    # Per-camera pool for this single frame
                    frame_percam = _percam_pool(norm_out_feat, 1)  # [1, 3, 3072]
                    frame_feats_percam.append(frame_percam.squeeze(0))  # [3, 3072]
                    feat_j = norm_out_feat.mean(dim=1).cpu().squeeze(0)  # [3072]
                    frame_feats.append(feat_j)

                feat_percam = torch.stack(frame_feats_percam, dim=0)  # [T, 3, 3072] bf16
                feat_pooled = torch.stack(frame_feats).mean(dim=0)  # [3072]

        except (ValueError, FileNotFoundError, OSError) as e:
            print(f"  WARNING: skipping sample {i} — {e}")
            skipped_indices.append(i)
            continue

        feats_percam_list.append(feat_percam)
        feats_list.append(feat_pooled)
        valid_indices.append(i)

        # Progress
        if (i + 1) % 50 == 0 or i == start_idx:
            elapsed = time.time() - t0
            processed = i - start_idx + 1
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = num_samples - i - 1
            eta = remaining / rate if rate > 0 else 0
            print(f"  [{i+1}/{num_samples}]  rate={rate:.2f}/s  ETA={eta/60:.1f}min"
                  f"  mode={args.mode}  m={args.m}")

        # Periodic save
        if args.save_every > 0 and (i + 1) % args.save_every == 0:
            torch.save(torch.stack(feats_list),
                       os.path.join(args.output_dir, "feats_A_partial.pt"))
            torch.save(torch.stack(feats_percam_list),
                       os.path.join(args.output_dir, "feats_A_percam_partial.pt"))
            print(f"    (partial save at {i+1})")

    # --- Cleanup & save ---
    hook_handle.remove()

    print("\nSaving final results...")

    # Save per-camera per-frame features (Solution C) — primary output
    feats_A_percam = torch.stack(feats_percam_list)  # [N, T, 3, 3072] bf16
    torch.save(feats_A_percam, os.path.join(args.output_dir, "feats_A_percam.pt"))
    print(f"  feats_A_percam.pt: {tuple(feats_A_percam.shape)} {feats_A_percam.dtype}"
          f"  ({os.path.getsize(os.path.join(args.output_dir, 'feats_A_percam.pt')) / 1e6:.0f} MB)")

    # Save global mean-pooled features (backward compat)
    feats_A = torch.stack(feats_list)  # [N, 3072] f32
    torch.save(feats_A, os.path.join(args.output_dir, "feats_A.pt"))
    print(f"  feats_A.pt: {tuple(feats_A.shape)}")

    # Clean up partial saves
    for partial_name in ["feats_A_partial.pt", "feats_A_percam_partial.pt"]:
        partial = os.path.join(args.output_dir, partial_name)
        if os.path.exists(partial):
            os.remove(partial)

    # Save extraction metadata
    extraction_meta = {
        "model": "lingbot-va",
        "experiment": "multiframe_cknna",
        "mode": args.mode,
        "mode_description": (
            "Exp A: joint multi-frame, temporal attention ON, causal VAE + transformer"
            if args.mode == "joint" else
            "Exp B: independent per-frame, temporal attention OFF, clear_cache each frame, f_shift=0"
        ),
        "m": args.m,
        "T": T,
        "m_max": m_max,
        "extraction_point": "norm_out",
        "pooling_percam": (
            f"per-camera per-frame: {args.num_cameras} cameras x {tokens_per_cam} tokens each, "
            f"mean-pooled per camera → [T, {args.num_cameras}, {hidden_dim}] bf16"
        ),
        "pooling_global": (
            f"mean over all {T}*{tokens_per_frame}={T*tokens_per_frame} tokens → [{hidden_dim}] f32"
            if args.mode == "joint" else
            f"mean of {T} independent features, each mean-pooled over {tokens_per_frame} tokens → [{hidden_dim}] f32"
        ),
        "camera_token_mapping": {
            "cam1_cam_high": f"w=0:{post_patch_w_single} ({tokens_per_cam} tokens/frame)",
            "cam2_cam_left_wrist": f"w={post_patch_w_single}:{2*post_patch_w_single} ({tokens_per_cam} tokens/frame)",
            "cam3_cam_right_wrist": f"w={2*post_patch_w_single}:{3*post_patch_w_single} ({tokens_per_cam} tokens/frame)",
        },
        "hidden_size": hidden_dim,
        "tokens_per_frame": tokens_per_frame,
        "tokens_per_camera": tokens_per_cam,
        "num_cameras": args.num_cameras,
        "num_samples": len(feats_list),
        "num_skipped": len(skipped_indices),
        "skipped_indices": skipped_indices,
        "valid_indices": valid_indices,
        "transformer_path": args.transformer_path,
        "image_size_per_camera": [args.image_height, args.image_width],
        "combined_latent_size": [latent_h, latent_w],
        "timestep": 0.0,
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    total_time = time.time() - t0
    print(f"\n=== Extraction complete in {total_time:.0f}s ({total_time/60:.1f} min) ===")
    print(f"  Mode: {args.mode}, m={args.m}, N={len(feats_list)}")
    if skipped_indices:
        print(f"  Skipped {len(skipped_indices)} samples: {skipped_indices}")


if __name__ == "__main__":
    main()
