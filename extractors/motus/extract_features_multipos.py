"""
Multi-position feature extraction from Motus for CKNNA.

Extracts features from 7 positions within the WAN 2.2-5B backbone in a single
forward pass per sample:

  P0: input_embed           - after prepare_input() (patch_embedding), before block 0
  P1: block14_post_selfattn  - after bimodal joint attention at layer 14
  P2: block14_post_crossattn - after T5 cross-attention at layer 14
  P3: block29_post_selfattn  - after bimodal joint attention at layer 29
  P4: block29_post_crossattn - after T5 cross-attention at layer 29
  P5: block29               - after FFN at layer 29 (= post-loop video_tokens, current primary)
  P6: norm_out              - after wan_model.head.norm (WanLayerNorm, affine=False)

Capture strategy: direct capture from the manual 30-layer loop (no hooks needed).

Note: P5 (block29) is what the original extract_features.py saves as feats_A.pt.
      P6 (norm_out via head.norm) is the true equivalent of LingBot-VA's norm_out.

Usage:
    python extract_features_multipos.py \\
        --checkpoint_path /path/to/motus_robotwin2/mp_rank_00_model_states.pt \\
        --wan_path /path/to/Wan2.2-TI2V-5B \\
        --vlm_path Qwen/Qwen3-VL-2B-Instruct \\
        --data_dir /path/to/cknna_data \\
        --output_dir /path/to/cknna_data/motus-robotwin2-multipos
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MOTUS_ROOT = os.environ.get(
    "MOTUS_ROOT",
    os.path.join(_PROJECT_ROOT, "repos", "Motus"),
)
MOTUS_ROOT = os.path.abspath(MOTUS_ROOT)
if MOTUS_ROOT not in sys.path:
    sys.path.insert(0, MOTUS_ROOT)

BAK_ROOT = os.path.join(MOTUS_ROOT, "bak")
if BAK_ROOT not in sys.path:
    sys.path.insert(0, BAK_ROOT)

from models.motus import Motus, MotusConfig
from wan.modules.t5 import T5EncoderModel


# --- Position names ---
POSITION_NAMES = [
    "input_embed",
    "block14_post_selfattn",
    "block14_post_crossattn",
    "block29_post_selfattn",
    "block29_post_crossattn",
    "block29",
    "norm_out",
]

# Layers at which to capture intra-block features
CAPTURE_LAYERS = {14, 29}


# ---------------------------------------------------------------------------
# Bimodal joint attention (unchanged from extract_features.py)
# ---------------------------------------------------------------------------

def process_bimodal_attention(video_module, video_tokens, video_adaln_modulation,
                              layer_idx, und_tokens, und_block):
    """Bimodal joint self-attention: video + understanding (no action)."""
    wan_layer = video_module.video_model.wan_model.blocks[layer_idx]
    v_mod = video_adaln_modulation

    norm_video = wan_layer.norm1(video_tokens).float() * (1 + v_mod[1].squeeze(2)) + v_mod[0].squeeze(2)
    norm_und = und_block.norm1(und_tokens)

    B, L_v, C = norm_video.shape
    L_u = norm_und.shape[1]
    n = video_module.video_model.wan_model.num_heads
    d = C // n

    u_qkv = torch.einsum("BTD,KNDE->KBTNE", norm_und, und_block.wan_und_qkv)
    u_q_h, u_k_h, u_v_h = u_qkv[0], u_qkv[1], u_qkv[2]
    u_q = und_block.wan_und_norm_q(u_q_h.flatten(-2)).view(B, L_u, n, d)
    u_k = und_block.wan_und_norm_k(u_k_h.flatten(-2)).view(B, L_u, n, d)
    u_v = u_v_h.view(B, L_u, n, d)

    seq_lens = torch.full((B,), L_v + L_u, dtype=torch.long, device=video_tokens.device)
    freqs = video_module.video_model.wan_model.freqs
    if freqs.device != video_tokens.device:
        freqs = freqs.to(video_tokens.device)

    y, _, und_out_h = wan_layer.self_attn(
        norm_video, seq_lens, video_module.grid_sizes, freqs,
        action_q=None, action_k=None, action_v=None,
        und_q=u_q, und_k=u_k, und_v=u_v,
    )

    und_out = und_block.wan_und_o(und_out_h.flatten(2))
    video_tokens = video_tokens + y * v_mod[2].squeeze(2)
    und_tokens = und_tokens + und_out

    return video_tokens, und_tokens


# ---------------------------------------------------------------------------
# Image preprocessing (unchanged from extract_features.py)
# ---------------------------------------------------------------------------

def resize_with_padding(frame, target_size):
    """Resize with aspect ratio preservation and black padding."""
    target_h, target_w = target_size
    orig_h, orig_w = frame.shape[:2]
    scale = min(target_h / orig_h, target_w / orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    padded = np.zeros((target_h, target_w, frame.shape[2]), dtype=frame.dtype)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    padded[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return padded


def prepare_vlm_inputs(vlm_processor, instruction, pil_image, device):
    """Build VLM inputs for UndModule."""
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "image", "image": pil_image},
        ]}
    ]
    text = vlm_processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    encoded = vlm_processor(text=[text], images=[pil_image], return_tensors="pt")
    vlm_inputs = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "pixel_values": encoded["pixel_values"].to(device),
        "image_grid_thw": encoded.get("image_grid_thw", None),
    }
    if vlm_inputs["image_grid_thw"] is not None:
        vlm_inputs["image_grid_thw"] = vlm_inputs["image_grid_thw"].to(device)
    return vlm_inputs


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-position Motus feature extraction for CKNNA.")
    # Model paths
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Motus checkpoint dir or .pt file")
    parser.add_argument("--wan_path", type=str, required=True,
                        help="Wan2.2-TI2V-5B directory (config, VAE, T5)")
    parser.add_argument("--vlm_path", type=str, required=True,
                        help="Qwen3-VL-2B-Instruct model path")
    # Data
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    # Options
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--video_height", type=int, default=384)
    parser.add_argument("--video_width", type=int, default=320)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    dtype = torch.bfloat16

    # --- Load metadata ---
    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")
    if args.max_samples > 0:
        num_samples = min(num_samples, args.max_samples)

    # --- Load Motus model ---
    print("Loading Motus model (no pretrained backbones -> load from checkpoint)...")
    config = MotusConfig(
        wan_checkpoint_path=args.wan_path,
        vae_path=os.path.join(args.wan_path, "Wan2.2_VAE.pth"),
        wan_config_path=args.wan_path,
        video_precision="bfloat16",
        vlm_checkpoint_path=args.vlm_path,
        und_expert_hidden_size=512,
        und_expert_ffn_dim_multiplier=4,
        und_expert_norm_eps=1e-5,
        vlm_adapter_input_dim=2048,
        vlm_adapter_projector_type="mlp3x_silu",
        num_layers=30,
        action_state_dim=14,
        action_dim=14,
        action_expert_dim=1024,
        action_expert_ffn_dim_multiplier=4,
        action_expert_norm_eps=1e-6,
        batch_size=1,
        video_height=args.video_height,
        video_width=args.video_width,
        num_video_frames=8,
        video_action_freq_ratio=2,
        load_pretrained_backbones=False,
        training_mode="finetune",
    )
    model = Motus(config)
    model = model.to(device)

    from pathlib import Path
    ckpt_path = Path(args.checkpoint_path)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "mp_rank_00_model_states.pt"
    checkpoint = torch.load(str(ckpt_path), map_location='cpu')
    state_dict = checkpoint.get('module', checkpoint)
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape != v.shape:
            skipped.append(f"{k}: ckpt={tuple(v.shape)} vs model={tuple(model_state[k].shape)}")
            continue
        filtered[k] = v
    if skipped:
        print(f"  Skipped {len(skipped)} keys with shape mismatch:")
        for s in skipped:
            print(f"    {s}")
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"  Loaded checkpoint: {len(filtered)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Monkey-patch VLM get_image_features for transformers >=5.x compatibility
    _orig_get_image_features = model.und_module.vlm_model.get_image_features
    def _compat_get_image_features(pixel_values, image_grid_thw, **kwargs):
        result = _orig_get_image_features(pixel_values, image_grid_thw, **kwargs)
        if not isinstance(result, tuple):
            return result.pooler_output, getattr(result, 'deepstack_features', None)
        return result
    model.und_module.vlm_model.get_image_features = _compat_get_image_features

    wan_dim = model.video_model.wan_model.config.dim
    num_layers = config.num_layers
    print(f"  WAN dim: {wan_dim}, layers: {num_layers}")

    # Fix grid_sizes for single-frame extraction (B=1)
    lat_H = args.video_height // 32
    lat_W = args.video_width // 32
    grid_sizes = torch.tensor([[1, lat_H, lat_W]], dtype=torch.long, device=device)
    model.video_module.grid_sizes = grid_sizes
    num_video_tokens = 1 * lat_H * lat_W
    print(f"  Single-frame grid: [1, {lat_H}, {lat_W}] = {num_video_tokens} video tokens")
    print(f"  Extraction positions: {POSITION_NAMES}")
    print(f"  Intra-block capture at layers: {sorted(CAPTURE_LAYERS)}")

    # --- Load T5 encoder ---
    print(f"Loading T5 encoder from {args.wan_path}...")
    t5_encoder = T5EncoderModel(
        text_len=512,
        dtype=dtype,
        device=device,
        checkpoint_path=os.path.join(args.wan_path, "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=os.path.join(args.wan_path, "google", "umt5-xxl"),
    )
    print("  T5 encoder loaded.")

    # --- Load VLM processor ---
    print(f"Loading VLM processor from {args.vlm_path}...")
    from transformers import AutoProcessor
    vlm_processor = AutoProcessor.from_pretrained(args.vlm_path, trust_remote_code=True)
    print("  VLM processor loaded.")

    print(f"  Samples to process: {num_samples}")

    # --- Per-position accumulator lists ---
    feats_lists = {name: [] for name in POSITION_NAMES}

    # --- Main extraction loop ---
    t0 = time.time()

    for i in range(num_samples):
        # a) Load and resize image
        img_path = os.path.join(images_dir, f"{i:06d}.png")
        img_np = cv2.imread(img_path)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_resized = resize_with_padding(img_np, (args.video_height, args.video_width))

        pil_image = Image.fromarray(img_resized)
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        task = task_descriptions[i]

        with torch.inference_mode():
            # b) VAE encode single frame
            frame_norm = (img_tensor * 2.0 - 1.0).unsqueeze(2).to(dtype)
            condition_latent = model.video_model.encode_video(frame_norm)

            # c) T5 encode text
            t5_out = t5_encoder([task], device)
            if isinstance(t5_out, list):
                t5_embeddings = t5_out
            elif isinstance(t5_out, torch.Tensor):
                if t5_out.dim() == 3:
                    t5_embeddings = [t5_out.squeeze(0)]
                else:
                    t5_embeddings = [t5_out]
            processed_t5_context = model.video_module.preprocess_t5_embeddings(t5_embeddings)

            # d) VLM -> understanding tokens
            vlm_inputs = prepare_vlm_inputs(vlm_processor, task, pil_image, device)
            vlm_out = model.und_module.vlm_model(
                **vlm_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = vlm_out.hidden_states[-1]
            und_tokens = model.und_module.und_expert.vlm_adapter(last_hidden)

            # e) Prepare video tokens from latent
            video_tokens = model.video_module.prepare_input(condition_latent.to(dtype))

            # P0: input_embed — capture after prepare_input, before any blocks
            feats_lists["input_embed"].append(
                video_tokens.detach().float().mean(dim=1).squeeze(0).cpu()
            )

            # f) Time embedding for t=0
            t_video = torch.zeros(1, device=device, dtype=dtype)
            video_head_time_emb, video_adaln_params = model.video_module.get_time_embedding(
                t_video, video_tokens.shape[1]
            )

            # g) 30-layer bimodal forward with intra-block captures
            with torch.autocast(device_type="cuda", dtype=dtype):
                for layer_idx in range(num_layers):
                    video_adaln_modulation = model.video_module.compute_adaln_modulation(
                        video_adaln_params, layer_idx
                    )

                    # Bimodal joint attention (video + understanding, no action)
                    video_tokens, und_tokens = process_bimodal_attention(
                        model.video_module, video_tokens, video_adaln_modulation,
                        layer_idx, und_tokens, model.und_expert.blocks[layer_idx]
                    )

                    # Capture post-self-attn at target layers
                    if layer_idx in CAPTURE_LAYERS:
                        key = f"block{layer_idx:02d}_post_selfattn"
                        feats_lists[key].append(
                            video_tokens.detach().float().mean(dim=1).squeeze(0).cpu()
                        )

                    # WAN cross-attention with T5 text embeddings
                    video_tokens = model.video_module.process_cross_attention(
                        video_tokens, video_adaln_params, layer_idx, processed_t5_context
                    )

                    # Capture post-cross-attn at target layers
                    if layer_idx in CAPTURE_LAYERS:
                        key = f"block{layer_idx:02d}_post_crossattn"
                        feats_lists[key].append(
                            video_tokens.detach().float().mean(dim=1).squeeze(0).cpu()
                        )

                    # FFNs
                    video_tokens = model.video_module.process_ffn(
                        video_tokens, video_adaln_modulation, layer_idx
                    )
                    und_tokens = model.und_module.process_ffn(und_tokens, layer_idx)

            # P5: block29 — post-loop video_tokens (= post-FFN of last block)
            feats_lists["block29"].append(
                video_tokens.detach().float().mean(dim=1).squeeze(0).cpu()
            )

            # P6: norm_out — apply head.norm (WanLayerNorm, affine=False)
            norm_out_tokens = model.video_model.wan_model.head.norm(video_tokens)
            feats_lists["norm_out"].append(
                norm_out_tokens.detach().float().mean(dim=1).squeeze(0).cpu()
            )

        # Progress
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{num_samples}]  D={wan_dim}  rate={rate:.2f}/s  ETA={eta/60:.1f}min")

        # Periodic save
        if args.save_every > 0 and (i + 1) % args.save_every == 0:
            for name in POSITION_NAMES:
                if feats_lists[name]:
                    torch.save(
                        torch.stack(feats_lists[name]),
                        os.path.join(args.output_dir, f"feats_A_{name}_partial.pt"),
                    )
            print(f"    (partial save at {i+1})")

    # --- Save final results ---
    print("\nSaving final results...")
    N = len(feats_lists["norm_out"])

    for name in POSITION_NAMES:
        feats = torch.stack(feats_lists[name])
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
        "model": "Motus (bimodal: video+understanding, no action)",
        "checkpoint": args.checkpoint_path,
        "extraction_mode": "multi-position (7 positions in single forward pass)",
        "positions": {
            "input_embed": "After prepare_input() (patch_embedding + flatten). Pure image, no text.",
            "block14_post_selfattn": "After bimodal joint attention (video+und) at layer 14.",
            "block14_post_crossattn": "After T5 cross-attention at layer 14.",
            "block29_post_selfattn": "After bimodal joint attention (video+und) at layer 29.",
            "block29_post_crossattn": "After T5 cross-attention at layer 29.",
            "block29": "After FFN at layer 29. Post-loop video_tokens. = original extract_features.py primary.",
            "norm_out": "After wan_model.head.norm (WanLayerNorm, affine=False). True equivalent of LingBot-VA norm_out.",
        },
        "feature_semantics": "img+text inseparable (text via T5 cross-attn + VLM und tokens). input_embed is image-only.",
        "hidden_size": wan_dim,
        "num_video_tokens": num_video_tokens,
        "grid_sizes": [1, lat_H, lat_W],
        "pooling": "mean over video tokens (3072D)",
        "timestep": 0.0,
        "num_layers": num_layers,
        "num_samples": N,
        "outputs": [f"feats_A_{name}.pt" for name in POSITION_NAMES] + ["feats_A.pt"],
        "notes": {
            "bimodal": "Action tokens excluded to prevent proprio leakage.",
            "norm_naming": "Motus WAN uses norm3 before cross-attn (vs LingBot-VA norm2). Capture is semantic, not by name.",
            "block29_vs_norm_out": "block29 = post-FFN (original primary). norm_out = head.norm applied. Difference is LayerNorm only.",
        },
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== Motus Multi-Position Extraction Complete ===")
    print(f"  N={N} samples, {elapsed/60:.1f} min ({elapsed/N:.2f} s/sample)")
    print(f"  Positions: {len(POSITION_NAMES)}")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
