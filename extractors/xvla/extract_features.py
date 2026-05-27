"""
Phase 2: Extract VLM features from X-VLA for CKNNA.

Architecture: Florence2 = DaViT vision encoder + BART encoder (encoder-only, decoder deleted).
              Single-view processing (1 image per sample for CKNNA).

Extraction point: BART encoder output via forward_vlm() → enc_out [B, N_img+L, D=1024]
  Token layout (from _merge_input_ids_with_image_features):
    [image_tokens (N_img) | text_tokens (L=50 padded)]
  N_img determined by DaViT + image_feature_source config (probed at startup).

Pooling:
  feats_A:     mean-pool over image + non-padding text tokens
  feats_A_img: mean-pool over image tokens only
  feats_A_txt: mean-pool over non-padding text tokens only

Usage:
    python extract_features.py \
        --ckpt 2toINF/X-VLA-WidowX \
        --data_dir /path/to/cknna_data \
        --output_dir /path/to/cknna_data/xvla-widowx
"""

import argparse
import json
import os
import sys
import time

import torch
from PIL import Image

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
XVLA_ROOT = os.environ.get(
    "XVLA_ROOT",
    os.path.join(_PROJECT_ROOT, "repos", "X-VLA"),
)
XVLA_ROOT = os.path.abspath(XVLA_ROOT)
if XVLA_ROOT not in sys.path:
    sys.path.insert(0, XVLA_ROOT)

from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor


def masked_mean_pool(hidden_states, mask):
    """Mean-pool hidden states over valid tokens indicated by mask."""
    h = hidden_states.float()
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Extract X-VLA features for CKNNA.")
    parser.add_argument("--ckpt", type=str, default="2toINF/X-VLA-WidowX",
                        help="HuggingFace model ID or local path")
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
    # Manual loading to avoid transformers >=5.x meta-tensor init issues with Florence2
    print(f"Loading X-VLA: {args.ckpt}")
    from models.modeling_xvla import XVLAConfig
    config = XVLAConfig.from_pretrained(args.ckpt)
    with torch.device("cpu"):
        model = XVLA(config)
    # Load weights from safetensors
    try:
        from safetensors.torch import load_file
        import glob
        st_files = sorted(glob.glob(os.path.join(args.ckpt, "*.safetensors")))
        state_dict = {}
        for f in st_files:
            state_dict.update(load_file(f))
        model.load_state_dict(state_dict, strict=False)
    except ImportError:
        # Fallback to torch loading
        model = XVLA.from_pretrained(args.ckpt, trust_remote_code=True, torch_dtype=torch.float32)
    model = model.eval().to(args.device)
    print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # --- Load processor ---
    # Try AutoProcessor first, fall back to manual construction
    try:
        processor = XVLAProcessor.from_pretrained(args.ckpt)
    except Exception:
        from transformers import AutoImageProcessor, AutoTokenizer
        print("  XVLAProcessor.from_pretrained failed, constructing manually...")
        image_processor = AutoImageProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
        processor = XVLAProcessor(image_processor=image_processor, tokenizer=tokenizer)

    tokenizer = processor.tokenizer
    # Florence2/BART tokenizer: <pad>=1, <s>=0, </s>=2 but transformers 5.x doesn't auto-set them
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 1  # <pad>
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(1)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = 2  # </s>
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(2)
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = 0  # <s>
        tokenizer.bos_token = tokenizer.convert_ids_to_tokens(0)
    pad_token_id = tokenizer.pad_token_id

    # --- Probe image token count (N_img) with a dummy image ---
    dummy_img = Image.new("RGB", (224, 224))
    dummy_inputs = processor(images=[[dummy_img]], language_instruction=["test"])
    dummy_inputs = {k: v.to(args.device) for k, v in dummy_inputs.items()}
    with torch.no_grad():
        dummy_enc = model.forward_vlm(
            dummy_inputs["input_ids"],
            dummy_inputs["image_input"],
            dummy_inputs["image_mask"],
        )
    T_enc = dummy_enc["vlm_features"].shape[1]
    L = dummy_inputs["input_ids"].shape[1]  # language_max_length (50)
    N_img = T_enc - L
    D = dummy_enc["vlm_features"].shape[2]
    print(f"  N_img={N_img}, L_text={L}, D={D}, T_enc={T_enc}")
    print(f"  Samples to process: {num_samples}")
    del dummy_inputs, dummy_enc

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
            # Also load img/txt partials if they exist
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

        # Preprocess: single image (1 view), single text instruction
        inputs = processor(images=[[img]], language_instruction=[task])
        input_ids = inputs["input_ids"].to(args.device)
        image_input = inputs["image_input"].to(args.device)
        image_mask = inputs["image_mask"].to(args.device)

        with torch.inference_mode():
            enc = model.forward_vlm(input_ids, image_input, image_mask)

        vlm_feat = enc["vlm_features"]  # [1, N_img + L, D]

        # Image tokens: first N_img positions (always valid)
        img_hidden = vlm_feat[:, :N_img, :]
        feat_i = img_hidden.float().mean(dim=1).squeeze(0).cpu()  # [D]

        # Text tokens: positions N_img onwards, mask out padding
        text_hidden = vlm_feat[:, N_img:, :]  # [1, L, D]
        text_ids = input_ids[0]  # [L]
        text_mask = (text_ids != pad_token_id).float()  # [L]
        feat_t = masked_mean_pool(text_hidden, text_mask.unsqueeze(0)).squeeze(0).cpu()  # [D]

        # Combined: image + non-padding text
        full_mask = torch.cat([
            torch.ones(1, N_img, device=args.device),
            text_mask.unsqueeze(0),
        ], dim=1)
        feat_it = masked_mean_pool(vlm_feat, full_mask).squeeze(0).cpu()  # [D]

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
        "model": args.ckpt,
        "extraction_point": "BART encoder output (forward_vlm → vlm_features)",
        "token_layout": f"[image({N_img}) | text({L} padded)]",
        "hidden_size": D,
        "N_img": N_img,
        "L_text": L,
        "pooling": {
            "feats_A": "mean-pool over image + non-padding text tokens",
            "feats_A_img": "mean-pool over image tokens only",
            "feats_A_txt": "mean-pool over non-padding text tokens only",
        },
        "num_samples": num_samples,
        "outputs": ["feats_A.pt", "feats_A_img.pt", "feats_A_txt.pt"],
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== X-VLA Phase 2 Complete ===")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()
