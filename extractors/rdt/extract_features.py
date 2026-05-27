"""
Phase 2: Extract features from RDT-1B's vision encoder (SigLIP) for CKNNA.

RDT-1B uses two separate frozen encoders:
  - Vision: SigLIP SO400M (siglip-so400m-patch14-384), 1152D, 729 patches/image
  - Language: T5-v1.1-XXL (google/t5-v1_1-xxl), 4096D, pre-computed offline

These encoders NEVER interact directly — they are fused only inside the
DiT backbone via alternating cross-attention (lang at even layers, img at odd).

Extraction point:
  SigLIP vision tower last_hidden_state, mean-pooled over 729 patches.
  This is the frozen image representation used by RDT before any
  task-specific processing. Shape: (1152,)

  This is analogous to extracting pre-transformer features from RT-1-X
  (EfficientNet → 512D) — both capture what the vision encoder "sees"
  before any fusion with language or action conditioning.

RoboTwin numbers: Easy 34.50%, Hard 13.72% (official leaderboard, per-task FT).
Base model: robotics-diffusion-transformer/rdt-1b on HuggingFace.

Usage:
    python extract_features.py \
        --data_dir /path/to/cknna_data_exp1v2_full_L \
        --output_dir /path/to/cknna_data_exp1v2_full_L/rdt-1b-siglip
"""

import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers import SiglipImageProcessor, SiglipVisionModel


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Extract RDT-1B (SigLIP) features for CKNNA.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Phase 1 data directory (images/, metadata.json)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save feats_A.pt")
    parser.add_argument("--model_name", type=str,
                        default="google/siglip-so400m-patch14-384",
                        help="SigLIP model name on HuggingFace")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume_from", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (SigLIP is ~0.9B params)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    images_dir = os.path.join(args.data_dir, "images")

    # Load SigLIP vision tower (frozen, same as used in RDT-1B)
    print(f"Loading SigLIP: {args.model_name}")
    processor = SiglipImageProcessor.from_pretrained(args.model_name)
    model = SiglipVisionModel.from_pretrained(
        args.model_name, torch_dtype=torch.float16
    ).eval().to(args.device)

    hidden_size = model.config.hidden_size  # 1152
    num_patches = (model.config.image_size // model.config.patch_size) ** 2  # 729
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_patches: {num_patches}")
    print(f"  image_size: {model.config.image_size}")
    print(f"  Samples to process: {num_samples}")

    # Resume support
    feats_path = os.path.join(args.output_dir, "feats_A.pt")
    if args.resume_from > 0 and os.path.exists(feats_path):
        all_feats = torch.load(feats_path, weights_only=True)
        print(f"  Resuming from sample {args.resume_from}, loaded {all_feats.shape}")
    else:
        all_feats = torch.zeros(num_samples, hidden_size, dtype=torch.float32)
        args.resume_from = 0

    t0 = time.time()
    batch_imgs = []
    batch_indices = []

    with torch.no_grad():
        for i in range(args.resume_from, num_samples):
            img_path = os.path.join(images_dir, f"{i:06d}.png")
            img = Image.open(img_path).convert("RGB")
            batch_imgs.append(img)
            batch_indices.append(i)

            # Process batch
            if len(batch_imgs) == args.batch_size or i == num_samples - 1:
                inputs = processor(images=batch_imgs, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(
                    device=args.device, dtype=torch.float16
                )

                outputs = model(pixel_values)
                # last_hidden_state: (B, num_patches, hidden_size)
                features = outputs.last_hidden_state.float()
                # Mean-pool over patches → (B, hidden_size)
                features = features.mean(dim=1).cpu()

                for j, idx in enumerate(batch_indices):
                    all_feats[idx] = features[j]

                batch_imgs = []
                batch_indices = []

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1 - args.resume_from) / elapsed
                print(f"  [{i+1}/{num_samples}]  {rate:.1f} samples/s  "
                      f"elapsed: {elapsed:.0f}s")

            if (i + 1) % args.save_every == 0:
                torch.save(all_feats, feats_path)

    torch.save(all_feats, feats_path)

    elapsed = time.time() - t0
    print(f"\nDone. {num_samples} samples in {elapsed:.1f}s "
          f"({num_samples/elapsed:.1f} samples/s)")
    print(f"  feats_A shape: {all_feats.shape}")
    print(f"  feats_A stats: mean={all_feats.mean():.4f}, std={all_feats.std():.4f}")
    print(f"  Saved to: {feats_path}")

    # Save metadata
    meta = {
        "model": "rdt-1b-siglip",
        "extraction_point": "SigLIP last_hidden_state mean-pooled over 729 patches",
        "vision_encoder": args.model_name,
        "hidden_size": hidden_size,
        "num_patches": num_patches,
        "note": "RDT-1B uses SigLIP (frozen) + T5-XXL (frozen) as separate encoders, "
                "fused only in DiT via alternating cross-attention. We extract the frozen "
                "SigLIP vision features as the base image representation. "
                "RoboTwin: Easy 34.50%, Hard 13.72% (official leaderboard).",
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved.")


if __name__ == "__main__":
    main()
