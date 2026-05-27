"""
Phase 2: Extract features from ImageNet-pretrained ResNet18 for CKNNA.

This serves as the base vision encoder for ACT (Action Chunking Transformer)
and DP (Diffusion Policy) which both use ResNet18 as their vision backbone
on the RoboTwin benchmark.

Architecture:
  Image (224x224) -> ResNet18 (ImageNet-pretrained) -> layer4 -> AdaptiveAvgPool2d -> (512,)

Extraction point:
  Output of the final residual block (layer4), after adaptive average pooling.
  This is the standard "feature vector" used by downstream heads in ACT/DP.
  Shape: (512,)

Both ACT and DP use this same frozen/pretrained ResNet18 backbone, so they
share identical base features. ACT fine-tunes with lr=1e-5 while DP uses
random init in RoboTwin config, but we extract from the ImageNet-pretrained
base for cross-dataset CKNNA comparison.

Usage:
    python extract_features.py \
        --data_dir /path/to/cknna_data_exp1v2_full_L \
        --output_dir /path/to/cknna_data_exp1v2_full_L/resnet18-imagenet
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Extract ResNet18 features for CKNNA.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Phase 1 data directory (images/, metadata.json)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save feats_A.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume_from", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for extraction (ResNet18 is lightweight)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    images_dir = os.path.join(args.data_dir, "images")

    # Load ImageNet-pretrained ResNet18 as feature extractor
    print("Loading ImageNet-pretrained ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Remove the final FC layer, keep avgpool output (512,)
    model.fc = nn.Identity()
    model = model.eval().to(args.device)

    hidden_size = 512
    print(f"  Feature dim: {hidden_size}")
    print(f"  Samples to process: {num_samples}")

    # Standard ImageNet preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

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
            img_tensor = preprocess(img)
            batch_imgs.append(img_tensor)
            batch_indices.append(i)

            # Process batch
            if len(batch_imgs) == args.batch_size or i == num_samples - 1:
                batch = torch.stack(batch_imgs).to(args.device)
                features = model(batch)  # (B, 512)
                features = features.float().cpu()

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
        "model": "resnet18-imagenet",
        "extraction_point": "layer4 -> adaptive_avg_pool -> fc(Identity)",
        "hidden_size": hidden_size,
        "weights": "IMAGENET1K_V1",
        "preprocessing": "Resize(256) -> CenterCrop(224) -> ImageNet normalize",
        "num_samples": num_samples,
        "note": "Base vision encoder for ACT (pretrained=True) and DP (random init in RT config). "
                "We use ImageNet-pretrained weights as the cross-dataset base representation.",
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved.")


if __name__ == "__main__":
    main()
