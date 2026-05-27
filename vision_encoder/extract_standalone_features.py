"""
Phase 1: Extract features from standalone vision encoders for CKNNA.

Extracts mean-pooled last_hidden_state from 4 standalone ViTs on DROID images.
Output: feats_A_vit.pt per model, matching P1-V convention from VLM extractors.

Pooling conventions (verified against existing VLM extraction scripts):
  - CLIP: skip CLS (idx 0), mean over patches 1..256 → (1024,)
    (matches extract_vit_levels_kosmos2.py line 88)
  - SigLIP: mean over all patches (no CLS)
    (matches extract_vit_levels_paligemma.py line 63)

Optimized for 40GB GPU: fp16 inference, large batch sizes, multi-worker data loading.

Usage:
    conda run -n starVLA python extract_standalone_features.py \
        --data_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid \
        --output_base /lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid \
        --device cuda
"""

import argparse
import json
import os
import time

import torch
torch.backends.cudnn.enabled = False
from torch.utils.data import Dataset, DataLoader
from PIL import Image

MODELS = [
    {
        "key": "clip-vit-l14-standalone",
        "path": "clip_vit_l14",
        "family": "clip",
        "model_class": "CLIPVisionModel",
        "processor_class": "CLIPImageProcessor",
        "dim": 1024,
        "skip_cls": True,
        "batch_size": 256,  # 224px, ~1.2GB model → plenty of room
    },
    {
        "key": "siglip-so400m-standalone",
        "path": "siglip_so400m_patch14",
        "family": "siglip",
        "model_class": "SiglipVisionModel",
        "processor_class": "SiglipImageProcessor",
        "dim": 1152,
        "skip_cls": False,
        "batch_size": 256,  # 224px, ~1.6GB model
    },
    {
        "key": "siglip2-large-standalone",
        "path": "siglip2_large_patch16",
        "family": "siglip",
        "model_class": "SiglipVisionModel",
        "processor_class": "SiglipImageProcessor",
        "dim": 1024,
        "skip_cls": False,
        "batch_size": 128,  # 384px → 576 patches, more memory per image
    },
    {
        "key": "siglip2-so400m-standalone",
        "path": "siglip2_so400m_patch16",
        "family": "siglip",
        "model_class": "SiglipVisionModel",
        "processor_class": "SiglipImageProcessor",
        "dim": 1152,
        "skip_cls": False,
        "batch_size": 128,  # 384px, ~1.6GB model
    },
]


class ImageFolderDataset(Dataset):
    """Simple dataset that loads images by index from a directory of {NNNNNN}.png files."""

    def __init__(self, images_dir, num_samples, processor):
        self.images_dir = images_dir
        self.num_samples = num_samples
        self.processor = processor

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.images_dir, f"{idx:06d}.png")).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)  # (C, H, W)


def load_model_and_processor(model_info, vision_encoder_dir, device):
    from transformers import (
        CLIPVisionModel, CLIPImageProcessor,
        SiglipVisionModel, SiglipImageProcessor,
    )
    cls_map = {
        "CLIPVisionModel": CLIPVisionModel,
        "CLIPImageProcessor": CLIPImageProcessor,
        "SiglipVisionModel": SiglipVisionModel,
        "SiglipImageProcessor": SiglipImageProcessor,
    }
    model_path = os.path.join(vision_encoder_dir, model_info["path"])
    model = cls_map[model_info["model_class"]].from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to(device).eval()
    processor = cls_map[model_info["processor_class"]].from_pretrained(model_path)
    return model, processor


def extract_one_model(model_info, model, processor, images_dir, num_samples,
                      output_dir, device, num_workers=8):
    os.makedirs(output_dir, exist_ok=True)
    feats_path = os.path.join(output_dir, "feats_A_vit.pt")

    # Check if already complete
    if os.path.exists(feats_path):
        existing = torch.load(feats_path, weights_only=True)
        if existing.shape[0] == num_samples:
            print(f"  Already complete: {feats_path} ({existing.shape})")
            return

    # Pre-allocate output tensor
    D = model_info["dim"]
    feats = torch.zeros(num_samples, D, dtype=torch.float32)
    skip_cls = model_info["skip_cls"]
    batch_size = model_info["batch_size"]

    dataset = ImageFolderDataset(images_dir, num_samples, processor)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=True, shuffle=False, drop_last=False)

    t0 = time.time()
    idx = 0

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for batch_pixels in loader:
            B = batch_pixels.shape[0]
            pixel_values = batch_pixels.to(device, non_blocking=True)

            outputs = model(pixel_values=pixel_values)
            hidden = outputs.last_hidden_state.float()  # (B, num_patches, D)

            if skip_cls:
                hidden = hidden[:, 1:, :]  # skip CLS token

            feat = hidden.mean(dim=1)  # (B, D)
            feats[idx:idx + B] = feat.cpu()
            idx += B

            if idx % (batch_size * 4) == 0 or idx == num_samples:
                elapsed = time.time() - t0
                rate = idx / elapsed
                eta = (num_samples - idx) / rate if rate > 0 else 0
                print(f"  [{idx}/{num_samples}] {rate:.0f} img/s, ETA {eta:.0f}s")

    assert idx == num_samples, f"Extracted {idx} != {num_samples}"
    assert feats.shape == (num_samples, D), f"Shape {feats.shape} != ({num_samples}, {D})"

    torch.save(feats, feats_path)

    meta = {
        "model_key": model_info["key"],
        "model_path": model_info["path"],
        "model_class": model_info["model_class"],
        "dim": D,
        "skip_cls": skip_cls,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "pooling": "mean over patches (skip CLS)" if skip_cls else "mean over all patches",
        "dtype": "float16 inference, float32 output",
    }
    with open(os.path.join(output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"  Done: {feats_path} shape={feats.shape} in {elapsed:.1f}s ({num_samples/elapsed:.0f} img/s)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract standalone vision encoder features for CKNNA")
    parser.add_argument("--data_dir", required=True,
                        help="DROID data dir (images/, metadata.json)")
    parser.add_argument("--output_base", required=True,
                        help="Base dir for output (model subdirs created here)")
    parser.add_argument("--vision_encoder_dir", default=None,
                        help="Dir with downloaded checkpoints (default: script's parent dir)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--models", nargs="*", default=None,
                        help="Specific model keys to extract (default: all 4)")
    args = parser.parse_args()

    if args.vision_encoder_dir is None:
        args.vision_encoder_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    images_dir = os.path.join(args.data_dir, "images")
    print(f"Data: {num_samples} images in {images_dir}")
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    models_to_run = MODELS
    if args.models:
        models_to_run = [m for m in MODELS if m["key"] in args.models]

    total_t0 = time.time()
    for model_info in models_to_run:
        print(f"\n{'='*60}")
        print(f"Model: {model_info['key']} ({model_info['dim']}D, "
              f"{'skip CLS' if model_info['skip_cls'] else 'all patches'}, "
              f"bs={model_info['batch_size']})")
        print(f"{'='*60}")

        model, processor = load_model_and_processor(
            model_info, args.vision_encoder_dir, args.device)
        output_dir = os.path.join(args.output_base, model_info["key"])

        extract_one_model(
            model_info, model, processor, images_dir, num_samples,
            output_dir, args.device, num_workers=args.num_workers)

        del model, processor
        torch.cuda.empty_cache()

    total_elapsed = time.time() - total_t0
    print(f"\nAll done in {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
