"""
Finding-5 extension — extract NEW standalone vision encoders for the
"vision-encoder paradigm" appendix in cknna_vla.

ROUND 1 (2026-05-10) — 7 encoders, n=11 cohort total:
  mae-base, mae-large, vc1-base, vc1-large, dinov2-base, dinov2-large, siglip2-base

ROUND 2 (2026-05-10) — 8 NEW encoders, n=19 cohort total:
  More sizes:    dinov2-small, dinov2-giant, mae-huge
  More SSL:      beit-base, dino-v1-base, ibot-base
  More V-L:      eva-clip-l14, openclip-l14-laion2b

Output dirs (under --output_base):
  <key>-standalone/feats_A_vit.pt   (mean-pooled features, fp32)

Pooling — match existing extractor convention (mean over PATCHES, skip CLS where present).

Image preprocessing — each model's NATIVE recipe.

Usage:
  /lambda/nfs/vla/conda/envs/starVLA/bin/python extract_finding5_features.py \
      --data_dir   /lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid \
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
from torchvision import transforms

# ---------------------------------------------------------------------------
# Cohort spec
# ---------------------------------------------------------------------------
MODELS = [
    # ===== ROUND 1 =====
    # ---- MAE (web pretraining) ----
    {"key": "mae-base-standalone",      "hf": "facebook/vit-mae-base",
     "loader": "hf", "model_class": "ViTMAEModel",      "dim": 768,  "skip_cls": True,  "batch_size": 256},
    {"key": "mae-large-standalone",     "hf": "facebook/vit-mae-large",
     "loader": "hf", "model_class": "ViTMAEModel",      "dim": 1024, "skip_cls": True,  "batch_size": 128},

    # ---- VC-1 (MAE on Ego4D + ImageNet + Manip) ----
    {"key": "vc1-base-standalone",      "hf": "facebook/vc1-base",
     "loader": "vc1_timm", "timm_arch": "vit_base_patch16_224",  "dim": 768,  "skip_cls": True, "batch_size": 256},
    {"key": "vc1-large-standalone",     "hf": "facebook/vc1-large",
     "loader": "vc1_timm", "timm_arch": "vit_large_patch16_224", "dim": 1024, "skip_cls": True, "batch_size": 128},

    # ---- DINOv2 (self-distillation SSL) ----
    {"key": "dinov2-base-standalone",   "hf": "facebook/dinov2-base",
     "loader": "hf", "model_class": "Dinov2Model",      "dim": 768,  "skip_cls": True,  "batch_size": 256},
    {"key": "dinov2-large-standalone",  "hf": "facebook/dinov2-large",
     "loader": "hf", "model_class": "Dinov2Model",      "dim": 1024, "skip_cls": True,  "batch_size": 128},

    # ---- SigLIP-2 base 224 (vision-language contrastive) ----
    {"key": "siglip2-base-standalone",  "hf": "google/siglip2-base-patch16-224",
     "loader": "hf", "model_class": "SiglipVisionModel","dim": 768,  "skip_cls": False, "batch_size": 256},

    # ===== ROUND 2 — size sweeps =====
    {"key": "dinov2-small-standalone",  "hf": "facebook/dinov2-small",
     "loader": "hf", "model_class": "Dinov2Model",      "dim": 384,  "skip_cls": True,  "batch_size": 256},
    {"key": "dinov2-giant-standalone",  "hf": "facebook/dinov2-giant",
     "loader": "hf", "model_class": "Dinov2Model",      "dim": 1536, "skip_cls": True,  "batch_size": 64},
    {"key": "mae-huge-standalone",      "hf": "facebook/vit-mae-huge",
     "loader": "hf", "model_class": "ViTMAEModel",      "dim": 1280, "skip_cls": True,  "batch_size": 64},

    # ===== ROUND 2 — alternative SSL paradigms =====
    {"key": "beit-base-standalone",     "hf": "microsoft/beit-base-patch16-224-pt22k",
     "loader": "hf", "model_class": "BeitModel",        "dim": 768,  "skip_cls": True,  "batch_size": 256},
    {"key": "dino-v1-base-standalone",  "hf": "facebook/dino-vitb16",
     "loader": "hf", "model_class": "ViTModel",         "dim": 768,  "skip_cls": True,  "batch_size": 256},
    {"key": "ibot-base-standalone",
     "ibot_url": "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth",
     "loader": "ibot_url", "timm_arch": "vit_base_patch16_224",  "dim": 768,  "skip_cls": True, "batch_size": 256},

    # ===== ROUND 2 — alternative V-L contrastive =====
    {"key": "eva-clip-l14-standalone",  "hf": "eva02_large_patch14_clip_224.merged2b",
     "loader": "timm_clip", "timm_arch": "eva02_large_patch14_clip_224.merged2b",
     "dim": 1024, "skip_cls": True, "batch_size": 128},
    {"key": "openclip-l14-laion2b-standalone", "hf": "vit_large_patch14_clip_224.laion2b",
     "loader": "timm_clip", "timm_arch": "vit_large_patch14_clip_224.laion2b",
     "dim": 1024, "skip_cls": True, "batch_size": 128},

    # ===== ROUND 3 (2026-05-13) — Darcet et al. ICLR 2024 "Vision Transformers
    # Need Registers" fix variants. Same DINOv2 training (LVD-142M, self-distill)
    # plus 4 register tokens that absorb high-norm artifact patches. Layout at
    # 224x224: [CLS @ idx 0, registers @ idx 1-4, patches @ idx 5-260]. We skip
    # all 5 non-patch tokens before mean-pool (num_skip_tokens=5).
    {"key": "dinov2-with-registers-small-standalone",  "hf": "facebook/dinov2-with-registers-small",
     "loader": "hf", "model_class": "Dinov2WithRegistersModel", "dim": 384,  "skip_cls": True, "num_skip_tokens": 5, "batch_size": 256},
    {"key": "dinov2-with-registers-base-standalone",   "hf": "facebook/dinov2-with-registers-base",
     "loader": "hf", "model_class": "Dinov2WithRegistersModel", "dim": 768,  "skip_cls": True, "num_skip_tokens": 5, "batch_size": 256},
    {"key": "dinov2-with-registers-large-standalone",  "hf": "facebook/dinov2-with-registers-large",
     "loader": "hf", "model_class": "Dinov2WithRegistersModel", "dim": 1024, "skip_cls": True, "num_skip_tokens": 5, "batch_size": 128},
    {"key": "dinov2-with-registers-giant-standalone",  "hf": "facebook/dinov2-with-registers-giant",
     "loader": "hf", "model_class": "Dinov2WithRegistersModel", "dim": 1536, "skip_cls": True, "num_skip_tokens": 5, "batch_size": 64},
]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def vc1_transform():
    """ImageNet-normalized 224 transform; matches both MAE and VC-1 paper preprocessing."""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class ImageFolderDataset(Dataset):
    def __init__(self, images_dir, num_samples, processor=None, tx=None):
        self.images_dir = images_dir
        self.num_samples = num_samples
        self.processor = processor
        self.tx = tx
        assert (processor is None) ^ (tx is None), "exactly one of processor/tx required"

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.images_dir, f"{idx:06d}.png")).convert("RGB")
        if self.processor is not None:
            return self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        return self.tx(img)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------
def load_hf(model_info, device):
    from transformers import (
        AutoImageProcessor,
        ViTMAEModel, Dinov2Model, SiglipVisionModel,
        BeitModel, ViTModel, Dinov2WithRegistersModel,
    )
    cls_map = {
        "ViTMAEModel": ViTMAEModel,
        "Dinov2Model": Dinov2Model,
        "Dinov2WithRegistersModel": Dinov2WithRegistersModel,
        "SiglipVisionModel": SiglipVisionModel,
        "BeitModel": BeitModel,
        "ViTModel": ViTModel,
    }
    M = cls_map[model_info["model_class"]]
    model = M.from_pretrained(model_info["hf"], torch_dtype=torch.float16).to(device).eval()
    # MAE encoder draws random patch masks unless config.mask_ratio == 0.
    if hasattr(model, "config") and hasattr(model.config, "mask_ratio"):
        model.config.mask_ratio = 0.0
    proc = AutoImageProcessor.from_pretrained(model_info["hf"], use_fast=True)
    return model, proc, None


def load_vc1_timm(model_info, device):
    """Load VC-1 from facebook/vc1-{base,large} into a timm ViT."""
    import timm
    from huggingface_hub import hf_hub_download
    ckpt = hf_hub_download(model_info["hf"], "pytorch_model.bin")
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = raw["model"] if "model" in raw else raw
    # Drop MAE decoder + mask_token keys; keep ViT encoder
    drop_prefixes = ("decoder", "mask_token")
    sd = {k: v for k, v in sd.items() if not any(k.startswith(p) for p in drop_prefixes)}
    # timm ViT expects no `norm.weight` etc. before fc — strip head if present
    sd = {k: v for k, v in sd.items() if not k.startswith("head.")}

    model = timm.create_model(model_info["timm_arch"], pretrained=False, num_classes=0)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  timm load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        # Print any non-trivial missing key (allow head.* missing — head dropped intentionally)
        nontrivial = [k for k in missing if not k.startswith("head.") and k != "norm.weight" and k != "norm.bias"]
        if nontrivial:
            print(f"    nontrivial missing: {nontrivial[:5]}{'...' if len(nontrivial) > 5 else ''}")
    if unexpected:
        print(f"    unexpected: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model = model.to(device).half().eval()
    tx = vc1_transform()
    return model, None, tx


def load_timm_clip(model_info, device):
    """Load timm-native pretrained checkpoint (EVA-CLIP / OpenCLIP variants)."""
    import timm
    model = timm.create_model(
        model_info["timm_arch"], pretrained=True, num_classes=0
    )
    print(f"  timm pretrained loaded: {model_info['timm_arch']}, "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Use the model's official preprocessing pipeline.
    cfg = timm.data.resolve_data_config({}, model=model)
    tx = timm.data.create_transform(**cfg, is_training=False)
    print(f"  timm transform: input_size={cfg.get('input_size')} mean={cfg.get('mean')} std={cfg.get('std')}")

    model = model.to(device).half().eval()
    return model, None, tx


def load_ibot_url(model_info, device):
    """Download iBOT checkpoint from URL and load encoder weights into timm ViT."""
    import timm
    import urllib.request
    cache_dir = "/lambda/nfs/vla/cache/ibot_ckpts"
    os.makedirs(cache_dir, exist_ok=True)
    url = model_info["ibot_url"]
    fname = url.rsplit("/", 1)[-1]
    ckpt_path = os.path.join(cache_dir, fname)
    if not os.path.exists(ckpt_path):
        print(f"  Downloading iBOT ckpt: {url}")
        urllib.request.urlretrieve(url, ckpt_path)
    print(f"  iBOT ckpt: {ckpt_path} ({os.path.getsize(ckpt_path)/1e6:.1f} MB)")

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # iBOT teacher checkpoint typically has 'state_dict' or 'teacher' top-level
    if "teacher" in raw:
        sd = raw["teacher"]
    elif "state_dict" in raw:
        sd = raw["state_dict"]
    elif "model" in raw:
        sd = raw["model"]
    else:
        sd = raw

    # Strip "module." / "backbone." prefix if present (DINO/iBOT convention)
    new_sd = {}
    for k, v in sd.items():
        nk = k
        for prefix in ("module.", "backbone.", "encoder."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        # Drop iBOT projection head + DINO last-layer
        if nk.startswith("head.") or nk.startswith("dino_head."):
            continue
        new_sd[nk] = v

    model = timm.create_model(model_info["timm_arch"], pretrained=False, num_classes=0)
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"  iBOT load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        nontrivial = [k for k in missing if not k.startswith("head.") and k != "norm.weight" and k != "norm.bias"]
        if nontrivial:
            print(f"    nontrivial missing: {nontrivial[:8]}")
    if unexpected:
        print(f"    unexpected: {unexpected[:8]}")

    model = model.to(device).half().eval()
    tx = vc1_transform()  # iBOT uses standard ImageNet preprocessing (256→224)
    return model, None, tx


# ---------------------------------------------------------------------------
# Forward pass + pooling
# ---------------------------------------------------------------------------
def forward_pool(model_info, model, batch_pixels):
    """Run forward pass, return mean-pooled feature (B, D)."""
    if model_info["loader"] == "hf":
        outputs = model(pixel_values=batch_pixels)
        hidden = outputs.last_hidden_state.float()  # (B, T, D)
    elif model_info["loader"] in ("vc1_timm", "timm_clip", "ibot_url"):
        # timm ViT: forward_features returns (B, T, D) including CLS at idx 0
        hidden = model.forward_features(batch_pixels).float()
    else:
        raise ValueError(model_info["loader"])

    # Number of leading non-patch tokens to skip before mean-pool. For most
    # ViTs this is 1 (CLS). For DINOv2-with-registers it is 5 (CLS + 4
    # register tokens). Backwards-compatible default: 1 if skip_cls else 0.
    n_skip = model_info.get(
        "num_skip_tokens",
        1 if model_info.get("skip_cls", False) else 0,
    )
    if n_skip > 0:
        hidden = hidden[:, n_skip:, :]
    return hidden.mean(dim=1)  # (B, D)


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------
def extract_one(model_info, model, processor, tx, images_dir, num_samples,
                output_dir, device, num_workers=8):
    os.makedirs(output_dir, exist_ok=True)
    feats_path = os.path.join(output_dir, "feats_A_vit.pt")

    if os.path.exists(feats_path):
        existing = torch.load(feats_path, weights_only=True)
        if existing.shape[0] == num_samples and existing.shape[1] == model_info["dim"]:
            print(f"  Already complete: {feats_path} ({existing.shape})")
            return
        else:
            print(f"  Existing has wrong shape {existing.shape} - re-running")

    D = model_info["dim"]
    feats = torch.zeros(num_samples, D, dtype=torch.float32)
    ds = ImageFolderDataset(images_dir, num_samples, processor=processor, tx=tx)
    loader = DataLoader(ds, batch_size=model_info["batch_size"], num_workers=num_workers,
                        pin_memory=True, shuffle=False, drop_last=False)

    t0 = time.time()
    idx = 0
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for batch in loader:
            B = batch.shape[0]
            x = batch.to(device, non_blocking=True).half()
            feat = forward_pool(model_info, model, x)
            feats[idx:idx + B] = feat.cpu().float()
            idx += B
            if idx % (model_info["batch_size"] * 4) == 0 or idx == num_samples:
                elapsed = time.time() - t0
                rate = idx / elapsed
                eta = (num_samples - idx) / max(rate, 1e-6)
                print(f"  [{idx}/{num_samples}] {rate:.0f} img/s ETA {eta:.0f}s")

    assert idx == num_samples, f"Extracted {idx} != {num_samples}"
    assert feats.shape == (num_samples, D), f"Shape {feats.shape} != ({num_samples}, {D})"
    assert torch.isfinite(feats).all(), "non-finite values in feats"

    torch.save(feats, feats_path)
    meta = {
        "model_key": model_info["key"],
        "hf_id": model_info.get("hf", model_info.get("ibot_url", "?")),
        "loader": model_info["loader"],
        "dim": D,
        "skip_cls": model_info["skip_cls"],
        "num_samples": num_samples,
        "batch_size": model_info["batch_size"],
        "pooling": "mean over patches (skip CLS)" if model_info["skip_cls"]
                   else "mean over all tokens",
        "dtype": "float16 inference, float32 output",
    }
    with open(os.path.join(output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"  Done: {feats_path} shape={feats.shape} in {elapsed:.0f}s ({num_samples/elapsed:.0f} img/s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_base", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--models", nargs="*", default=None,
                    help="Specific model keys to extract (default: all)")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Cap N for smoke testing (default: full N from metadata)")
    args = ap.parse_args()

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        meta = json.load(f)
    N = meta["num_samples"]
    if args.max_samples is not None:
        N = min(N, args.max_samples)
    images_dir = os.path.join(args.data_dir, "images")
    print(f"Data: {N} images in {images_dir}")
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    to_run = MODELS
    if args.models:
        to_run = [m for m in MODELS if m["key"] in args.models]
    print(f"Will extract {len(to_run)} models: {[m['key'] for m in to_run]}")

    total_t0 = time.time()
    for model_info in to_run:
        src = model_info.get("hf", model_info.get("ibot_url", "?"))
        print(f"\n{'='*70}\nModel: {model_info['key']}  ({src})  "
              f"dim={model_info['dim']} bs={model_info['batch_size']}\n{'='*70}")
        if model_info["loader"] == "hf":
            model, proc, tx = load_hf(model_info, args.device)
        elif model_info["loader"] == "vc1_timm":
            model, proc, tx = load_vc1_timm(model_info, args.device)
        elif model_info["loader"] == "timm_clip":
            model, proc, tx = load_timm_clip(model_info, args.device)
        elif model_info["loader"] == "ibot_url":
            model, proc, tx = load_ibot_url(model_info, args.device)
        else:
            raise ValueError(model_info["loader"])

        out_dir = os.path.join(args.output_base, model_info["key"])
        extract_one(model_info, model, proc, tx, images_dir, N, out_dir,
                    args.device, num_workers=args.num_workers)
        del model
        torch.cuda.empty_cache()

    total_elapsed = time.time() - total_t0
    print(f"\nAll done in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
