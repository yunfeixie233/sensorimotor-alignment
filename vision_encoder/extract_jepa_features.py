"""
extract_jepa_features.py — Stage 1 of the §F "Pretraining Paradigm" JEPA extension.
See notes/tabs/2026-05-16_vjepa-paradigm-extension-plan.md.

Extracts frozen, mean-pooled features for 8 JEPA-family encoders on DROID, in the
SAME format as extract_finding5_features.py so compute_cknna.py reads them
unchanged:  <output_base>/<dir>/feats_A_vit.pt   shape (N, D) float32.

Encoders (all PRETRAINED SSL encoder backbones — never fine-tuned heads):
  ijepa-vith14     I-JEPA ViT-H/14   image   HF IJepaModel
  vjepa2-vit{l,h,g} V-JEPA 2         video   HF VJEPA2Model
  vjepa-v1-vit{l,h} V-JEPA v1        video   facebookresearch/jepa repo
  vjepa2-1-vit{l,g} V-JEPA 2.1       video   facebookresearch/vjepa2 repo

Two input settings:
  --setting 1  tiled still image -> 16-frame static clip (I-JEPA: single image).
               N=6000, data_dir = cknna_data_droid           (images/{i:06d}.png)
  --setting 2  real 16-frame past clip @ stride 4 (4.0 s window @ 15 fps).
               N=4890, data_dir = cknna_data_droid_multiframe (images/{i:06d}/{f:03d}.png)

Output dir name: setting 1 -> <key>-standalone ; setting 2 -> <key>-realclip[-lasttube]

The jepa and vjepa2 repos both ship top-level `src`/`app` packages, so V-JEPA v1
and V-JEPA 2.1 must run in SEPARATE invocations (asserted in main).

dtype: fp32 weights + torch.autocast(fp16). V-JEPA 2.1 uses RoPE and breaks under
a hard model.half(); fp32-weights + autocast is correct for all four families.

Usage:
  python extract_jepa_features.py --setting 1 \
     --data_dir   /lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid \
     --output_base /lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid \
     --models ijepa-vith14 vjepa2-vitl vjepa2-vith vjepa2-vitg
"""
import argparse
import json
import os
import sys
import time

import torch

torch.backends.cudnn.enabled = False
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_FRAMES = 16
TUBELET = 2
S2_STRIDE = 4
# stride-4 last-16 frames of the 65-frame (000..064) cache -> [4,8,...,64], anchor=064
S2_FRAME_IDS = list(range(64 - S2_STRIDE * (NUM_FRAMES - 1), 65, S2_STRIDE))
assert len(S2_FRAME_IDS) == NUM_FRAMES and S2_FRAME_IDS[-1] == 64, S2_FRAME_IDS

JEPA_REPO = "/lambda/nfs/vla/cknna_project/repos/jepa"
VJEPA2_REPO = "/lambda/nfs/vla/cknna_project/repos/vjepa2"

MODELS = {
    "ijepa-vith14": dict(family="ijepa", hf="facebook/ijepa_vith14_1k",
                         dim=1280, video=False, batch_size=128),
    "vjepa2-vitl": dict(family="vjepa2_hf", hf="facebook/vjepa2-vitl-fpc64-256",
                        dim=1024, video=True, batch_size=32),
    "vjepa2-vith": dict(family="vjepa2_hf", hf="facebook/vjepa2-vith-fpc64-256",
                        dim=1280, video=True, batch_size=24),
    "vjepa2-vitg": dict(family="vjepa2_hf", hf="facebook/vjepa2-vitg-fpc64-256",
                        dim=1408, video=True, batch_size=16),
    "vjepa-v1-vitl": dict(family="vjepa_v1", arch="vit_large", res=224,
                          ckpt="/lambda/nfs/vla/cache/jepa_ckpts/vitl16.pth.tar",
                          ckpt_key="target_encoder", dim=1024, video=True, batch_size=32),
    "vjepa-v1-vith": dict(family="vjepa_v1", arch="vit_huge", res=224,
                          ckpt="/lambda/nfs/vla/cache/jepa_ckpts/vith16.pth.tar",
                          ckpt_key="target_encoder", dim=1280, video=True, batch_size=24),
    "vjepa2-1-vitl": dict(family="vjepa2_1", arch="vit_large", res=384,
                          ckpt="/lambda/nfs/vla/cache/vjepa2_ckpts/vjepa2_1_vitl_dist_vitG_384.pt",
                          ckpt_key="ema_encoder", dim=1024, video=True, batch_size=16),
    "vjepa2-1-vitg": dict(family="vjepa2_1", arch="vit_giant_xformers", res=384,
                          ckpt="/lambda/nfs/vla/cache/vjepa2_ckpts/vjepa2_1_vitg_384.pt",
                          ckpt_key="target_encoder", dim=1408, video=True, batch_size=8),
}

VJEPA_V1_KWARGS = dict(img_size=224, patch_size=16, num_frames=16, tubelet_size=2,
                       uniform_power=True, use_sdpa=True, use_SiLU=False, tight_SiLU=False)
VJEPA2_1_KWARGS = dict(patch_size=16, img_size=(384, 384), num_frames=64, tubelet_size=2,
                       use_sdpa=True, use_SiLU=False, wide_SiLU=True, uniform_power=False,
                       use_rope=True, img_temporal_dim_size=1, interpolate_rope=True)


# --------------------------------------------------------------------------- #
# Frame loading + preprocessing
# --------------------------------------------------------------------------- #
def load_frames(idx, setting, data_dir, video):
    if setting == 1:
        img = Image.open(os.path.join(data_dir, "images", f"{idx:06d}.png")).convert("RGB")
        return [img] * (NUM_FRAMES if video else 1)
    sub = os.path.join(data_dir, "images", f"{idx:06d}")
    return [Image.open(os.path.join(sub, f"{f:03d}.png")).convert("RGB") for f in S2_FRAME_IDS]


def torchvision_tx(res):
    return transforms.Compose([
        transforms.Resize(round(res * 256 / 224),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(res),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class JepaDataset(Dataset):
    """Yields one model-ready input tensor per sample."""

    def __init__(self, n, setting, data_dir, spec, hf_proc=None, tx=None):
        self.n, self.setting, self.data_dir = n, setting, data_dir
        self.spec, self.hf_proc, self.tx = spec, hf_proc, tx

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        frames = load_frames(idx, self.setting, self.data_dir, self.spec["video"])
        fam = self.spec["family"]
        if fam == "ijepa":
            return self.hf_proc(images=frames[0], return_tensors="pt")["pixel_values"][0]
        if fam == "vjepa2_hf":
            return self.hf_proc(videos=[frames], return_tensors="pt")["pixel_values_videos"][0]
        # vjepa_v1 / vjepa2_1 -> (C, T, H, W)
        clip = torch.stack([self.tx(f) for f in frames])      # (T, C, H, W)
        return clip.permute(1, 0, 2, 3).contiguous()          # (C, T, H, W)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def _load_ckpt(model, path, key):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck[key]
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    print(f"  ckpt['{key}']: missing={len(msg.missing_keys)} "
          f"unexpected={len(msg.unexpected_keys)}", flush=True)
    assert len(msg.missing_keys) == 0, f"missing keys: {msg.missing_keys[:5]}"


def load_model(spec, device):
    fam = spec["family"]
    if fam == "ijepa":
        from transformers import AutoImageProcessor, IJepaModel
        model = IJepaModel.from_pretrained(spec["hf"]).to(device).eval()
        return model, AutoImageProcessor.from_pretrained(spec["hf"], use_fast=True), None
    if fam == "vjepa2_hf":
        from transformers import AutoVideoProcessor, VJEPA2Model
        model = VJEPA2Model.from_pretrained(spec["hf"]).to(device).eval()
        return model, AutoVideoProcessor.from_pretrained(spec["hf"]), None
    if fam == "vjepa_v1":
        sys.path.insert(0, JEPA_REPO)
        import src.models.vision_transformer as vit
        model = vit.__dict__[spec["arch"]](**VJEPA_V1_KWARGS)
        _load_ckpt(model, spec["ckpt"], spec["ckpt_key"])
        return model.to(device).eval(), None, torchvision_tx(spec["res"])
    if fam == "vjepa2_1":
        sys.path.insert(0, VJEPA2_REPO)
        from app.vjepa_2_1.models import vision_transformer as vit
        model = vit.__dict__[spec["arch"]](**VJEPA2_1_KWARGS)
        _load_ckpt(model, spec["ckpt"], spec["ckpt_key"])
        return model.to(device).eval(), None, torchvision_tx(spec["res"])
    raise ValueError(fam)


def forward_hidden(spec, model, batch):
    fam = spec["family"]
    if fam == "ijepa":
        return model(pixel_values=batch).last_hidden_state
    if fam == "vjepa2_hf":
        return model(pixel_values_videos=batch).last_hidden_state
    out = model(batch)                                        # (B, C, T, H, W) -> tokens
    return out[0] if isinstance(out, (tuple, list)) else out


def pool(hidden, mode, video):
    """meanall: mean over all tokens. lasttube: mean over the last temporal group.
    maxpool: max over all tokens."""
    hidden = hidden.float()
    if mode == "lasttube" and video:
        n_temporal = NUM_FRAMES // TUBELET                    # 8
        spatial = hidden.shape[1] // n_temporal
        return hidden[:, -spatial:, :].mean(dim=1)
    if mode == "maxpool":
        return hidden.amax(dim=1)
    return hidden.mean(dim=1)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def out_dirname(key, setting, pool_mode):
    base = f"{key}-standalone" if setting == 1 else f"{key}-realclip"
    if pool_mode != "meanall":
        base += f"-{pool_mode}"
    return base


def extract_one(key, spec, setting, pool_mode, data_dir, output_base, N, device, num_workers):
    out_dir = os.path.join(output_base, out_dirname(key, setting, pool_mode))
    os.makedirs(out_dir, exist_ok=True)
    feats_path = os.path.join(out_dir, "feats_A_vit.pt")
    D = spec["dim"]
    if os.path.exists(feats_path):
        ex = torch.load(feats_path, weights_only=True)
        if tuple(ex.shape) == (N, D):
            print(f"  already complete: {feats_path} {tuple(ex.shape)}", flush=True)
            return
        print(f"  existing shape {tuple(ex.shape)} != ({N},{D}) - re-running", flush=True)

    model, hf_proc, tx = load_model(spec, device)
    ds = JepaDataset(N, setting, data_dir, spec, hf_proc=hf_proc, tx=tx)
    loader = DataLoader(ds, batch_size=spec["batch_size"], num_workers=num_workers,
                        pin_memory=True, shuffle=False, drop_last=False)

    feats = torch.zeros(N, D, dtype=torch.float32)
    t0, idx, last_print = time.time(), 0, time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for batch in loader:
            B = batch.shape[0]
            hidden = forward_hidden(spec, model, batch.to(device, non_blocking=True))
            feats[idx:idx + B] = pool(hidden, pool_mode, spec["video"]).cpu().float()
            idx += B
            now = time.time()
            if now - last_print > 60 or idx == N:
                el = now - t0
                print(f"  [{idx}/{N}] {idx / el:.1f} samp/s  ETA {(N - idx) / max(idx / el, 1e-6):.0f}s",
                      flush=True)
                last_print = now

    assert idx == N and feats.shape == (N, D), f"{idx}!={N} or {feats.shape}"
    assert torch.isfinite(feats).all(), "non-finite values"
    torch.save(feats, feats_path)
    with open(os.path.join(out_dir, "extraction_metadata.json"), "w") as f:
        json.dump(dict(key=key, family=spec["family"], setting=setting, pool=pool_mode,
                       dim=D, num_samples=N, num_frames=NUM_FRAMES,
                       s2_frame_ids=S2_FRAME_IDS if setting == 2 else None,
                       data_dir=data_dir), f, indent=2)
    print(f"  DONE {feats_path}  shape={tuple(feats.shape)}  "
          f"{time.time() - t0:.0f}s ({N / (time.time() - t0):.1f} samp/s)", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", type=int, required=True, choices=[1, 2])
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_base", required=True)
    ap.add_argument("--models", nargs="+", required=True, choices=list(MODELS))
    ap.add_argument("--pool", default="meanall", choices=["meanall", "lasttube", "maxpool"])
    ap.add_argument("--max_samples", type=int, default=None, help="cap N for smoke tests")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    families = {MODELS[k]["family"] for k in args.models}
    assert not ({"vjepa_v1", "vjepa2_1"} <= families), \
        "Run V-JEPA v1 and V-JEPA 2.1 in separate invocations (repo package collision)."

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        N = json.load(f)["num_samples"]
    if args.max_samples:
        N = min(N, args.max_samples)
    print(f"setting={args.setting}  N={N}  pool={args.pool}  data_dir={args.data_dir}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    t0 = time.time()
    failed = []
    for key in args.models:
        spec = MODELS[key]
        if args.setting == 2 and not spec["video"]:
            print(f"\n=== {key}: SKIP (image-only model, no Setting 2) ===", flush=True)
            continue
        print(f"\n{'=' * 64}\n{key}  ({spec['family']}, dim={spec['dim']}, "
              f"bs={spec['batch_size']})\n{'=' * 64}", flush=True)
        try:
            extract_one(key, spec, args.setting, args.pool, args.data_dir,
                        args.output_base, N, args.device, args.num_workers)
        except Exception:
            import traceback
            print(f"  [FAIL] {key}", flush=True)
            traceback.print_exc()
            failed.append(key)
            torch.cuda.empty_cache()

    print(f"\nALL DONE in {(time.time() - t0) / 60:.1f} min"
          + (f"  [FAILED: {failed}]" if failed else "  [all ok]"), flush=True)


if __name__ == "__main__":
    main()
