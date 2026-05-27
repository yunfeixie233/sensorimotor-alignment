"""
Stage-0 smoke test — V-JEPA v1 (facebookresearch/jepa) + V-JEPA 2.1 (facebookresearch/vjepa2).
Run ONE family per invocation (the two repos both ship top-level `src`/`app`
packages, so loading both in one process collides):

  python smoke_test_jepa_repos.py --family vjepa_v1
  python smoke_test_jepa_repos.py --family vjepa_2_1

Verifies: repo code imports, model builds (no xformers needed), checkpoint
loads (if downloaded), forward pass returns patch tokens.
"""
import argparse
import os
import sys
import traceback

import torch

torch.backends.cudnn.enabled = False
DEV = "cuda"

JEPA_REPO = "/lambda/nfs/vla/cknna_project/repos/jepa"
VJEPA2_REPO = "/lambda/nfs/vla/cknna_project/repos/vjepa2"
JEPA_CKPT = "/lambda/nfs/vla/cache/jepa_ckpts"
VJEPA2_CKPT = "/lambda/nfs/vla/cache/vjepa2_ckpts"


def _clean(sd):
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}


def test_vjepa_v1():
    """V-JEPA v1: jepa repo. ViT-L/H @224. Config from configs/evals/vitl16_in1k.yaml."""
    sys.path.insert(0, JEPA_REPO)
    import src.models.vision_transformer as vit
    specs = [("vit_large", "vitl16.pth.tar"), ("vit_huge", "vith16.pth.tar")]
    for arch, fname in specs:
        print(f"\n--- V-JEPA v1 {arch} ---", flush=True)
        enc = vit.__dict__[arch](
            img_size=224, patch_size=16, num_frames=16, tubelet_size=2,
            uniform_power=True, use_sdpa=True, use_SiLU=False, tight_SiLU=False,
        )
        print(f"  built OK  embed_dim={enc.embed_dim} num_heads={enc.num_heads}", flush=True)
        ckpt = os.path.join(JEPA_CKPT, fname)
        if os.path.exists(ckpt) and os.path.getsize(ckpt) > 1e8:
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            key = "target_encoder" if "target_encoder" in ck else "encoder"
            msg = enc.load_state_dict(_clean(ck[key]), strict=False)
            print(f"  ckpt['{key}'] -> missing={len(msg.missing_keys)} "
                  f"unexpected={len(msg.unexpected_keys)}", flush=True)
            if msg.missing_keys:
                print(f"    missing eg: {msg.missing_keys[:3]}", flush=True)
            if msg.unexpected_keys:
                print(f"    unexpected eg: {msg.unexpected_keys[:3]}", flush=True)
        else:
            print(f"  ckpt not ready ({ckpt}) -- construction+forward only", flush=True)
        enc = enc.to(DEV).half().eval()
        clip = torch.randn(1, 3, 16, 224, 224, device=DEV, dtype=torch.half)  # (B,C,T,H,W)
        with torch.no_grad():
            out = enc(clip)
        out = out[0] if isinstance(out, (tuple, list)) else out
        print(f"  forward (B,C,T,H,W)={tuple(clip.shape)} -> {tuple(out.shape)}", flush=True)
        del enc
        torch.cuda.empty_cache()


def test_vjepa_2_1():
    """V-JEPA 2.1: vjepa2 repo, app/vjepa_2_1/models. ViT-L + ViT-g @384."""
    sys.path.insert(0, VJEPA2_REPO)
    from app.vjepa_2_1.models import vision_transformer as vit
    enc_kwargs = dict(
        patch_size=16, img_size=(384, 384), num_frames=64, tubelet_size=2,
        use_sdpa=True, use_SiLU=False, wide_SiLU=True, uniform_power=False,
        use_rope=True, img_temporal_dim_size=1, interpolate_rope=True,
    )
    specs = [
        ("vit_large", "vjepa2_1_vitl_dist_vitG_384.pt", "ema_encoder"),
        ("vit_giant_xformers", "vjepa2_1_vitg_384.pt", "target_encoder"),
    ]
    for arch, fname, key in specs:
        print(f"\n--- V-JEPA 2.1 {arch} ---", flush=True)
        enc = vit.__dict__[arch](**enc_kwargs)
        print(f"  built OK  embed_dim={enc.embed_dim}", flush=True)
        ckpt = os.path.join(VJEPA2_CKPT, fname)
        if os.path.exists(ckpt) and os.path.getsize(ckpt) > 1e8:
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            print(f"  ckpt top-level keys: {list(ck.keys())[:8]}", flush=True)
            use_key = key if key in ck else next(iter(ck))
            msg = enc.load_state_dict(_clean(ck[use_key]), strict=False)
            print(f"  ckpt['{use_key}'] -> missing={len(msg.missing_keys)} "
                  f"unexpected={len(msg.unexpected_keys)}", flush=True)
        else:
            print(f"  ckpt not ready ({ckpt}) -- construction+forward only", flush=True)
        enc = enc.to(DEV).half().eval()
        for shp in [(1, 3, 16, 384, 384), (1, 16, 3, 384, 384)]:
            try:
                clip = torch.randn(*shp, device=DEV, dtype=torch.half)
                with torch.no_grad():
                    out = enc(clip)
                out = out[0] if isinstance(out, (tuple, list)) else out
                print(f"  forward {shp} -> {tuple(out.shape)}  OK", flush=True)
                break
            except Exception as e:
                print(f"  forward {shp} failed: {str(e)[:120]}", flush=True)
        del enc
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["vjepa_v1", "vjepa_2_1"])
    args = ap.parse_args()
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}", flush=True)
    try:
        (test_vjepa_v1 if args.family == "vjepa_v1" else test_vjepa_2_1)()
        print("\nSMOKE DONE", flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
