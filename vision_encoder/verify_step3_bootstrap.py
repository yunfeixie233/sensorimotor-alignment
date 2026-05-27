"""Step 3 — paired subsampling bootstrap for CKNNA significance.

For B resamples: subsample M = 0.8*N_valid indices (no replacement), recompute
the DTW top-k @ h=7 on the subset AND the CKNNA for every encoder (production
code from compute_dtw.py / compute_cknna.py). Paired — all encoders share each
resample, so the resample noise cancels in the differences.

  --setting 1 : Setting-1 cohort (N=6000), JEPA vs DINOv2/iBOT/MAE
  --setting 2 : Setting-2 (N=4890), tiled vs real-clip per video encoder
"""
import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _os.path.join(_REPO_ROOT, "compute"))
import compute_dtw as CD          # noqa: E402
import compute_cknna as CC        # noqa: E402

H = 7
K = 10
DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", type=int, required=True, choices=[1, 2])
    ap.add_argument("--B", type=int, default=40)
    args = ap.parse_args()

    if args.setting == 1:
        DD = "/lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid"
        encoders = {n: f"{DD}/{d}-standalone/feats_A_vit.pt" for n, d in [
            ("dinov2-base", "dinov2-base"), ("ibot-base", "ibot-base"),
            ("siglip2-large", "siglip2-large"), ("clip-l14", "clip-vit-l14"),
            ("vjepa2-vitg", "vjepa2-vitg"), ("vjepa2-vith", "vjepa2-vith"),
            ("vjepa2-vitl", "vjepa2-vitl"), ("ijepa-vith14", "ijepa-vith14"),
            ("mae-base", "mae-base")]}
    else:
        MF = "/lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid_multiframe"
        DD = MF
        vids = ["vjepa2-vitl", "vjepa2-vith", "vjepa2-vitg",
                "vjepa-v1-vitl", "vjepa-v1-vith", "vjepa2-1-vitl", "vjepa2-1-vitg"]
        encoders = {}
        for v in vids:
            encoders[f"{v}__tiled"] = f"{MF}/{v}-tiled/feats_A_vit.pt"
            encoders[f"{v}__real"] = f"{MF}/{v}-realclip/feats_A_vit.pt"

    meta = json.load(open(f"{DD}/metadata.json"))
    valid = np.array([bool(t and t.strip()) for t in meta["task_descriptions"]])
    fB = torch.load(f"{DD}/feats_B_seq.pt", weights_only=True).float()[valid]   # (Nv,221,7)
    Nv = int(valid.sum())
    M = int(0.8 * Nv)
    print(f"setting={args.setting}  N_valid={Nv}  M={M}  B={args.B}  h={H}", flush=True)

    feats = {n: torch.load(p, weights_only=True).float()[valid] for n, p in encoders.items()}
    for n, f in feats.items():
        assert f.shape[0] == Nv, f"{n}: {f.shape[0]} != {Nv}"

    rng = np.random.default_rng(0)
    res = {n: [] for n in encoders}
    t0 = time.time()
    for b in range(args.B):
        sub = np.sort(rng.choice(Nv, size=M, replace=False))
        traj = fB[sub][:, 1:H + 1, :].numpy()                       # future [k+1..k+H]
        dtw_topk = CD.compute_dtw_topk_gpu(traj, K)                  # (M,K)
        mask_L = CC.build_mask_from_topk(dtw_topk, M, DEV)
        hsic_LL = CC.hsic_unbiased(mask_L.clone(), mask_L.clone())
        for n in encoders:
            fn = F.normalize(feats[n][sub], p=2, dim=-1).to(DEV)
            K_sim, mask_K = CC.compute_topk_mask_and_sim(fn, K, DEV)
            res[n].append(CC.asymmetric_platonic_cknna(K_sim, mask_K, mask_L, hsic_LL))
            del fn, K_sim, mask_K
        torch.cuda.empty_cache()
        if (b + 1) % 10 == 0:
            print(f"  [{b + 1}/{args.B}] {time.time() - t0:.0f}s", flush=True)

    R = {n: np.array(v) for n, v in res.items()}
    print("\n=== per-encoder S2 (mean +/- std over resamples) ===", flush=True)
    for n in encoders:
        v = R[n]
        print(f"  {n:24s} {v.mean():.4f} +/- {v.std():.4f}", flush=True)

    print("\n=== paired differences (sign-consistency across resamples) ===", flush=True)
    if args.setting == 1:
        pairs = [("dinov2-base", "vjepa2-vitg"), ("ibot-base", "vjepa2-vitg"),
                 ("vjepa2-vitg", "clip-l14"), ("vjepa2-vitg", "mae-base"),
                 ("vjepa2-vitl", "vjepa2-vith"), ("ijepa-vith14", "mae-base")]
    else:
        pairs = [(f"{v}__real", f"{v}__tiled") for v in vids]
    for a_, b_ in pairs:
        d = R[a_] - R[b_]
        frac = float((np.sign(d) == np.sign(d.mean())).mean())
        z = d.mean() / (d.std() + 1e-12)
        sig = "SIGNIFICANT" if frac >= 0.975 else ("marginal" if frac >= 0.90 else "n.s.")
        print(f"  {a_:22s} - {b_:22s}: mean={d.mean():+.4f} std={d.std():.4f} "
              f"z={z:+.1f} sign-consistent={frac:.0%}  {sig}", flush=True)


if __name__ == "__main__":
    main()
