"""Step 4 — reverse-verification: does a max-pool readout lift any video JEPA
into the DINO band? Computes full-N CKNNA @ h=7 (no resample) with the exact
production code path, for the three -standalone-maxpool dirs."""
import sys

import numpy as np
import torch
import torch.nn.functional as F

import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _os.path.join(_REPO_ROOT, "compute"))
import compute_dtw as CD          # noqa: E402
import compute_cknna as CC        # noqa: E402

import json

H, K, DEV = 7, 10, "cuda"
DD = "/lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid"

# (display name, maxpool dir, meanall baseline CKNNA from the §F CSV)
ENC = [
    ("V-JEPA ViT-L/16", "vjepa-v1-vitl-standalone-maxpool", 0.0228),
    ("V-JEPA ViT-H/16", "vjepa-v1-vith-standalone-maxpool", 0.0238),
    ("V-JEPA 2 ViT-L", "vjepa2-vitl-standalone-maxpool", 0.0228),
    ("V-JEPA 2 ViT-H", "vjepa2-vith-standalone-maxpool", 0.0223),
    ("V-JEPA 2 ViT-g", "vjepa2-vitg-standalone-maxpool", 0.0245),
    ("V-JEPA 2.1 ViT-L", "vjepa2-1-vitl-standalone-maxpool", 0.0240),
    ("V-JEPA 2.1 ViT-g", "vjepa2-1-vitg-standalone-maxpool", 0.0235),
]
DINO_BAND = (0.0257, 0.0278)   # DINOv2/iBOT band from §F


def main():
    meta = json.load(open(f"{DD}/metadata.json"))
    valid = np.array([bool(t and t.strip()) for t in meta["task_descriptions"]])
    fB = torch.load(f"{DD}/feats_B_seq.pt", weights_only=True).float()[valid]
    Nv = int(valid.sum())
    print(f"N_valid={Nv}  h={H}  k={K}", flush=True)

    traj = fB[:, 1:H + 1, :].numpy()
    dtw_topk = CD.compute_dtw_topk_gpu(traj, K)
    mask_L = CC.build_mask_from_topk(dtw_topk, Nv, DEV)
    hsic_LL = CC.hsic_unbiased(mask_L.clone(), mask_L.clone())

    print(f"\nDINO/iBOT band = [{DINO_BAND[0]}, {DINO_BAND[1]}]\n", flush=True)
    for name, d, base in ENC:
        f = torch.load(f"{DD}/{d}/feats_A_vit.pt", weights_only=True).float()[valid]
        fn = F.normalize(f, p=2, dim=-1).to(DEV)
        K_sim, mask_K = CC.compute_topk_mask_and_sim(fn, K, DEV)
        s2 = CC.asymmetric_platonic_cknna(K_sim, mask_K, mask_L, hsic_LL)
        verdict = ("IN DINO BAND" if s2 >= DINO_BAND[0]
                   else f"below band by {DINO_BAND[0]-s2:+.4f}")
        print(f"  {name:16s} maxpool S2={s2:.4f}  (meanall {base:.4f}, "
              f"delta {s2-base:+.4f})  -> {verdict}", flush=True)
        del fn, K_sim, mask_K
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
