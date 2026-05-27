"""
Phase 2: Compute Asymmetric Platonic CKNNA for 7 features across multiple models.

Reads:
    - DTW top-k cache:  {dtw_cache_dir}/dtw_topk_h{H}_k{K}_sym2_cuda_nowin_nopad.npy
    - Model features:   {feat_dir}/{model}/feats_A_*.pt  (7 feature files per model)
    - Metadata:         {data_dir}/metadata.json          (for valid-task filtering)

Outputs:
    - {output_csv}  — one row per (model, feature, horizon) with cknna_k10 value

Method:
    Asymmetric Platonic CKNNA (Huh et al., adapted):
      K-side: cosine similarity on model features, top-k binary mask
      L-side: binary DTW top-k mask (also used as similarity matrix)
      CKNNA = HSIC(mask_inter * K_sim, mask_inter * mask_L)
              / sqrt(HSIC(mask_K * K_sim, mask_K * K_sim) * HSIC(mask_L, mask_L))

Usage:
    conda run -n starVLA python compute_cknna.py \
        --data_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid \
        --feat_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid_7feat \
        --dtw_cache_dir /lambda/nfs/vla/cache/cknna_data_store/record/dtw_cache/droid \
        --output_csv ../droid/results_cknna_7feat.csv

Typical runtime: ~5-15 minutes on GPU for 14 models, 7 features, 10 horizons.
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F


TOPK = 10

VARIANT_FILES = [
    ("vit_raw",     "feats_A_vit.pt"),         # P1-V
    ("pre_llm",     "feats_A_pre_llm.pt"),     # P2-V
    ("pre_llm_txt", "feats_A_pre_llm_txt.pt"), # P2-T  (instruction-only, scaffolding-free)
    ("pre_llm_vt",  "feats_A_pre_llm_vt.pt"),  # P2-VT (image ∪ instruction, scaffolding-free)
    ("img",         "feats_A_img.pt"),         # P3-V
    ("txt",         "feats_A_txt.pt"),         # P3-T  (instruction-only, scaffolding-free)
    ("imgtext",     "feats_A.pt"),             # P3-VT (image ∪ instruction, scaffolding-free)
]


# ── HSIC (Song et al. 2012, unbiased estimator) ──────────────────────────────

def hsic_unbiased(K, L):
    m = K.shape[0]
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)
    chunk = 2000
    term1 = torch.tensor(0.0, device=K.device, dtype=K.dtype)
    for i in range(0, m, chunk):
        end = min(i + chunk, m)
        term1 += (K_tilde[i:end] * L_tilde.T[i:end]).sum()
    term2 = K_tilde.sum() * L_tilde.sum() / ((m - 1) * (m - 2))
    k_col = K_tilde.sum(dim=0)
    l_row = L_tilde.sum(dim=1)
    term3 = 2.0 * (k_col @ l_row) / (m - 2)
    return (term1 + term2 - term3) / (m * (m - 3))


# ── Mask + similarity helpers ────────────────────────────────────────────────

def build_mask_from_topk(topk_indices, N, device):
    topk_idx = torch.from_numpy(topk_indices).long().to(device)
    mask = torch.zeros(N, N, device=device)
    mask.scatter_(1, topk_idx, 1.0)
    return mask


def compute_topk_mask_and_sim(feats_norm, k, device):
    sim = feats_norm @ feats_norm.T
    sim_for_topk = sim.clone().fill_diagonal_(float("-inf"))
    _, topk_idx = torch.topk(sim_for_topk, k, dim=1)
    mask = torch.zeros(sim.shape[0], sim.shape[0], device=device)
    mask.scatter_(1, topk_idx, 1.0)
    return sim, mask


def asymmetric_platonic_cknna(K_sim, mask_K, mask_L, hsic_LL):
    """Asymmetric Platonic CKNNA with pre-cached HSIC(mask_L, mask_L)."""
    mask_inter = mask_K * mask_L
    sim_kl = hsic_unbiased(
        (mask_inter * K_sim).clone(), (mask_inter * mask_L).clone())
    sim_kk = hsic_unbiased(
        (mask_K * K_sim).clone(), (mask_K * K_sim).clone())
    denom = (sim_kk * hsic_LL).clamp(min=1e-10).sqrt()
    return (sim_kl / denom).item()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Compute Asymmetric Platonic CKNNA for 7 features.")
    parser.add_argument("--data_dir", required=True,
                        help="Dir with metadata.json (for valid-task mask)")
    parser.add_argument("--feat_dir", required=True,
                        help="Dir with per-model feature subdirectories")
    parser.add_argument("--dtw_cache_dir", required=True,
                        help="Dir with dtw_topk_*.npy files from Phase 1")
    parser.add_argument("--output_csv", required=True,
                        help="Output CSV path")
    parser.add_argument("--horizons", type=int, nargs="+",
                        default=[1, 3, 7, 15, 25, 40, 75, 110, 150, 220])
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtw_version", default="sym2_cuda_nowin",
                        help="DTW version string for cache filename")
    parser.add_argument("--model_filter", default=None,
                        help="Regex filter for model directory names (e.g. '.*-raw$')")
    args = parser.parse_args()

    device = args.device
    K = args.topk

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    # Load metadata for valid-task mask
    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        meta = json.load(f)
    N = meta["num_samples"]
    valid_mask = torch.tensor(
        [bool(t and t.strip()) for t in meta["task_descriptions"]],
        dtype=torch.bool)
    N_valid = valid_mask.sum().item()
    print(f"N={N}, valid={N_valid}")

    # Discover models from feat_dir
    import re
    models = sorted([
        d for d in os.listdir(args.feat_dir)
        if os.path.isdir(os.path.join(args.feat_dir, d))
    ])
    if args.model_filter:
        pat = re.compile(args.model_filter)
        models = [m for m in models if pat.match(m)]
    print(f"Models found: {len(models)}")
    for m in models:
        print(f"  {m}")

    # Load DTW caches and pre-compute HSIC(mask_L, mask_L)
    dtw_masks = {}
    hsic_LL_cache = {}
    for h in args.horizons:
        cache_file = os.path.join(
            args.dtw_cache_dir,
            f"dtw_topk_h{h}_k{K}_{args.dtw_version}_nopad.npy")
        if not os.path.exists(cache_file):
            print(f"  h={h}: SKIP (no cache at {cache_file})")
            continue
        topk_idx = np.load(cache_file)
        N_h = topk_idx.shape[0]
        mask_B = build_mask_from_topk(topk_idx, N_h, device)
        dtw_masks[h] = mask_B
        hsic_LL_cache[h] = hsic_unbiased(mask_B.clone(), mask_B.clone())
        print(f"  h={h}: DTW loaded (N={N_h}), hsic_LL={hsic_LL_cache[h].item():.6f}")

    if not dtw_masks:
        print("ERROR: No DTW cache found. Run compute_dtw.py first.")
        return

    # Compute CKNNA for all models x features x horizons
    csv_rows = []
    t0 = time.time()

    for model_name in models:
        model_dir = os.path.join(args.feat_dir, model_name)

        # Load all available feature variants
        fa_variants = {}
        for vname, fname in VARIANT_FILES:
            p = os.path.join(model_dir, fname)
            if os.path.exists(p):
                fa_variants[vname] = torch.load(p, weights_only=True).float()

        if not fa_variants:
            print(f"\n  [SKIP] {model_name}: no feature files")
            continue

        print(f"\n  Model: {model_name} ({len(fa_variants)} variants)")

        for vname, fa_full in fa_variants.items():
            # Apply valid mask if needed
            if fa_full.shape[0] == N:
                fa_g = fa_full[valid_mask]
            elif fa_full.shape[0] == N_valid:
                fa_g = fa_full
            else:
                print(f"    [WARN] {vname}: shape {fa_full.shape[0]} != N({N}) or N_valid({N_valid})")
                fa_g = fa_full

            fa_g_norm = F.normalize(fa_g, p=2, dim=-1).to(device)
            K_sim, mask_K = compute_topk_mask_and_sim(fa_g_norm, K, device)
            del fa_g_norm

            for h in args.horizons:
                if h not in dtw_masks:
                    continue
                mask_L = dtw_masks[h]
                cknna_val = asymmetric_platonic_cknna(
                    K_sim, mask_K, mask_L, hsic_LL_cache[h])
                mknn_val = ((mask_K * mask_L).sum() / (K * mask_L.shape[0])).item()

                csv_rows.append({
                    "model": model_name,
                    "group": "L",
                    "N": mask_L.shape[0],
                    "feats_A_variant": vname,
                    "feats_B_type": "proprio_seq_dtw",
                    "horizon": h,
                    "cknna_k10": round(cknna_val, 6),
                    "mutual_knn_k10": round(mknn_val, 6),
                })

            del K_sim, mask_K
            print(f"    {vname}: done")
            torch.cuda.empty_cache()

    # Save CSV
    fieldnames = ["model", "group", "N", "feats_A_variant", "feats_B_type",
                  "horizon", "cknna_k10", "mutual_knn_k10"]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    elapsed = time.time() - t0
    print(f"\nSaved: {args.output_csv} ({len(csv_rows)} rows)")
    print(f"Total time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
