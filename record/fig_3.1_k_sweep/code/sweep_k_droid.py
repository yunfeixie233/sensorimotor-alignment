#!/usr/bin/env python3
"""k-sweep on DROID for the finetuned-VLA cohort, LLM-out V+T (`imgtext`).

Sweeps k in {2, 5, 10, 15, 20} for h in {1, 3, 7, 15, 25, 40}, n=12 VLAs.

Two modes for the trajectory-side mask:
  * k <= 10 : slice the first-k columns of the existing k=10 DTW topk cache.
  * k >  10 : load the new topk-20 cache (must be precomputed via
              compute_dtw.py --topk 20).

CKNNA: Asymmetric Platonic, identical math to
  compute/compute_cknna.py
  (HSIC unbiased, hsic_LL pre-cached, mask_K and mask_L masks).

Note: the four cache paths below (DATA_DIR / FEAT_DIR / DTW_K10_DIR /
DTW_K20_DIR) resolve from the `DATA_STORE` environment variable (or
the canonical `paths.env` value, sourced as `set -a && source paths.env && set +a`
before running this script). The committed results_k{2,5,10,15,20}.csv
+ summary_r_p.csv already capture the output of this script for the
n=12 cohort, and render_figures.py reads them directly — you only need
to re-run this script if you want to recompute CKNNA for a different
cohort or cache.

Outputs:
  results_k{K}.csv  -- one row per (model, horizon) with cknna value.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr


# 12 finetuned VLAs that appear in the headline §3.1 panel.
FT_VLAS = [
    "Qwen-GR00T-Bridge",
    "Qwen-GR00T-Bridge-RT-1",
    "Qwen-OFT-Bridge-RT-1",
    "Qwen3VL-GR00T-Bridge-RT-1",
    "Qwen3VL-OFT-Bridge-RT-1",
    "cogact-small-bridge",
    "cogact-base-bridge",
    "cogact-large-bridge",
    "spatialvla-sft-bridge",
    "pi0-lerobot-bridge",
    "groot-n15-bridge",
    "groot-n16-bridge",
]

# LLM-out vision-text feature, mean-pooled per episode.
FEAT_FILE = "feats_A.pt"  # imgtext (P3-VT, legacy)

_DATA_STORE  = os.environ.get("DATA_STORE", "/lambda/nfs/vla/cache/cknna_data_store")
DATA_DIR     = f"{_DATA_STORE}/cknna_data_droid"
FEAT_DIR     = f"{_DATA_STORE}/cknna_data_droid_7feat"
DTW_K10_DIR  = f"{_DATA_STORE}/record/dtw_cache/droid"
DTW_K20_DIR  = f"{_DATA_STORE}/record/dtw_cache/droid_topk20"

VLA_SR_CSV   = str((Path(__file__).resolve().parent.parent.parent / "fig_3.1_pa_predicts_sr_1x3" / "data" / "widowx_sr.csv"))

HORIZONS = [1, 3, 7, 15, 25, 40]
KS       = [2, 5, 10, 15, 20]


# ---------------------------------------------------------------------------
# CKNNA math (verbatim from compute_cknna.py)
# ---------------------------------------------------------------------------
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


def build_mask_from_topk(topk_idx_np, N, device):
    idx = torch.from_numpy(topk_idx_np).long().to(device)
    mask = torch.zeros(N, N, device=device)
    mask.scatter_(1, idx, 1.0)
    return mask


def topk_mask_and_sim(feats_norm, k, device):
    sim = feats_norm @ feats_norm.T
    sim_for_topk = sim.clone().fill_diagonal_(float("-inf"))
    _, topk_idx = torch.topk(sim_for_topk, k, dim=1)
    mask = torch.zeros(sim.shape[0], sim.shape[0], device=device)
    mask.scatter_(1, topk_idx, 1.0)
    return sim, mask


def asymmetric_platonic_cknna(K_sim, mask_K, mask_L, hsic_LL):
    mask_inter = mask_K * mask_L
    sim_kl = hsic_unbiased(
        (mask_inter * K_sim).clone(), (mask_inter * mask_L).clone())
    sim_kk = hsic_unbiased(
        (mask_K * K_sim).clone(), (mask_K * K_sim).clone())
    denom = (sim_kk * hsic_LL).clamp(min=1e-10).sqrt()
    return (sim_kl / denom).item()


# ---------------------------------------------------------------------------
# DTW topk loader -- slices a wider cache to the requested k
# ---------------------------------------------------------------------------
def load_dtw_topk(h: int, k: int, k20_required: bool) -> np.ndarray:
    if k <= 10:
        # Slice from the existing k=10 cache.
        path = os.path.join(
            DTW_K10_DIR,
            f"dtw_topk_h{h}_k10_sym2_cuda_nowin_nopad.npy")
        topk_full = np.load(path)
        return topk_full[:, :k].copy()
    # k > 10 needs the topk-20 cache.
    path = os.path.join(
        DTW_K20_DIR,
        f"dtw_topk_h{h}_k20_sym2_cuda_nowin_nopad.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Need DTW topk-20 cache at {path} (re-run compute_dtw.py --topk 20).")
    topk_full = np.load(path)
    return topk_full[:, :k].copy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=KS)
    ap.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    ap.add_argument("--out_dir",
                    default=str(Path(__file__).resolve().parent.parent / "data"),
                    help="defaults to ../data/ next to this script")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(DATA_DIR, "metadata.json")) as f:
        meta = json.load(f)
    N = meta["num_samples"]
    valid_mask = torch.tensor(
        [bool(t and t.strip()) for t in meta["task_descriptions"]],
        dtype=torch.bool)
    N_valid = int(valid_mask.sum().item())
    print(f"N={N}  N_valid={N_valid}")

    # Load SR
    sr = {}
    for r in csv.DictReader(open(VLA_SR_CSV)):
        if r.get("excluded", "").strip().lower() == "true":
            continue
        sr[r["model_key"]] = float(r["success_rate"])

    # Load features for the 12 finetuned VLAs (imgtext only).
    feats_norm = {}
    for m in FT_VLAS:
        p = os.path.join(FEAT_DIR, m, FEAT_FILE)
        if not os.path.exists(p):
            print(f"  [SKIP] {m}: missing {FEAT_FILE}")
            continue
        fa = torch.load(p, weights_only=True).float()
        if fa.shape[0] == N:
            fa = fa[valid_mask]
        elif fa.shape[0] != N_valid:
            print(f"  [WARN] {m}: shape {fa.shape[0]} != {N} or {N_valid}")
        feats_norm[m] = F.normalize(fa, p=2, dim=-1).to(device)
        print(f"  {m:<32}  fa shape={tuple(fa.shape)}")

    print(f"\nLoaded {len(feats_norm)}/{len(FT_VLAS)} model features")

    # For each k, run the sweep.
    summary_rows = []
    t0 = time.time()
    for k in args.ks:
        # Pre-compute K-side topk mask + sim for each model at this k.
        per_model_K = {}
        for m, fn in feats_norm.items():
            sim, mask_K = topk_mask_and_sim(fn, k, device)
            per_model_K[m] = (sim, mask_K)

        # Pre-compute DTW-side mask + hsic_LL for each horizon.
        dtw_masks = {}
        hsic_LL = {}
        for h in args.horizons:
            topk_idx = load_dtw_topk(h, k, k20_required=(k > 10))
            mask_L = build_mask_from_topk(topk_idx, N_valid, device)
            dtw_masks[h] = mask_L
            hsic_LL[h] = hsic_unbiased(mask_L.clone(), mask_L.clone())

        # Compute CKNNA for every (model, horizon).
        rows = []
        for m, (sim, mask_K) in per_model_K.items():
            for h in args.horizons:
                cknna = asymmetric_platonic_cknna(
                    sim, mask_K, dtw_masks[h], hsic_LL[h])
                rows.append({
                    "model": m,
                    "k": k,
                    "horizon": h,
                    "cknna": round(cknna, 6),
                    "success_rate": sr.get(m, float("nan")),
                })

        # Write per-k CSV.
        out_csv = os.path.join(out_dir, f"results_k{k}.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["model", "k", "horizon", "cknna", "success_rate"])
            w.writeheader()
            w.writerows(rows)
        print(f"  k={k:>2}  wrote {out_csv}  ({len(rows)} rows)")

        # Summary: Pearson r per horizon at this k.
        for h in args.horizons:
            xs = np.array([r["cknna"] for r in rows if r["horizon"] == h])
            ys = np.array([r["success_rate"] for r in rows if r["horizon"] == h])
            if len(xs) < 3 or np.std(xs) == 0:
                r_val, p_one = float("nan"), float("nan")
            else:
                r_val, p_two = pearsonr(xs, ys)
                p_one = p_two / 2 if r_val > 0 else 1 - p_two / 2
            summary_rows.append({"k": k, "h": h, "n": len(xs),
                                  "r": r_val, "p_one": p_one})

        # Free per-k tensors.
        del per_model_K, dtw_masks, hsic_LL
        torch.cuda.empty_cache()

    # Print + save summary table.
    print("\n=== Pearson r summary (rows: k, cols: h) ===")
    print(f"{'k':>4}", end="")
    for h in args.horizons:
        print(f"   h={h:<3}", end="")
    print()
    for k in args.ks:
        print(f"{k:>4}", end="")
        for h in args.horizons:
            row = next((r for r in summary_rows if r["k"] == k and r["h"] == h), None)
            if row is None or np.isnan(row["r"]):
                print("    ----", end="")
            else:
                star = "*" if row["p_one"] < 0.05 else " "
                print(f"  {row['r']:+.3f}{star}", end="")
        print()

    out_summary = os.path.join(out_dir, "summary_r_p.csv")
    with open(out_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "h", "n", "r", "p_one"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSummary -> {out_summary}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
