"""
run_bidir_cknna.py

Bidirectional CKNNA experiment: does the VLM primarily encode the FUTURE
or the PAST trajectory?

Design
------
Observation: VLM sees the image at frame t+20 (midpoint of the stored window).
feats_B_seq from cknna_data_exp1v2_full_L has shape (N, 41, 7):
  index 0  = state at t
  index 20 = state at t+20  <- observation point
  index 40 = state at t+40

For horizon h in {1, 3, 7, 10, 15, 20}:
  future trajectory: feats_B_seq[:, 21 : 21+h, :]  -- states after t+20
  past   trajectory: feats_B_seq[:, 20-h : 20, :]  -- states before t+20

Both past and future have exactly h steps, symmetric around the observation.
mask_A (VLM kNN) is identical for both: same VLM features, same observation.
Only the trajectory direction changes.

Hypothesis: if VLM primarily encodes future trajectory, then
  CKNNA_future(h) >> CKNNA_past(h) for all h, especially large h.
If VLM encodes task context equally predictive in both directions:
  CKNNA_future(h) ~ CKNNA_past(h).

Inputs
------
  --data_dir     : cknna_data_bidir_L/  (contains feats_A from t+20 images)
  --traj_dir     : cknna_data_exp1v2_full_L/  (contains feats_B_seq)

Usage
-----
  cd /home/ubuntu/verl/starVLA
  python cknna/record/scripts/run_bidir_cknna.py \\
      --data_dir  cknna/cknna_data_bidir_L \\
      --traj_dir  cknna/cknna_data_exp1v2_full_L \\
      --device cuda
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ---- reuse helpers from run_dtw_cknna.py -----------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from run_dtw_cknna import (
    compute_sym2_cuda_topk,
    compute_pose_distance_topk,
    build_mask_from_topk,
    _hsic_unbiased_lowmem,
    _compute_topk_mask_from_sim,
    POSE_WEIGHTS_V5,
)

TOPK       = 10
MID_IDX    = 20   # feats_B_seq column for t+20
HORIZONS   = [1, 3, 7, 10, 15, 20]


# ============================================================================
# Trajectory slicing
# ============================================================================

def get_traj_future(feats_B_seq, h):
    """States at t+21 to t+20+h  (indices 21..20+h)."""
    return feats_B_seq[:, MID_IDX + 1 : MID_IDX + 1 + h, :]   # (N, h, 7)


def get_traj_past(feats_B_seq, h):
    """States at t+20-h to t+19  (indices 20-h..19), REVERSED to put t+19 first.

    We reverse so that DTW computes temporal distance 'from the observation
    backwards', analogous to future direction 'from the observation forwards'.
    This ensures both comparisons are symmetric in terms of proximity to t+20.
    """
    return feats_B_seq[:, MID_IDX - h : MID_IDX, :].flip(dims=[1])   # (N, h, 7)


# ============================================================================
# DTW top-k (reusing sym2_cuda_nowin_eq)
# ============================================================================

def compute_topk(traj_tensor, h, weights, cache_path):
    """Compute DTW top-k indices for a trajectory batch.

    traj_tensor: (N, h, 7) float32 tensor
    Returns:     (N, TOPK) int32 numpy array
    """
    if os.path.exists(cache_path):
        print(f"    [cache] {os.path.basename(cache_path)}")
        return np.load(cache_path)

    traj = traj_tensor.numpy()
    if h == 1:
        topk = compute_pose_distance_topk(traj, weights)
    else:
        topk = compute_sym2_cuda_topk(traj, weights, sc_window=None)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, topk)
    return topk


# ============================================================================
# CKNNA for one (feats_A, trajectory) pair
# ============================================================================

def mknn_score(mask_A, mask_B, N_h):
    return ((mask_A * mask_B).sum() / (TOPK * N_h)).item()


# ============================================================================
# Plot helpers
# ============================================================================

COLOR_CARD = {
    "StarVLA-Qwen2.5": "#1f77b4",
    "StarVLA-Qwen3":   "#aec7e8",
    "CogACT":          "#ff7f0e",
    "GR00T":           "#2ca02c",
    "OpenVLA":         "#d62728",
    "Pi0":             "#9467bd",
    "SpatialVLA":      "#8c564b",
    "RT-1-X":          "#7f7f7f",
    "Octo":            "#bcbd22",
}

_FAMILY_MAP = {
    "Qwen-GR00T-Bridge":           "StarVLA-Qwen2.5",
    "Qwen-GR00T-Bridge-RT-1":      "StarVLA-Qwen2.5",
    "Qwen-OFT-Bridge-RT-1":        "StarVLA-Qwen2.5",
    "Qwen3VL-GR00T-Bridge-RT-1":   "StarVLA-Qwen3",
    "Qwen3VL-OFT-Bridge-RT-1":     "StarVLA-Qwen3",
    "qwen25vl-3b-raw":             "StarVLA-Qwen2.5",
    "qwen3vl-4b-raw":              "StarVLA-Qwen3",
    "prismatic-raw":               "OpenVLA",
    "cogact-small-bridge":         "CogACT",
    "cogact-base-bridge":          "CogACT",
    "cogact-large-bridge":         "CogACT",
    "groot-n15-bridge":            "GR00T",
    "groot-n16-bridge":            "GR00T",
    "openvla-7b-bridge":           "OpenVLA",
    "openvla-7b-bridge-ft-200k":   "OpenVLA",
    "pi0-lerobot-bridge":          "Pi0",
    "spatialvla-sft-bridge":       "SpatialVLA",
    "rt1x-bridge":                 "RT-1-X",
    "octo-base-bridge":            "Octo",
}


def model_color(model_key):
    family = _FAMILY_MAP.get(model_key, "Octo")
    return COLOR_CARD.get(family, "#333333")


def plot_summary(rows, out_dir):
    """Summary plot: mean CKNNA_future vs CKNNA_past across all models."""
    from collections import defaultdict
    data = defaultdict(lambda: defaultdict(list))
    for r in rows:
        data[r["direction"]][r["horizon"]].append(r["cknna_k10"])

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for direction, color, ls in [("future", "#1f77b4", "-"), ("past", "#d62728", "--")]:
        if direction not in data:
            continue
        hs = sorted(data[direction].keys())
        means = [np.mean(data[direction][h]) for h in hs]
        ax.plot(hs, means, color=color, ls=ls, lw=2,
                marker="o", label=f"CKNNA_{direction} (mean)")

    ax.set_xlabel("Horizon h (steps)")
    ax.set_ylabel("CKNNA (k=10, DTW sym2_nowin_eq)")
    ax.set_title("Bidirectional CKNNA: Future vs Past Trajectory\n(mean over all models)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "bidir_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_per_model(rows, model_keys, out_dir):
    """Per-model line plot: CKNNA_future vs CKNNA_past."""
    from collections import defaultdict

    # future/past per model per horizon (imgtext only)
    data = defaultdict(lambda: {"future": {}, "past": {}})
    for r in rows:
        if r["feats_A_variant"] != "imgtext":
            continue
        data[r["model"]][r["direction"]][r["horizon"]] = r["cknna_k10"]

    n_models = len([m for m in model_keys if m in data])
    ncols = 4
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             sharey=False)
    axes = np.array(axes).flatten()

    plotted = 0
    for model_key in model_keys:
        if model_key not in data:
            continue
        ax  = axes[plotted]
        d   = data[model_key]
        col = model_color(model_key)
        hs  = sorted(HORIZONS)

        fut_vals  = [d["future"].get(h, None) for h in hs]
        past_vals = [d["past"].get(h, None) for h in hs]
        ax.plot(hs, fut_vals,  color=col,       ls="-",  lw=2, marker="o", label="future")
        ax.plot(hs, past_vals, color="#aaaaaa",  ls="--", lw=1.5, marker="s", label="past")
        ax.set_title(model_key, fontsize=7)
        ax.set_xlabel("h")
        ax.set_ylabel("CKNNA")
        ax.grid(True, alpha=0.3)
        if plotted == 0:
            ax.legend(fontsize=6)
        plotted += 1

    for i in range(plotted, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Bidirectional CKNNA (imgtext, sym2_nowin_eq)", fontsize=11, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "bidir_per_model.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_asymmetry(rows, model_keys, out_dir):
    """Bar plot of delta = CKNNA_future - CKNNA_past at h=20 per model."""
    from collections import defaultdict
    delta = {}
    for r in rows:
        if r["feats_A_variant"] != "imgtext" or r["horizon"] != 20:
            continue
        m = r["model"]
        if m not in delta:
            delta[m] = {}
        delta[m][r["direction"]] = r["cknna_k10"]

    models = [m for m in model_keys if m in delta
              and "future" in delta[m] and "past" in delta[m]]
    vals = [delta[m]["future"] - delta[m]["past"] for m in models]
    colors = ["#1f77b4" if v > 0 else "#d62728" for v in vals]

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 0.6), 4))
    ax.bar(range(len(models)), vals, color=colors)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("CKNNA_future - CKNNA_past  (h=20)")
    ax.set_title("Future - Past Asymmetry at h=20  (blue = future > past)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "bidir_asymmetry_h20.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  required=True,
                        help="Directory containing feats_A from t+20 images "
                             "(i.e. cknna_data_bidir_L/)")
    parser.add_argument("--traj_dir",  required=True,
                        help="Directory containing feats_B_seq.pt "
                             "(i.e. cknna_data_exp1v2_full_L/)")
    parser.add_argument("--device",    default="cuda")
    parser.add_argument("--horizons",  type=int, nargs="+", default=HORIZONS)
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    traj_dir = os.path.abspath(args.traj_dir)
    device   = args.device
    horizons = sorted(args.horizons)
    max_h    = max(horizons)
    assert max_h <= MID_IDX, f"max horizon {max_h} > MID_IDX {MID_IDX}"

    out_dir  = os.path.join(
        os.path.dirname(SCRIPT_DIR), "bidir", "dtw_sym2_nowin_eq"
    )
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.join(
        os.path.dirname(SCRIPT_DIR), "dtw_cache", "bidir_L"
    )

    print("=== run_bidir_cknna.py ===")
    print(f"  data_dir : {data_dir}")
    print(f"  traj_dir : {traj_dir}")
    print(f"  horizons : {horizons}")
    print(f"  output   : {out_dir}")

    # ----------------------------------------------------------------
    # Load trajectory data (from original full_L dataset)
    # ----------------------------------------------------------------
    feats_B_seq = torch.load(
        os.path.join(traj_dir, "feats_B_seq.pt"), weights_only=True
    ).float()
    print(f"\nfeats_B_seq: {tuple(feats_B_seq.shape)}")  # (7762, 41, 7)

    with open(os.path.join(traj_dir, "metadata.json")) as f:
        meta = json.load(f)
    task_descs = meta["task_descriptions"]
    valid_mask = torch.tensor(
        [bool(t and t.strip()) for t in task_descs], dtype=torch.bool
    )
    N_valid = valid_mask.sum().item()
    print(f"Valid (labeled) samples: {N_valid} / {len(valid_mask)}")

    feats_B_seq_valid = feats_B_seq[valid_mask]   # (N_valid, 41, 7)
    del feats_B_seq

    # ----------------------------------------------------------------
    # Discover models in data_dir
    # ----------------------------------------------------------------
    model_dirs = {}
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        fa_path   = os.path.join(full_path, "feats_A.pt")
        if os.path.isdir(full_path) and os.path.exists(fa_path):
            model_dirs[entry] = full_path

    print(f"\nModels found: {list(model_dirs.keys())}")

    # ----------------------------------------------------------------
    # Pre-compute DTW top-k for both directions at each horizon
    # ----------------------------------------------------------------
    print("\n--- Computing DTW top-k for past + future trajectories ---")
    topk_future = {}
    topk_past   = {}

    for h in horizons:
        traj_fut  = get_traj_future(feats_B_seq_valid, h)   # (N, h, 7)
        traj_past = get_traj_past(feats_B_seq_valid, h)     # (N, h, 7)

        cp_fut  = os.path.join(cache_dir, f"topk_future_h{h}_k{TOPK}.npy")
        cp_past = os.path.join(cache_dir, f"topk_past_h{h}_k{TOPK}.npy")

        print(f"  h={h}: future traj shape {tuple(traj_fut.shape)}")
        topk_future[h] = compute_topk(traj_fut,  h, POSE_WEIGHTS_V5, cp_fut)
        topk_past[h]   = compute_topk(traj_past, h, POSE_WEIGHTS_V5, cp_past)

    # ----------------------------------------------------------------
    # Pre-build trajectory masks and HSIC_BB
    # ----------------------------------------------------------------
    print("\n--- Building trajectory masks on GPU ---")
    mask_B_future = {}
    mask_B_past   = {}
    hsic_BB_future = {}
    hsic_BB_past   = {}

    for h in horizons:
        N_h = N_valid
        mf = build_mask_from_topk(topk_future[h], N_h, device)
        mp = build_mask_from_topk(topk_past[h],   N_h, device)
        mask_B_future[h] = mf
        mask_B_past[h]   = mp
        hsic_BB_future[h] = _hsic_unbiased_lowmem(mf.clone(), mf.clone())
        hsic_BB_past[h]   = _hsic_unbiased_lowmem(mp.clone(), mp.clone())
        print(f"  h={h}: hsic_BB_future={hsic_BB_future[h].item():.6f}  "
              f"hsic_BB_past={hsic_BB_past[h].item():.6f}")

    # ----------------------------------------------------------------
    # Per-model CKNNA
    # ----------------------------------------------------------------
    print("\n--- Computing per-model CKNNA ---")
    csv_rows = []

    for model_key, model_path in model_dirs.items():
        fa_files = {"imgtext": "feats_A.pt"}
        for tag, fname in [("img", "feats_A_img.pt"), ("txt", "feats_A_txt.pt")]:
            if os.path.exists(os.path.join(model_path, fname)):
                fa_files[tag] = fname

        print(f"\n  Model: {model_key}")

        for variant, fname in fa_files.items():
            fa_full = torch.load(
                os.path.join(model_path, fname), weights_only=True
            ).float()
            fa_valid = fa_full[valid_mask]
            fa_norm  = F.normalize(fa_valid.to(device), p=2, dim=-1)

            # Compute mask_A once (same for both directions)
            sim_A  = fa_norm @ fa_norm.T
            mask_A = _compute_topk_mask_from_sim(sim_A, TOPK, N_valid, device)
            del sim_A
            hsic_AA = _hsic_unbiased_lowmem(mask_A.clone(), mask_A.clone())

            for h in horizons:
                N_h = N_valid

                # Future
                cknna_fut = cknna_score_with_cached(
                    mask_A, mask_B_future[h], hsic_AA, hsic_BB_future[h]
                )
                mknn_fut  = mknn_score(mask_A, mask_B_future[h], N_h)

                # Past
                cknna_past_val = cknna_score_with_cached(
                    mask_A, mask_B_past[h], hsic_AA, hsic_BB_past[h]
                )
                mknn_past = mknn_score(mask_A, mask_B_past[h], N_h)

                print(f"    {variant} h={h}: "
                      f"future={cknna_fut:.5f}  past={cknna_past_val:.5f}  "
                      f"delta={cknna_fut - cknna_past_val:+.5f}")

                for direction, cknna_val, mknn_val in [
                    ("future", cknna_fut, mknn_fut),
                    ("past",   cknna_past_val, mknn_past),
                ]:
                    csv_rows.append({
                        "model":           model_key,
                        "feats_A_variant": variant,
                        "direction":       direction,
                        "horizon":         h,
                        "N":               N_h,
                        "cknna_k10":       round(cknna_val, 6),
                        "mutual_knn_k10":  round(mknn_val, 6),
                    })

            del mask_A, fa_norm, fa_valid, fa_full
            torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # Save CSV
    # ----------------------------------------------------------------
    csv_path = os.path.join(out_dir, "results_bidir.csv")
    fieldnames = ["model", "feats_A_variant", "direction", "horizon",
                  "N", "cknna_k10", "mutual_knn_k10"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved CSV: {csv_path}  ({len(csv_rows)} rows)")

    # ----------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------
    print("\n--- Generating plots ---")
    plot_summary(csv_rows, out_dir)
    plot_per_model(csv_rows, list(model_dirs.keys()), out_dir)
    plot_asymmetry(csv_rows, list(model_dirs.keys()), out_dir)

    print(f"\n=== Done. Results in {out_dir}/ ===")


def cknna_score_with_cached(mask_A, mask_B, hsic_AA, hsic_BB):
    """Compute CKNNA using pre-computed hsic_AA and hsic_BB."""
    hsic_AB = _hsic_unbiased_lowmem(mask_A.clone(), mask_B.clone())
    denom   = (hsic_AA * hsic_BB).clamp(min=1e-10).sqrt()
    return (hsic_AB / denom).item()


if __name__ == "__main__":
    main()
