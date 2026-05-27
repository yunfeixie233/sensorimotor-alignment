"""
Re-compute multi-frame CKNNA with CORRECT formula:
  - A-side: cosine kNN → binary mask_A
  - B-side: DTW kNN (precomputed or recomputed) → binary mask_B
  - CKNNA = HSIC(mask_A, mask_B) / sqrt(HSIC(mask_A, mask_A) * HSIC(mask_B, mask_B))

This replaces compute_multiframe_cknna.py which used the WRONG formula:
  HSIC(intersection_mask * sim_A, intersection_mask * sim_B) with cosine B-side.

Supports multiple pooling strategies from feats_A_percam.pt [N, T, 3, 3072]:
  - cam1_anchor: last frame, camera 0 only → [N, 3072]
  - cam1_meanT:  mean across T frames, camera 0 only → [N, 3072]
  - 3cam_anchor: last frame, all 3 cameras mean → [N, 3072]
  - 3cam_meanT:  mean across T frames and 3 cameras → [N, 3072]
  - global_mean: feats_A.pt as-is → [N, 3072]

Usage:
  python analysis/compute_multiframe_cknna_correct.py \
      --m_values 0 1 2 4 8 16 32 64 --h 75 --topk 10 --device cuda
"""

import argparse
import datetime
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

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = "/lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid_multiframe"
OUT_DIR  = "/home/ubuntu/vla/cknna_project/record/multiframe_cknna"
os.makedirs(OUT_DIR, exist_ok=True)


# ── HSIC (unbiased estimator, matches run_dtw_cknna.py) ─────────────────────

def hsic_unbiased(K_mat, L_mat):
    """Unbiased HSIC estimator (Song et al., 2012).

    Applied to BINARY kNN masks [N, N]. Modifies inputs in-place (fill_diagonal_).
    """
    m = K_mat.shape[0]
    K_mat.fill_diagonal_(0)
    L_mat.fill_diagonal_(0)
    # term1: trace(K @ L.T) = sum(K * L.T)
    term1 = (K_mat * L_mat.T).sum()
    term2 = K_mat.sum() * L_mat.sum() / ((m - 1) * (m - 2))
    k_col = K_mat.sum(dim=0)
    l_row = L_mat.sum(dim=1)
    term3 = 2.0 * (k_col @ l_row) / (m - 2)
    return (term1 + term2 - term3) / (m * (m - 3))


def build_mask_from_topk(topk_np, n, device):
    """Build [N, N] binary kNN mask from top-k indices [N, k]."""
    topk_idx = torch.from_numpy(topk_np).long().to(device)
    mask = torch.zeros(n, n, dtype=torch.float32, device=device)
    mask.scatter_(1, topk_idx, 1.0)
    return mask


def compute_cosine_topk(feats, topk, device):
    """Compute cosine kNN top-k indices from features [N, D]."""
    feats_norm = F.normalize(feats.float().to(device), p=2, dim=-1)
    sim = feats_norm @ feats_norm.T
    sim.fill_diagonal_(float("-inf"))
    _, topk_idx = sim.topk(topk, dim=1)
    return topk_idx


def compute_cknna(mask_A, mask_B):
    """CKNNA = HSIC(mask_A, mask_B) / sqrt(HSIC(mask_A, mask_A) * HSIC(mask_B, mask_B))"""
    hsic_AB = hsic_unbiased(mask_A.clone(), mask_B.clone())
    hsic_AA = hsic_unbiased(mask_A.clone(), mask_A.clone())
    hsic_BB = hsic_unbiased(mask_B.clone(), mask_B.clone())
    denom = (hsic_AA * hsic_BB).clamp(min=1e-10).sqrt()
    return (hsic_AB / denom).item()


def compute_mutual_knn(mask_A, mask_B, topk):
    """Simple mutual kNN overlap: fraction of neighbors shared."""
    n = mask_A.shape[0]
    return ((mask_A * mask_B).sum() / (topk * n)).item()


# ── DTW recomputation (soft_cuda) ───────────────────────────────────────────

def recompute_dtw_topk(feats_B_seq, valid_mask, h, topk, device):
    """Recompute DTW top-k using GPU SoftDTW on subset of samples.

    Args:
        feats_B_seq: [N_total, T_max, 7] full trajectory data
        valid_mask: [N_total] bool, which samples to include
        h: horizon (use first h steps of trajectory)
        topk: k for kNN
        device: cuda device

    Returns:
        topk_indices: [N_valid, topk] int64 numpy array
    """
    try:
        from sdtw_cuda import SoftDTW
    except ImportError:
        print("WARNING: sdtw_cuda not available, falling back to L2 distance")
        return _fallback_l2_topk(feats_B_seq, valid_mask, h, topk, device)

    # Extract valid trajectories up to horizon h
    traj = feats_B_seq[valid_mask, 1:h+1, :].float()  # [N, h, 7]
    N = traj.shape[0]
    print(f"  Computing SoftDTW distance matrix: N={N}, h={h}")

    # SoftDTW expects [B, T, D]
    sdtw = SoftDTW(use_cuda=True, gamma=0.001, bandwidth=int(0.1 * h))

    # Compute pairwise DTW in chunks to save memory
    chunk = 500
    dist_topk = torch.full((N, topk), -1, dtype=torch.long)

    traj_gpu = traj.to(device)

    for i in range(0, N, chunk):
        end_i = min(i + chunk, N)
        batch_i = traj_gpu[i:end_i]  # [chunk, h, 7]

        # Compute distances from batch_i to all samples
        dists_row = []
        for j in range(0, N, chunk):
            end_j = min(j + chunk, N)
            batch_j = traj_gpu[j:end_j]

            # SoftDTW: compute pairwise
            ni, nj = batch_i.shape[0], batch_j.shape[0]
            bi = batch_i.unsqueeze(1).expand(-1, nj, -1, -1).reshape(ni * nj, h, 7)
            bj = batch_j.unsqueeze(0).expand(ni, -1, -1, -1).reshape(ni * nj, h, 7)
            d = sdtw(bi, bj).reshape(ni, nj)
            dists_row.append(d)

        dists_full = torch.cat(dists_row, dim=1)  # [chunk, N]
        # Set self-distance to inf
        for idx in range(end_i - i):
            dists_full[idx, i + idx] = float("inf")

        # Top-k smallest distances
        _, topk_idx = dists_full.topk(topk, dim=1, largest=False)
        dist_topk[i:end_i] = topk_idx.cpu()

        if i % 1000 == 0:
            print(f"    DTW progress: {i}/{N}")

    return dist_topk.numpy()


def _fallback_l2_topk(feats_B_seq, valid_mask, h, topk, device):
    """Fallback: L2 distance on flattened trajectory if SoftDTW unavailable."""
    traj = feats_B_seq[valid_mask, 1:h+1, :].reshape(-1, h * 7).float().to(device)
    traj_norm = F.normalize(traj, p=2, dim=-1)
    sim = traj_norm @ traj_norm.T
    sim.fill_diagonal_(float("-inf"))
    _, topk_idx = sim.topk(topk, dim=1)
    # Convert to distance-based topk (negate sim)
    return topk_idx.cpu().numpy()


# ── Pooling strategies ───────────────────────────────────────────────────────

def pool_features(percam_feat, m, num_cameras):
    """Pool feats_A_percam.pt [N, T, C, 3072] into various strategies.

    Returns dict of {strategy_name: [N, 3072] tensor}.
    """
    N, T, C, D = percam_feat.shape
    result = {}

    # cam1 anchor: last frame, camera 0
    result["cam1_anchor"] = percam_feat[:, -1, 0, :].float()

    # cam1 mean-T: mean across T frames, camera 0
    result["cam1_meanT"] = percam_feat[:, :, 0, :].float().mean(dim=1)

    if C >= 3:
        # 3cam anchor: last frame, CONCAT 3 cameras → [N, 3*D]
        result["3cam_anchor"] = percam_feat[:, -1, :, :].float().reshape(N, -1)

        # 3cam mean-T: mean across T frames, then CONCAT 3 cameras → [N, 3*D]
        # For each camera, mean across T → [N, 3, D], then flatten → [N, 3*D]
        result["3cam_meanT"] = percam_feat.float().mean(dim=1).reshape(N, -1)

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Re-compute multiframe CKNNA with CORRECT HSIC+DTW formula."
    )
    parser.add_argument("--m_values", type=int, nargs="+",
                        default=[0, 1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--h", type=int, default=75)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_prefix", default="lingbot-va-base-joint",
                        help="Model directory prefix (e.g., 'lingbot-va-base-joint' or 'wan-raw-joint')")
    parser.add_argument("--output_tag", default="",
                        help="Optional tag for output filenames")
    parser.add_argument("--dir_suffix", default="",
                        help="Suffix appended to model dir name (e.g., '-withactions')")
    args = parser.parse_args()

    device = args.device
    m_values = sorted(args.m_values)
    h = args.h
    topk = args.topk
    tag = f"_{args.output_tag}" if args.output_tag else ""

    print(f"=== Multi-Frame CKNNA (CORRECT formula) ===")
    print(f"  m values: {m_values}")
    print(f"  h={h}, k={topk}")
    print(f"  Model prefix: {args.model_prefix}")
    print(f"  Data dir: {DATA_DIR}")
    print()

    # ── Step 1: Find common valid indices across all m values ────────────────
    print("Step 1: Finding common valid indices...")
    valid_sets = {}
    for m in m_values:
        meta_path = os.path.join(
            DATA_DIR, f"{args.model_prefix}-m{m}{args.dir_suffix}", "extraction_metadata.json"
        )
        if not os.path.exists(meta_path):
            print(f"  m={m}: metadata NOT FOUND — skipping")
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        valid_sets[m] = set(meta["valid_indices"])
        print(f"  m={m}: N_valid={len(valid_sets[m])}, skipped={meta['skipped_indices']}")

    # Intersection of all valid sets
    common_valid = set.intersection(*valid_sets.values())
    common_valid_sorted = sorted(common_valid)
    N_common = len(common_valid_sorted)
    print(f"\n  Common valid samples: {N_common}")

    # Build index mapping: for each m, which rows of feats_A correspond to common set
    # feats_A[i] corresponds to valid_indices[i], so we need to map common_valid → row index
    def get_row_indices(m):
        """Return row indices into feats_A for this m that correspond to common_valid_sorted."""
        meta_path = os.path.join(
            DATA_DIR, f"{args.model_prefix}-m{m}{args.dir_suffix}", "extraction_metadata.json"
        )
        with open(meta_path) as f:
            vi = json.load(f)["valid_indices"]
        vi_to_row = {v: i for i, v in enumerate(vi)}
        return [vi_to_row[v] for v in common_valid_sorted]

    # ── Step 2: Prepare B-side DTW ───────────────────────────────────────────
    print("\nStep 2: Preparing B-side DTW...")

    # Check if we can reuse existing DTW topk
    dtw_cache_path = os.path.join(DATA_DIR, f"dtw_softcuda_n{N_common}_h{h}.npy")
    existing_dtw_path = os.path.join(DATA_DIR, f"dtw_softcuda_n1999_h{h}.npy")

    if os.path.exists(dtw_cache_path):
        print(f"  Loading cached DTW topk: {dtw_cache_path}")
        dtw_topk = np.load(dtw_cache_path)
    elif N_common == 1999 and os.path.exists(existing_dtw_path):
        print(f"  N_common=1999 matches existing DTW, loading: {existing_dtw_path}")
        dtw_topk = np.load(existing_dtw_path)
    else:
        # Need to recompute DTW for the common subset
        print(f"  N_common={N_common} != 1999, need to reindex or recompute DTW...")

        # Try reindexing: if common_valid is a subset of the 1999-sample set,
        # we can remap the DTW indices
        if os.path.exists(existing_dtw_path):
            # The 1999-sample set is samples 0..1999 minus 968
            full_1999_set = [i for i in range(2000) if i != 968]
            full_1999_to_row = {v: i for i, v in enumerate(full_1999_set)}

            # Check if all common_valid are in the 1999 set
            if all(v in full_1999_to_row for v in common_valid_sorted):
                print(f"  Reindexing existing DTW from 1999 → {N_common}...")
                dtw_1999 = np.load(existing_dtw_path)

                # Map common_valid to rows in the 1999-sample DTW
                rows_in_1999 = [full_1999_to_row[v] for v in common_valid_sorted]

                # Build reverse mapping: 1999-row → common-row (or -1 if excluded)
                row_1999_to_common = np.full(1999, -1, dtype=np.int64)
                for common_row, row_1999 in enumerate(rows_in_1999):
                    row_1999_to_common[row_1999] = common_row

                # For each common sample, get its DTW neighbors and remap
                dtw_topk_list = []
                for common_row, row_1999 in enumerate(rows_in_1999):
                    old_neighbors = dtw_1999[row_1999]  # [k] indices into 1999
                    new_neighbors = row_1999_to_common[old_neighbors]
                    # Keep only valid (non -1) neighbors
                    valid_neighbors = new_neighbors[new_neighbors >= 0]
                    if len(valid_neighbors) >= topk:
                        dtw_topk_list.append(valid_neighbors[:topk])
                    else:
                        # Pad with -1 or recompute — for now pad with first valid
                        padded = np.full(topk, valid_neighbors[0] if len(valid_neighbors) > 0 else 0)
                        padded[:len(valid_neighbors)] = valid_neighbors
                        dtw_topk_list.append(padded)

                dtw_topk = np.stack(dtw_topk_list)

                # Count how many samples lost neighbors
                full_count = sum(1 for r in range(N_common)
                               if all(row_1999_to_common[dtw_1999[rows_in_1999[r]]] >= 0))
                print(f"  Reindexed: {full_count}/{N_common} samples retained all {topk} neighbors")

                # Save cache
                np.save(dtw_cache_path, dtw_topk)
                print(f"  Saved reindexed DTW → {dtw_cache_path}")
            else:
                print("  ERROR: common_valid contains samples not in 1999 set. Need full recompute.")
                feats_B_seq = torch.load(os.path.join(DATA_DIR, "feats_B_seq.pt"), weights_only=True)
                common_mask = torch.zeros(feats_B_seq.shape[0], dtype=torch.bool)
                for v in common_valid_sorted:
                    common_mask[v] = True
                dtw_topk = recompute_dtw_topk(feats_B_seq, common_mask, h, topk, device)
                np.save(dtw_cache_path, dtw_topk)
                del feats_B_seq
        else:
            print("  No existing DTW to reindex. Need full recompute.")
            feats_B_seq = torch.load(os.path.join(DATA_DIR, "feats_B_seq.pt"), weights_only=True)
            common_mask = torch.zeros(feats_B_seq.shape[0], dtype=torch.bool)
            for v in common_valid_sorted:
                common_mask[v] = True
            dtw_topk = recompute_dtw_topk(feats_B_seq, common_mask, h, topk, device)
            np.save(dtw_cache_path, dtw_topk)
            del feats_B_seq

    assert dtw_topk.shape == (N_common, topk), f"DTW shape mismatch: {dtw_topk.shape} vs ({N_common}, {topk})"

    # Build B-side mask and precompute HSIC_BB
    print("  Building B-side mask...")
    mask_B = build_mask_from_topk(dtw_topk, N_common, device)
    hsic_BB = hsic_unbiased(mask_B.clone(), mask_B.clone())
    print(f"  HSIC_BB = {hsic_BB.item():.6f}")

    # ── Step 3: Compute CKNNA per m per pooling strategy ─────────────────────
    print("\nStep 3: Computing CKNNA...")

    all_results = {}  # {pooling_strategy: {m: cknna_value}}
    mutual_knn_results = {}

    for m in m_values:
        if m not in valid_sets:
            continue

        row_indices = get_row_indices(m)
        T = m + 1

        # Try percam first, fall back to global mean
        percam_path = os.path.join(DATA_DIR, f"{args.model_prefix}-m{m}{args.dir_suffix}", "feats_A_percam.pt")
        global_path = os.path.join(DATA_DIR, f"{args.model_prefix}-m{m}{args.dir_suffix}", "feats_A.pt")

        pooled_feats = {}

        if os.path.exists(percam_path):
            percam = torch.load(percam_path, weights_only=True)  # [N_m, T, C, D]
            # Filter to common valid set
            percam_common = percam[row_indices]
            pooled_feats = pool_features(percam_common, m, percam_common.shape[2])
            del percam, percam_common

        if os.path.exists(global_path):
            global_feat = torch.load(global_path, weights_only=True).float()
            pooled_feats["global_mean"] = global_feat[row_indices]
            del global_feat

        print(f"\nm={m} (T={T}): {list(pooled_feats.keys())}")

        for strategy, feats in pooled_feats.items():
            assert feats.shape[0] == N_common, f"Shape mismatch: {feats.shape[0]} vs {N_common}"

            t0 = time.time()
            topk_A = compute_cosine_topk(feats, topk, device)
            mask_A = torch.zeros(N_common, N_common, dtype=torch.float32, device=device)
            mask_A.scatter_(1, topk_A, 1.0)

            cknna_val = compute_cknna(mask_A, mask_B)
            mknn_val = compute_mutual_knn(mask_A, mask_B, topk)
            dt = time.time() - t0

            if strategy not in all_results:
                all_results[strategy] = {}
                mutual_knn_results[strategy] = {}
            all_results[strategy][m] = cknna_val
            mutual_knn_results[strategy][m] = mknn_val

            # % change from m=0
            if 0 in all_results.get(strategy, {}):
                pct = (cknna_val - all_results[strategy][0]) / all_results[strategy][0] * 100
                pct_str = f" ({pct:+.1f}%)"
            else:
                pct_str = ""

            print(f"  {strategy}: CKNNA={cknna_val:.5f}  mknn={mknn_val:.5f}{pct_str}  [{dt:.1f}s]")

            del mask_A, topk_A
            torch.cuda.empty_cache()

    # ── Step 4: Save JSON ────────────────────────────────────────────────────
    out_json = os.path.join(OUT_DIR, f"cknna_vs_m_correct_h{h}_k{topk}{tag}.json")
    payload = {
        "_meta": {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "formula": "HSIC(mask_A, mask_B) / sqrt(HSIC(mask_A, mask_A) * HSIC(mask_B, mask_B))",
            "A_side": "cosine kNN on model features (binary mask)",
            "B_side": f"DTW kNN (h={h}, soft_cuda) on proprio trajectory (binary mask)",
            "note": "CORRECT formula matching run_dtw_cknna.py official pipeline",
            "m_values": m_values,
            "h": h,
            "topk": topk,
            "N_common": N_common,
            "model_prefix": args.model_prefix,
            "data_dir": DATA_DIR,
        },
        "cknna": {strategy: {str(m): v for m, v in d.items()}
                  for strategy, d in all_results.items()},
        "mutual_knn": {strategy: {str(m): v for m, v in d.items()}
                       for strategy, d in mutual_knn_results.items()},
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # ── Step 5: Plot ─────────────────────────────────────────────────────────
    # Plot all pooling strategies on one figure
    colors = {
        "cam1_anchor": "#e74c3c",
        "cam1_meanT": "#e67e22",
        "3cam_anchor": "#2980b9",
        "3cam_meanT": "#27ae60",
        "global_mean": "#8e44ad",
    }
    labels = {
        "cam1_anchor": "cam1 anchor (last frame)",
        "cam1_meanT": "cam1 mean-T",
        "3cam_anchor": "3cam anchor (last frame)",
        "3cam_meanT": "3cam mean-T",
        "global_mean": "global mean",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for strategy, d in all_results.items():
        ms = sorted(d.keys())
        scores = [d[m] for m in ms]
        color = colors.get(strategy, "#333333")
        label = labels.get(strategy, strategy)
        ax.plot(ms, scores, "o-", color=color, linewidth=2, markersize=7, label=label)
        # Annotate m=0 and m=64
        if ms:
            ax.annotate(f"{scores[0]:.4f}", (ms[0], scores[0]),
                       textcoords="offset points", xytext=(5, 8),
                       fontsize=8, color=color)
            ax.annotate(f"{scores[-1]:.4f}", (ms[-1], scores[-1]),
                       textcoords="offset points", xytext=(5, -12),
                       fontsize=8, color=color)

    ax.set_xlabel("Context frames m (m=0: single frame)", fontsize=12)
    ax.set_ylabel(f"CKNNA (DTW h={h}, k={topk})", fontsize=12)
    model_label = args.model_prefix.replace("-", " ").title()
    ax.set_title(f"Multi-Frame CKNNA: {model_label}\n"
                 f"DROID, N={N_common}, HSIC on binary kNN masks + DTW B-side",
                 fontsize=12)
    ax.set_xticks(m_values)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()

    out_png = os.path.join(OUT_DIR, f"cknna_vs_m_correct_h{h}_k{topk}{tag}.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {out_png}")

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"SUMMARY: {args.model_prefix}, h={h}, k={topk}, N={N_common}")
    print(f"{'='*70}")
    header = f"{'m':>4}"
    strategies = sorted(all_results.keys())
    for s in strategies:
        header += f"  {s:>14}"
    print(header)
    print("-" * len(header))
    for m in m_values:
        row = f"{m:>4}"
        for s in strategies:
            v = all_results.get(s, {}).get(m)
            if v is not None:
                pct = (v - all_results[s].get(0, v)) / all_results[s].get(0, v) * 100 if 0 in all_results[s] else 0
                row += f"  {v:.5f}({pct:+.0f}%)"
            else:
                row += f"  {'N/A':>14}"
        print(row)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
