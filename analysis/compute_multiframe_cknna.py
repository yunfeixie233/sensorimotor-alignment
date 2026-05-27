"""
Phase 3 + Plot: Multi-frame CKNNA for Exp A (joint temporal attention)

Computes CKNNA(m) for each completed m value using:
  - A-side: LingBot-VA joint features (norm_out, all-token pool)  → (N, 3072)
  - B-side: flattened h-step action trajectory                    → (N, h*7)

Both sides are L2-normalized before cosine k-NN.

Usage:
  python analysis/compute_multiframe_cknna.py --m_values 0 1 2 4 8 --h 75 --topk 10
  python analysis/compute_multiframe_cknna.py --m_values 0 1 2 4   --h 75  # partial
"""

import argparse
import json
import os
import sys
import datetime

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = "/lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid_multiframe"
OUT_DIR  = "/home/ubuntu/vla/cknna_project/record/multiframe_cknna"
os.makedirs(OUT_DIR, exist_ok=True)


# ── CKNNA core ────────────────────────────────────────────────────────────────

def hsic_unbiased(K, L):
    m = K.shape[0]
    K_t = K.clone().fill_diagonal_(0)
    L_t = L.clone().fill_diagonal_(0)
    t1 = torch.sum(K_t * L_t.T)
    t2 = torch.sum(K_t) * torch.sum(L_t) / ((m - 1) * (m - 2))
    t3 = 2 * torch.sum(torch.mm(K_t, L_t)) / (m - 2)
    return (t1 + t2 - t3) / (m * (m - 3))


def cknna(feats_A, feats_B, topk=10):
    """CKNNA between L2-normalized feats_A (N,D) and feats_B (N,D')."""
    K = feats_A @ feats_A.T
    L = feats_B @ feats_B.T
    n = feats_A.shape[0]
    device = feats_A.device

    def _sim(Km, Lm, k):
        Kh = Km.clone().fill_diagonal_(float("-inf"))
        Lh = Lm.clone().fill_diagonal_(float("-inf"))
        _, ki = torch.topk(Kh, k, dim=1)
        _, li = torch.topk(Lh, k, dim=1)
        mK = torch.zeros(n, n, device=device).scatter_(1, ki, 1)
        mL = torch.zeros(n, n, device=device).scatter_(1, li, 1)
        mask = mK * mL
        return hsic_unbiased(mask * Km, mask * Lm)

    sim_kl = _sim(K, L, topk)
    sim_kk = _sim(K, K, topk)
    sim_ll = _sim(L, L, topk)
    return sim_kl.item() / (torch.sqrt(sim_kk * sim_ll) + 1e-6).item()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m_values", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--h", type=int, default=75,
                        help="Horizon: number of future action steps for B-side")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--also_h1", action="store_true",
                        help="Also compute CKNNA at h=1 for cross-check")
    args = parser.parse_args()

    device = args.device
    m_values = sorted(args.m_values)
    h_values = [1, args.h] if args.also_h1 else [args.h]

    print(f"Multi-Frame CKNNA  |  m={m_values}  |  h={h_values}  |  k={args.topk}")
    print(f"Data dir: {DATA_DIR}")
    print()

    # ── Load and prepare B-side ───────────────────────────────────────────────
    feats_B_seq_path = os.path.join(DATA_DIR, "feats_B_seq.pt")
    print(f"Loading B-side sequence: {feats_B_seq_path}")
    feats_B_seq = torch.load(feats_B_seq_path, weights_only=True).float()  # (N, T_max, 7)
    N_total = feats_B_seq.shape[0]
    T_max = feats_B_seq.shape[1]
    print(f"  feats_B_seq: {tuple(feats_B_seq.shape)}  (N={N_total}, T_max={T_max})")

    # Detect valid samples per horizon (no padding = consecutive frames differ)
    def compute_max_valid_horizon(seq):
        diff = (seq[:, 1:] != seq[:, :-1]).any(dim=-1)  # (N, T_max-1)
        positions = torch.arange(diff.shape[1]).unsqueeze(0)
        masked_pos = positions * diff.long()
        has_any = diff.any(dim=1)
        rightmost = masked_pos.max(dim=1).values
        max_avail = rightmost + 2
        max_avail[~has_any] = 1
        return (max_avail - 1).int()

    max_valid_h = compute_max_valid_horizon(feats_B_seq)

    def get_valid_mask(h):
        return max_valid_h >= h

    b_side_norms = {}
    for h in h_values:
        mask = get_valid_mask(h)
        N_h = mask.sum().item()
        print(f"  h={h}: {N_h}/{N_total} valid samples")

        # Flatten h-step trajectory: (N_h, h*7)
        traj = feats_B_seq[mask, 1:h + 1, :].reshape(N_h, h * 7)
        traj_norm = F.normalize(traj.float(), p=2, dim=-1).to(device)
        b_side_norms[h] = (mask, traj_norm)
        print(f"  h={h}: B-side tensor {tuple(traj_norm.shape)}")

    # ── Compute CKNNA per m per h ─────────────────────────────────────────────
    results = {h: {} for h in h_values}

    for m in m_values:
        feats_A_path = os.path.join(
            DATA_DIR, f"lingbot-va-base-joint-m{m}", "feats_A.pt"
        )
        if not os.path.exists(feats_A_path):
            print(f"  m={m}: feats_A.pt NOT FOUND — skipping")
            continue

        feats_A = torch.load(feats_A_path, weights_only=True).float()
        print(f"\nm={m}  feats_A: {tuple(feats_A.shape)}")

        for h in h_values:
            mask, b_norm = b_side_norms[h]
            N_h = mask.sum().item()

            # Filter A-side to same valid subset as B-side
            a_norm = F.normalize(feats_A[mask], p=2, dim=-1).to(device)
            assert a_norm.shape[0] == N_h

            score = cknna(a_norm, b_norm, topk=args.topk)
            results[h][m] = score
            print(f"  h={h}: CKNNA(m={m}) = {score:.6f}  (N={N_h})")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_json = os.path.join(OUT_DIR, f"cknna_vs_m_h{args.h}_k{args.topk}.json")
    payload = {
        "_meta": {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "m_values": m_values,
            "h": args.h,
            "topk": args.topk,
            "model": "lingbot-va-base-joint",
            "data_dir": DATA_DIR,
        },
        "results": {str(h): {str(m): v for m, v in d.items()}
                    for h, d in results.items()},
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    for h in h_values:
        d = results[h]
        if not d:
            continue
        ms = sorted(d.keys())
        scores = [d[m] for m in ms]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(ms, scores, "o-", color="#1f77b4", linewidth=2, markersize=8,
                label="LingBot-VA (joint, temporal attn ON)")
        for m, s in zip(ms, scores):
            ax.annotate(f"{s:.4f}", (m, s), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9)

        ax.set_xlabel("Number of context frames m\n(m=0: single frame; m=N: N preceding frames)", fontsize=11)
        ax.set_ylabel(f"CKNNA (h={h}, k={args.topk})", fontsize=11)
        N_h = b_side_norms[h][0].sum().item()
        ax.set_title(f"Multi-Frame CKNNA: LingBot-VA Exp A\n"
                     f"Context frames vs action alignment (DROID, N={N_h})",
                     fontsize=12)
        ax.set_xticks(ms)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        # Add annotation about m=0 being the single-frame baseline
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(0.1, min(scores) * 0.98, "single-frame\nbaseline",
                fontsize=8, color="gray", va="top")

        fig.tight_layout()
        out_png = os.path.join(OUT_DIR, f"cknna_vs_m_h{h}_k{args.topk}.png")
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved → {out_png}")

    print("\nDone.")


if __name__ == "__main__":
    main()
