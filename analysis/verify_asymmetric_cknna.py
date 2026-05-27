"""
verify_asymmetric_cknna.py

Compares two CKNNA formulations on the SAME flatten data
(where continuous kernels exist for both sides):

  (1) Reference:   HSIC(mask*K, mask*L) / sqrt(HSIC(mA*K,mA*K) * HSIC(mB*L,mB*L))
  (2) Asymmetric:  HSIC(mask*K, mask*1) / sqrt(HSIC(mA*K,mA*K) * HSIC(mB,mB))

This isolates the effect of using binary B-side vs continuous B-side,
independent of DTW vs flatten or padding differences.

Usage:
    cd /home/ubuntu/verl/starVLA
    python cknna/record/scripts/verify_asymmetric_cknna.py --device cuda
"""

import json
import os
import time

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKNNA_ROOT = os.environ.get("DATA_STORE", "/home/ubuntu/verl/starVLA/cknna")
DATA_DIR = os.path.abspath(os.path.join(CKNNA_ROOT, "cknna_data_50k"))

MODELS = [
    ("Qwen-GR00T-Bridge",         "Qwen-GR00T-Bridge",         71.4),
    ("Qwen-GR00T-Bridge-RT-1",    "Qwen-GR00T-Bridge-RT-1",    63.6),
    ("Qwen-OFT-Bridge-RT-1",      "Qwen-OFT-Bridge-RT-1",      44.2),
    ("Qwen3VL-GR00T-Bridge-RT-1", "Qwen3VL-GR00T-Bridge-RT-1", 65.3),
    ("Qwen3VL-OFT-Bridge-RT-1",   "Qwen3VL-OFT-Bridge-RT-1",   42.7),
    ("cogact-small-bridge",        "cogact-small-bridge",        51.0),
    ("cogact-base-bridge",         "cogact-base-bridge",         51.3),
    ("cogact-large-bridge",        "cogact-large-bridge",        58.3),
    ("groot-n15-bridge",           "groot-n15-bridge",           36.5),
    ("groot-n16-bridge",           "groot-n16-bridge",           57.1),
    ("openvla-7b-bridge-ft-200k",  "openvla-7b-bridge-ft-200k", 10.4),
    ("pi0-lerobot-bridge",         "pi0-lerobot-bridge",        47.9),
    ("spatialvla-sft-bridge",      "spatialvla-sft-bridge",     42.7),
    ("rt1x-bridge",                "rt1x-bridge",                0.0),
    ("octo-base-bridge",           "octo-base-bridge",          20.3),
]

SEQ_HORIZONS = [1, 3, 7, 15]
TOPK = 10


def _hsic_unbiased(K, L):
    m = K.shape[0]
    K = K.clone().fill_diagonal_(0)
    L = L.clone().fill_diagonal_(0)
    term1 = (K * L.T).sum()
    term2 = K.sum() * L.sum() / ((m - 1) * (m - 2))
    term3 = 2.0 * (K.sum(dim=0) @ L.sum(dim=1)) / (m - 2)
    return (term1 + term2 - term3) / (m * (m - 3))


def _topk_mask(sim_mat, topk):
    n = sim_mat.shape[0]
    sim_tmp = sim_mat.clone().fill_diagonal_(float("-inf"))
    _, idx = sim_tmp.topk(topk, dim=1)
    mask = torch.zeros_like(sim_mat)
    mask.scatter_(1, idx, 1.0)
    return mask


def cknna_reference(K, L, topk):
    """Full reference: HSIC(mask*K, mask*L) normalized."""
    mask_K = _topk_mask(K, topk)
    mask_L = _topk_mask(L, topk)
    intersection = mask_K * mask_L

    sim_kl = _hsic_unbiased(intersection * K, intersection * L)

    mask_A_self = _topk_mask(K, topk)
    sim_kk = _hsic_unbiased(mask_A_self * K, mask_A_self * K)

    mask_B_self = _topk_mask(L, topk)
    sim_ll = _hsic_unbiased(mask_B_self * L, mask_B_self * L)

    return (sim_kl / (torch.sqrt(sim_kk * sim_ll) + 1e-6)).item()


def cknna_asymmetric(K, L, topk):
    """Asymmetric: sim-weighted A-side, binary B-side.
    This is what the DTW oricknna code does."""
    mask_K = _topk_mask(K, topk)
    mask_L = _topk_mask(L, topk)
    intersection = mask_K * mask_L

    hsic_AB = _hsic_unbiased(intersection * K, intersection.clone())

    mask_A_self = _topk_mask(K, topk)
    hsic_AA = _hsic_unbiased(mask_A_self * K, (mask_A_self * K).clone())

    hsic_BB = _hsic_unbiased(mask_L.clone(), mask_L.clone())

    denom = (hsic_AA * hsic_BB).clamp(min=1e-10).sqrt()
    return (hsic_AB / denom).item()


def norm(t, device):
    return F.normalize(t.to(device), p=2, dim=-1)


def compute_max_valid_horizon(feats_B_seq):
    diff = (feats_B_seq[:, 1:] != feats_B_seq[:, :-1]).any(dim=-1)
    positions = torch.arange(diff.shape[1]).unsqueeze(0)
    masked_pos = positions * diff.long()
    has_any = diff.any(dim=1)
    rightmost = masked_pos.max(dim=1).values
    max_avail = rightmost + 2
    max_avail[~has_any] = 1
    return (max_avail - 1).int()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = args.device

    with open(os.path.join(DATA_DIR, "metadata.json")) as f:
        meta = json.load(f)
    valid_mask = torch.tensor(
        [bool(t and t.strip()) for t in meta["task_descriptions"]],
        dtype=torch.bool,
    )

    proprio_seq_full = torch.load(
        os.path.join(DATA_DIR, "feats_B_seq.pt"), weights_only=True
    ).float()

    max_valid_h = compute_max_valid_horizon(proprio_seq_full)
    seq_masks = {}
    for h in SEQ_HORIZONS:
        seq_masks[h] = valid_mask & (max_valid_h >= h)

    print(f"{'Model':<30} {'h':>3} {'Reference':>10} {'Asymmetric':>10} {'Diff':>8} {'RelDiff%':>8}")
    print("-" * 80)

    results_by_h = {h: {"ref": [], "asym": [], "sr": []} for h in SEQ_HORIZONS}

    for model_key, model_dir_name, sr in MODELS:
        fa_path = os.path.join(DATA_DIR, model_dir_name, "feats_A.pt")
        if not os.path.exists(fa_path):
            continue

        fa_full = torch.load(fa_path, weights_only=True).float()

        for h in SEQ_HORIZONS:
            mask = seq_masks[h]
            N_h = mask.sum().item()

            fa_h = norm(fa_full[mask], device)
            K = fa_h @ fa_h.T

            ps = proprio_seq_full[mask][:, 1:h + 1, :].reshape(N_h, -1)
            fb_h = norm(ps, device)
            L = fb_h @ fb_h.T
            del fa_h, fb_h, ps

            t0 = time.time()
            ref_val = cknna_reference(K, L, TOPK)
            asym_val = cknna_asymmetric(K, L, TOPK)
            dt = time.time() - t0

            diff = asym_val - ref_val
            rel_diff = 100 * diff / (abs(ref_val) + 1e-10)

            print(f"{model_key:<30} {h:>3} {ref_val:>10.5f} {asym_val:>10.5f} "
                  f"{diff:>+8.5f} {rel_diff:>+7.1f}%  [{dt:.1f}s]")

            results_by_h[h]["ref"].append(ref_val)
            results_by_h[h]["asym"].append(asym_val)
            results_by_h[h]["sr"].append(sr)

            del K, L
            torch.cuda.empty_cache()

        del fa_full
        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("SUMMARY: Spearman rho (CKNNA vs WidowX SR)")
    print(f"{'Horizon':>8} {'rho_ref':>10} {'p_ref':>10} {'rho_asym':>10} {'p_asym':>10} {'rho_diff':>10}")
    print("-" * 60)

    for h in SEQ_HORIZONS:
        d = results_by_h[h]
        if len(d["ref"]) < 3:
            continue
        rho_ref, p_ref = spearmanr(d["ref"], d["sr"])
        rho_asym, p_asym = spearmanr(d["asym"], d["sr"])
        print(f"{'t+1..t+' + str(h):>8} {rho_ref:>10.4f} {p_ref:>10.4f} "
              f"{rho_asym:>10.4f} {p_asym:>10.4f} {rho_asym - rho_ref:>+10.4f}")

    print("\nCorrelation between reference and asymmetric values:")
    for h in SEQ_HORIZONS:
        d = results_by_h[h]
        if len(d["ref"]) < 3:
            continue
        rho, p = spearmanr(d["ref"], d["asym"])
        print(f"  h={h}: Spearman(ref, asym) = {rho:.4f}  p={p:.4e}")


if __name__ == "__main__":
    main()
