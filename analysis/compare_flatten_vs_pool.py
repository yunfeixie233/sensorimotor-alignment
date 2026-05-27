"""
compare_flatten_vs_pool.py

Computes CKNNA for proprio_seq using MEAN POOL (across time dim), then
compares against the existing FLATTEN results from cknna_50k_full_results.csv.

Produces side-by-side figures:
  fig_a_proprio_t_{suffix}.png          -- single-timestep (reference, same for both)
  fig_b_proprio_seq_flatten_{suffix}.png -- flatten (existing)
  fig_b_proprio_seq_pool_{suffix}.png    -- mean pool (new)

Usage:
    cd /home/ubuntu/verl/starVLA
    python cknna/record/scripts/compare_flatten_vs_pool.py --device cuda
"""

import argparse
import csv
import json
import os
import time

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.environ.get("RECORD_DIR", "/home/ubuntu/verl/starVLA/cknna/record")
RESULTS_CSV = os.path.join(RECORD_DIR, "cknna_50k_full_results.csv")
OUT_DIR = os.path.join(RECORD_DIR, "figures_50k_flatten_vs_pool")
os.makedirs(OUT_DIR, exist_ok=True)

CKNNA_ROOT = os.environ.get("DATA_STORE", "/home/ubuntu/verl/starVLA/cknna")
DATA_DIR = os.path.join(CKNNA_ROOT, "cknna_data_50k")

# ============================================================================
# Model registry (same as run_50k_cknna_computation.py)
# ============================================================================
MODELS_3WAY = [
    ("Qwen-GR00T-Bridge",       "Qwen-GR00T-Bridge"),
    ("Qwen-GR00T-Bridge-RT-1",  "Qwen-GR00T-Bridge-RT-1"),
    ("Qwen-OFT-Bridge-RT-1",    "Qwen-OFT-Bridge-RT-1"),
    ("Qwen3VL-GR00T-Bridge-RT-1", "Qwen3VL-GR00T-Bridge-RT-1"),
    ("Qwen3VL-OFT-Bridge-RT-1", "Qwen3VL-OFT-Bridge-RT-1"),
    ("cogact-small-bridge",     "cogact-small-bridge"),
    ("cogact-base-bridge",      "cogact-base-bridge"),
    ("cogact-large-bridge",     "cogact-large-bridge"),
    ("groot-n15-bridge",        "groot-n15-bridge"),
    ("groot-n16-bridge",        "groot-n16-bridge"),
    ("openvla-7b-bridge",       "openvla-7b-bridge"),
    ("openvla-7b-bridge-ft-200k", "openvla-7b-bridge-ft-200k"),
    ("pi0-lerobot-bridge",      "pi0-lerobot-bridge"),
    ("spatialvla-sft-bridge",   "spatialvla-sft-bridge"),
]

MODELS_IMGTEXT_ONLY = [
    ("rt1x-bridge",      "rt1x-bridge"),
    ("octo-base-bridge", "octo-base-bridge"),
]

SEQ_HORIZONS = [1, 3, 7, 15]
K_VALUES = [10]
K = 10

# ============================================================================
# CKNNA / Mutual k-NN (from run_50k_cknna_computation.py)
# ============================================================================
def _hsic_unbiased_lowmem(K_mat, L_mat):
    m = K_mat.shape[0]
    K_mat.fill_diagonal_(0)
    L_mat.fill_diagonal_(0)
    term1 = torch.tensor(0.0, device=K_mat.device, dtype=K_mat.dtype)
    chunk = 2000
    for i in range(0, m, chunk):
        end = min(i + chunk, m)
        term1 += (K_mat[i:end] * L_mat.T[i:end]).sum()
    term2 = K_mat.sum() * L_mat.sum() / ((m - 1) * (m - 2))
    k_col = K_mat.sum(dim=0)
    l_row = L_mat.sum(dim=1)
    term3 = 2.0 * (k_col @ l_row) / (m - 2)
    del k_col, l_row
    return (term1 + term2 - term3) / (m * (m - 3))


def _compute_topk_mask(sim_mat, topk, n, device):
    sim_orig = sim_mat.diagonal().clone()
    sim_mat.fill_diagonal_(float("-inf"))
    _, topk_idx = sim_mat.topk(topk, dim=1)
    sim_mat.diagonal().copy_(sim_orig)
    del sim_orig
    mask = torch.zeros(n, n, dtype=sim_mat.dtype, device=device)
    mask.scatter_(1, topk_idx, 1.0)
    del topk_idx
    return mask


def cknna_lowmem(feats_A_norm, feats_B_norm, topk):
    n = feats_A_norm.shape[0]
    device = feats_A_norm.device
    sim_A = feats_A_norm @ feats_A_norm.T
    mask_A = _compute_topk_mask(sim_A, topk, n, device)
    del sim_A
    sim_B = feats_B_norm @ feats_B_norm.T
    mask_B = _compute_topk_mask(sim_B, topk, n, device)
    del sim_B
    hsic_AB = _hsic_unbiased_lowmem(mask_A.clone(), mask_B.clone())
    hsic_AA = _hsic_unbiased_lowmem(mask_A.clone(), mask_A.clone())
    hsic_BB = _hsic_unbiased_lowmem(mask_B.clone(), mask_B.clone())
    del mask_A, mask_B
    denom = (hsic_AA * hsic_BB).clamp(min=1e-10).sqrt()
    return (hsic_AB / denom).item()


def mutual_knn_lowmem(feats_A_norm, feats_B_norm, topk):
    n = feats_A_norm.shape[0]
    device = feats_A_norm.device
    sim_A = feats_A_norm @ feats_A_norm.T
    mask_A = _compute_topk_mask(sim_A, topk, n, device)
    del sim_A
    sim_B = feats_B_norm @ feats_B_norm.T
    mask_B = _compute_topk_mask(sim_B, topk, n, device)
    del sim_B
    intersection = (mask_A * mask_B).sum()
    union = topk * n
    score = (intersection / union).item()
    del mask_A, mask_B
    return score


def compute_metrics(feats_A_norm, feats_B_norm, k_values, device):
    feats_A_norm = feats_A_norm.to(device)
    feats_B_norm = feats_B_norm.to(device)
    result = {}
    for k in k_values:
        t0 = time.time()
        c = cknna_lowmem(feats_A_norm, feats_B_norm, k)
        m = mutual_knn_lowmem(feats_A_norm, feats_B_norm, k)
        dt = time.time() - t0
        print(f"    k={k}: CKNNA={c:.5f}  mutualKNN={m:.5f}  [{dt:.1f}s]")
        result[f"cknna_k{k}"] = round(c, 6)
        result[f"mutual_knn_k{k}"] = round(m, 6)
    return result


def norm(t, device):
    return F.normalize(t.to(device), p=2, dim=-1)


# ============================================================================
# Phase 1: Compute mean-pool CKNNA for proprio_seq
# ============================================================================
def compute_meanpool_results(data_dir, device):
    with open(os.path.join(data_dir, "metadata.json")) as f:
        meta = json.load(f)
    task_descriptions = meta["task_descriptions"]
    valid_mask = torch.tensor(
        [bool(t and t.strip()) for t in task_descriptions], dtype=torch.bool
    )
    N_valid = valid_mask.sum().item()
    print(f"N_valid = {N_valid}")

    proprio_seq_raw = torch.load(
        os.path.join(data_dir, "feats_B_seq.pt"), weights_only=True
    ).float()
    proprio_seq = proprio_seq_raw[valid_mask]
    del proprio_seq_raw
    print(f"proprio_seq shape: {tuple(proprio_seq.shape)}")

    # Precompute mean-pooled targets at each horizon
    # proprio_seq[:, 0] = t, [:, 1] = t+1, ..., [:, 15] = t+15
    # Slice future only: [1 : h+1], then MEAN over time dim
    proprio_pool_norm = {}
    for h in SEQ_HORIZONS:
        ps = proprio_seq[:, 1:h + 1, :].mean(dim=1)  # [N, 7]
        proprio_pool_norm[h] = norm(ps, device)
        print(f"  h={h}: mean-pooled shape {tuple(ps.shape)}")

    del proprio_seq

    all_models = MODELS_3WAY + MODELS_IMGTEXT_ONLY
    csv_rows = []

    for model_key, model_dir_name in all_models:
        model_dir = os.path.join(data_dir, model_dir_name)
        feats_A_path = os.path.join(model_dir, "feats_A.pt")
        if not os.path.exists(feats_A_path):
            print(f"\n[SKIP] {model_key}: feats_A.pt not found")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {model_key}")
        print(f"{'='*60}")

        is_3way = (model_key, model_dir_name) in MODELS_3WAY

        # Load feats_A variants
        fa_imgtext_raw = torch.load(feats_A_path, weights_only=True).float()
        fa_imgtext = fa_imgtext_raw[valid_mask]
        fa_imgtext_norm = norm(fa_imgtext, device)
        feats_A_variants = {"imgtext": fa_imgtext_norm}
        del fa_imgtext_raw, fa_imgtext

        if is_3way:
            img_path = os.path.join(model_dir, "feats_A_img.pt")
            if os.path.exists(img_path):
                fa_img = torch.load(img_path, weights_only=True).float()[valid_mask]
                feats_A_variants["img"] = norm(fa_img, device)
                del fa_img
            txt_path = os.path.join(model_dir, "feats_A_txt.pt")
            if os.path.exists(txt_path):
                fa_txt = torch.load(txt_path, weights_only=True).float()[valid_mask]
                feats_A_variants["txt"] = norm(fa_txt, device)
                del fa_txt

        for variant_name, fa_norm in feats_A_variants.items():
            print(f"\n  feats_A variant: {variant_name}")
            for h in SEQ_HORIZONS:
                print(f"  -- proprio_seq_pool h={h}")
                metrics = compute_metrics(
                    fa_norm, proprio_pool_norm[h], K_VALUES, device
                )
                csv_rows.append({
                    "model": model_key,
                    "feats_A_variant": variant_name,
                    "feats_B_type": "proprio_seq_pool",
                    "horizon": h,
                    **metrics,
                })

        for v in feats_A_variants.values():
            del v
        torch.cuda.empty_cache()

    return csv_rows


# ============================================================================
# Phase 2: Load existing flatten results from CSV
# ============================================================================
def load_flatten_csv():
    rows = []
    with open(RESULTS_CSV) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


# ============================================================================
# Phase 3: Plotting
# ============================================================================
MODEL_META = {
    "Qwen-GR00T-Bridge":         {"sr": 71.4, "group": "StarVLA-Qwen2.5", "label": "Qwen2.5-GR00T-Br"},
    "Qwen-GR00T-Bridge-RT-1":    {"sr": 63.6, "group": "StarVLA-Qwen2.5", "label": "Qwen2.5-GR00T"},
    "Qwen-OFT-Bridge-RT-1":      {"sr": 44.2, "group": "StarVLA-Qwen2.5", "label": "Qwen2.5-OFT"},
    "Qwen3VL-GR00T-Bridge-RT-1": {"sr": 65.3, "group": "StarVLA-Qwen3",   "label": "Qwen3-GR00T"},
    "Qwen3VL-OFT-Bridge-RT-1":   {"sr": 42.7, "group": "StarVLA-Qwen3",   "label": "Qwen3-OFT"},
    "cogact-small-bridge":        {"sr": 51.0, "group": "CogACT",          "label": "CogACT-S"},
    "cogact-base-bridge":         {"sr": 51.3, "group": "CogACT",          "label": "CogACT-B"},
    "cogact-large-bridge":        {"sr": 58.3, "group": "CogACT",          "label": "CogACT-L"},
    "groot-n15-bridge":           {"sr": 36.5, "group": "GR00T",           "label": "GR00T-N1.5"},
    "groot-n16-bridge":           {"sr": 57.1, "group": "GR00T",           "label": "GR00T-N1.6"},
    "openvla-7b-bridge":          {"sr":  1.0, "group": "OpenVLA",         "label": "OpenVLA-7B"},
    "openvla-7b-bridge-ft-200k":  {"sr": 10.4, "group": "OpenVLA",         "label": "OpenVLA-ft"},
    "pi0-lerobot-bridge":         {"sr": 47.9, "group": "Pi0",             "label": "Pi0"},
    "spatialvla-sft-bridge":      {"sr": 42.7, "group": "SpatialVLA",      "label": "SpatialVLA"},
    "rt1x-bridge":                {"sr":  0.0, "group": "RT-1-X",          "label": "RT-1-X"},
    "octo-base-bridge":           {"sr": 20.3, "group": "Octo",            "label": "Octo"},
}

EXCLUDE_ALWAYS = {"openvla-7b-bridge"}
EXCLUDE_ONLYVLA = {"rt1x-bridge", "octo-base-bridge"}

GROUP_COLORS = {
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

GROUP_MARKERS = {
    "StarVLA-Qwen2.5": "o",
    "StarVLA-Qwen3":   "s",
    "CogACT":          "^",
    "GR00T":           "D",
    "OpenVLA":         "v",
    "Pi0":             "P",
    "SpatialVLA":      "X",
    "RT-1-X":          "*",
    "Octo":            "h",
}

FA_VARIANTS = ["imgtext", "img", "txt"]
FA_LABELS = {"imgtext": "img+txt", "img": "image only", "txt": "text only"}


def get_val(rows, model, fa_var, fb_type, horizon):
    col = f"cknna_k{K}"
    for r in rows:
        if (r["model"] == model and
                r["feats_A_variant"] == fa_var and
                r["feats_B_type"] == fb_type and
                int(r["horizon"]) == horizon and
                col in r):
            v = r[col]
            return float(v) if v else None
    return None


def draw_panel(ax, rows, fa_var, fb_type, horizon,
               exclude=None, row_label="", col_label=""):
    exclude = exclude or set()
    xs, ys = [], []
    for model_key, meta in MODEL_META.items():
        if model_key in exclude:
            continue
        val = get_val(rows, model_key, fa_var, fb_type, horizon)
        if val is None:
            continue
        c = GROUP_COLORS[meta["group"]]
        m = GROUP_MARKERS[meta["group"]]
        ax.scatter(val, meta["sr"], c=c, marker=m, s=80,
                   edgecolors="k", linewidths=0.4, zorder=3)
        ax.annotate(meta["label"], (val, meta["sr"]),
                    textcoords="offset points", xytext=(4, 2),
                    fontsize=5.5, color=c)
        xs.append(val)
        ys.append(meta["sr"])

    if len(xs) >= 3:
        rho, pval = spearmanr(xs, ys)
        p_str = f"{pval:.3f}" if pval >= 0.001 else f"{pval:.2e}"
        ax.text(0.04, 0.97, f"rho={rho:.3f}  p={p_str}\nn={len(xs)}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#aaaaaa", alpha=0.85))

    ax.set_xlabel(f"CKNNA k={K}", fontsize=8)
    ax.set_ylabel("WidowX SR (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25)

    parts = []
    if col_label:
        parts.append(col_label)
    if row_label:
        parts.append(row_label)
    if parts:
        ax.set_title("  |  ".join(parts), fontsize=8, pad=3)


def add_legend(fig, exclude=None):
    exclude = exclude or set()
    shown_groups = {meta["group"] for key, meta in MODEL_META.items()
                    if key not in exclude}
    patches = [mpatches.Patch(color=GROUP_COLORS[g], label=g)
               for g in GROUP_COLORS if g in shown_groups]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=7, bbox_to_anchor=(0.5, 0.0), framealpha=0.9)


def make_side_by_side(flatten_rows, pool_rows, title_prefix, stem, exclude):
    """6 columns: 3 for flatten (imgtext/img/txt) + 3 for pool (imgtext/img/txt).
    Rows = horizons."""
    n_h = len(SEQ_HORIZONS)
    fig, axes = plt.subplots(n_h, 6, figsize=(26, 4 * n_h))
    fig.suptitle(f"{title_prefix}: FLATTEN (left 3) vs MEAN POOL (right 3)",
                 fontsize=13, y=1.005)

    for ri, h in enumerate(SEQ_HORIZONS):
        for ci, fa in enumerate(FA_VARIANTS):
            # Flatten (left 3 columns)
            draw_panel(axes[ri][ci], flatten_rows, fa, "proprio_seq", h,
                       exclude=exclude,
                       row_label=f"t+1..t+{h}",
                       col_label=(f"FLATTEN | {FA_LABELS[fa]}" if ri == 0
                                  else FA_LABELS[fa]))
            # Pool (right 3 columns)
            draw_panel(axes[ri][ci + 3], pool_rows, fa, "proprio_seq_pool", h,
                       exclude=exclude,
                       row_label=f"t+1..t+{h}",
                       col_label=(f"MEAN POOL | {FA_LABELS[fa]}" if ri == 0
                                  else FA_LABELS[fa]))

    # Align x-axis ranges across flatten/pool columns for fair visual comparison
    for ri in range(n_h):
        for ci in range(3):
            ax_f = axes[ri][ci]
            ax_p = axes[ri][ci + 3]
            xmin = min(ax_f.get_xlim()[0], ax_p.get_xlim()[0])
            xmax = max(ax_f.get_xlim()[1], ax_p.get_xlim()[1])
            ax_f.set_xlim(xmin, xmax)
            ax_p.set_xlim(xmin, xmax)

    add_legend(fig, exclude)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, stem)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def make_separate_figs(rows, fb_type, label_tag, stem_tag, exclude):
    """4-row x 3-col figure for one method (flatten or pool)."""
    n_h = len(SEQ_HORIZONS)
    fig, axes = plt.subplots(n_h, 3, figsize=(13, 4 * n_h))
    fig.suptitle(f"proprio seq ({label_tag})", fontsize=12, y=1.005)
    for ri, h in enumerate(SEQ_HORIZONS):
        for ci, fa in enumerate(FA_VARIANTS):
            draw_panel(axes[ri][ci], rows, fa, fb_type, h,
                       exclude=exclude,
                       row_label=f"t+1..t+{h}",
                       col_label=FA_LABELS[fa] if ri == 0 else "")
    add_legend(fig, exclude)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, stem_tag)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_compute", action="store_true",
                        help="Skip CKNNA computation; use existing pool CSV")
    args = parser.parse_args()

    pool_csv_path = os.path.join(RECORD_DIR, "cknna_50k_proprio_seq_pool.csv")

    if not args.skip_compute:
        data_dir = os.path.abspath(DATA_DIR)
        print(f"Data dir: {data_dir}")
        print(f"Device: {args.device}")
        pool_rows_dicts = compute_meanpool_results(data_dir, args.device)

        fieldnames = list(pool_rows_dicts[0].keys())
        with open(pool_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(pool_rows_dicts)
        print(f"\nSaved pool CSV: {pool_csv_path}  ({len(pool_rows_dicts)} rows)")

    # Load both CSVs
    print(f"\nLoading flatten CSV: {RESULTS_CSV}")
    flatten_rows = load_flatten_csv()
    print(f"  {len(flatten_rows)} rows")

    print(f"Loading pool CSV: {pool_csv_path}")
    pool_rows = []
    with open(pool_csv_path) as f:
        for r in csv.DictReader(f):
            pool_rows.append(r)
    print(f"  {len(pool_rows)} rows")

    EXCLUDE_FT = {"openvla-7b-bridge-ft-200k"}
    VERSIONS = [
        ("all",          EXCLUDE_ALWAYS),
        ("all_noft",     EXCLUDE_ALWAYS | EXCLUDE_FT),
        ("onlyvla",      EXCLUDE_ALWAYS | EXCLUDE_ONLYVLA),
        ("onlyvla_noft", EXCLUDE_ALWAYS | EXCLUDE_ONLYVLA | EXCLUDE_FT),
    ]

    for suffix, exclude in VERSIONS:
        print(f"\n--- version: {suffix} ---")

        # Side-by-side 6-col figure
        make_side_by_side(
            flatten_rows, pool_rows,
            f"(b) proprio seq",
            f"fig_b_proprio_seq_comparison_{suffix}.png",
            exclude,
        )

        # Individual figures
        make_separate_figs(
            flatten_rows, "proprio_seq", "flatten",
            f"fig_b_proprio_seq_flatten_{suffix}.png",
            exclude,
        )
        make_separate_figs(
            pool_rows, "proprio_seq_pool", "mean pool",
            f"fig_b_proprio_seq_pool_{suffix}.png",
            exclude,
        )

    print(f"\nDone. Figures in {OUT_DIR}/")


if __name__ == "__main__":
    main()
