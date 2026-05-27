"""
Frozen-VLM hypothesis plots for Group L (H_max=40, N=7762).

Same controlled comparison as exp_horizon/Frozen-VLM/ but using the
Group L BridgeV2 dataset instead of the H=75 sweep-dominated dataset.

Figures produced:
  frozen_vlm_hypothesis_confirmed.png -- 2x3 grid of raw vs fine-tuned pairs
  frozen_vlm_summary.png              -- all models on shared absolute y-axis
  fig_cknna_vs_sr_L_with_raw.png      -- CKNNA-vs-SR scatter with raw VLMs added
"""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKNNA_ROOT = os.environ.get("DATA_STORE", "/home/ubuntu/verl/starVLA/cknna")

CSV_PATH = os.path.join(
    CKNNA_ROOT, "record", "cknna_data_exp1v2_full_L",
    "finetune", "dtw_sym2_cuda_nowin",
    "results_groups_dtw_sym2_cuda_nowin.csv",
)
META_CSV = os.path.join(CKNNA_ROOT, "cknna_data", "performance", "model_metadata.csv")
OUT_DIR = SCRIPT_DIR

HORIZONS = [1, 3, 7, 15, 25, 40]

# (raw_model_key, ft_model_key) -- same pairs as H=75 experiment
PAIRS = [
    ("qwen25vl-3b-raw", "Qwen-GR00T-Bridge"),
    ("qwen25vl-3b-raw", "Qwen-GR00T-Bridge-RT-1"),
    ("qwen25vl-3b-raw", "Qwen-OFT-Bridge-RT-1"),
    ("qwen3vl-4b-raw",  "Qwen3VL-GR00T-Bridge-RT-1"),
    ("qwen3vl-4b-raw",  "Qwen3VL-OFT-Bridge-RT-1"),
    ("prismatic-raw",   "openvla-7b-bridge"),
]

RAW_MODELS = ["qwen25vl-3b-raw", "qwen3vl-4b-raw", "prismatic-raw"]

COLOR_RAW = "#2196F3"
COLOR_FT = "#E53935"

RAW_DISPLAY = {
    "qwen25vl-3b-raw": "Qwen2.5-VL-3B (raw)",
    "qwen3vl-4b-raw":  "Qwen3-VL-4B (raw)",
    "prismatic-raw":   "Prismatic VLM (raw)",
}

# Color card for scatter plot (matches plot_exp_horizon_L.py)
COLOR_CARD = {
    "StarVLA-Qwen2.5": {"color": "#1f77b4", "marker": "o"},
    "StarVLA-Qwen3":   {"color": "#aec7e8", "marker": "s"},
    "CogACT":          {"color": "#ff7f0e", "marker": "^"},
    "GR00T":           {"color": "#2ca02c", "marker": "D"},
    "OpenVLA":         {"color": "#d62728", "marker": "v"},
    "Pi0":             {"color": "#9467bd", "marker": "P"},
    "SpatialVLA":      {"color": "#8c564b", "marker": "X"},
    "RT-1-X":          {"color": "#7f7f7f", "marker": "*"},
    "Octo":            {"color": "#bcbd22", "marker": "h"},
    "RawVLM":          {"color": "#17becf", "marker": "D"},
}


def _load_display_names(csv_path):
    names = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["model_key"] not in names:
                names[row["model_key"]] = row["figure_label"]
    return names


_DISPLAY_NAMES = _load_display_names(META_CSV)


def display_name(key):
    if key in RAW_DISPLAY:
        return RAW_DISPLAY[key]
    return _DISPLAY_NAMES.get(key, key)


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def get_curve(rows, model, variant="imgtext"):
    data = {}
    for r in rows:
        if r["model"] == model and r["feats_A_variant"] == variant:
            data[int(r["horizon"])] = float(r["cknna_k10"])
    hs = sorted(data)
    return hs, [data[h] for h in hs]


def load_model_meta():
    meta = {}
    with open(META_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["finetune_on_bridge"] != "yes":
                continue
            meta[row["model_key"]] = {
                "sr":    float(row["success_rate"]),
                "group": row["group"],
                "label": row["figure_label"],
            }
    return meta


# Fixed y-axis range shared across all pair panels.
# Derived from global min/max of all models in PAIRS at HORIZONS (imgtext):
#   global min = 0.026  (qwen25vl-3b-raw at h=1)
#   global max = 0.171  (Qwen3VL-GR00T-Bridge-RT-1 at h=40)
PAIR_Y_LO = 0.018
PAIR_Y_HI = 0.185


def plot_pair(ax, rows, raw_model, ft_model, y_lo=PAIR_Y_LO, y_hi=PAIR_Y_HI):
    hs_raw, vals_raw = get_curve(rows, raw_model)
    hs_ft, vals_ft = get_curve(rows, ft_model)

    ax.plot(hs_raw, vals_raw, "o-", color=COLOR_RAW, linewidth=2.2,
            markersize=5, label=display_name(raw_model), zorder=3)
    ax.plot(hs_ft, vals_ft, "s-", color=COLOR_FT, linewidth=2.2,
            markersize=5, label=display_name(ft_model), zorder=3)

    ax.set_ylabel("CKNNA (imgtext, k=10)", fontsize=9)
    ax.set_xlabel("Horizon h", fontsize=9)

    if not vals_raw and not vals_ft:
        return
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks(HORIZONS)

    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.2)


def plot_absolute_summary(ax, rows):
    frozen_data = {}
    all_vals = []

    for model in RAW_MODELS:
        hs, vals = get_curve(rows, model)
        label = display_name(model)
        ax.plot(hs, vals, "o-", linewidth=1.8, markersize=4,
                alpha=0.85, color=COLOR_RAW, label=label)
        all_vals.extend(vals)
        for h, v in zip(hs, vals):
            frozen_data.setdefault(h, []).append(v)

    ft_keys = [ft for _, ft in PAIRS]
    ft_data = {}
    for model in ft_keys:
        hs_ft, vals_ft = get_curve(rows, model)
        ax.plot(hs_ft, vals_ft, "s-", linewidth=1.3, markersize=3,
                alpha=0.5, color=COLOR_FT, label=display_name(model))
        all_vals.extend(vals_ft)
        for h, v in zip(hs_ft, vals_ft):
            ft_data.setdefault(h, []).append(v)

    hs_common = sorted(set(frozen_data) & set(ft_data))
    if hs_common:
        mean_frozen = [np.mean(frozen_data[h]) for h in hs_common]
        mean_ft = [np.mean(ft_data[h]) for h in hs_common]
        ax.plot(hs_common, mean_frozen, "o-", color="#0D47A1", linewidth=3.5,
                markersize=7, label=f"Mean raw VLM (n={len(RAW_MODELS)})", zorder=5)
        ax.plot(hs_common, mean_ft, "s-", color="#B71C1C", linewidth=3.5,
                markersize=7, label=f"Mean VLA fine-tuned (n={len(ft_keys)})", zorder=5)

    if all_vals:
        y_lo = max(0.0, min(all_vals) - 0.02)
        y_hi = max(all_vals) + 0.025
        ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel("Horizon h", fontsize=10)
    ax.set_ylabel("CKNNA (imgtext, k=10)", fontsize=10)
    ax.set_xticks(HORIZONS)
    ax.set_title("All models  |  shared absolute y-axis", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower right", ncol=1)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=9)


def fig_cknna_vs_sr_with_raw(rows, model_meta):
    """Scatter: CKNNA vs SR, rows=horizons, cols=3 variants.
    VLA models plotted as colored markers, raw VLMs as cyan diamonds.
    Fixed axes across all panels.
    """
    data = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        data[r["model"]][r["feats_A_variant"]][int(r["horizon"])] = float(r["cknna_k10"])

    FA_VARIANTS = ["imgtext", "img", "txt"]
    FA_LABELS = {"imgtext": "img+txt", "img": "image only", "txt": "text only"}

    # Collect global axis range from VLA models only (for consistent comparison)
    all_cknna_vla, all_sr = [], []
    for model_key, meta in model_meta.items():
        if model_key not in data:
            continue
        for fa in FA_VARIANTS:
            if fa not in data[model_key]:
                continue
            for h in HORIZONS:
                if h in data[model_key][fa]:
                    all_cknna_vla.append(data[model_key][fa][h])
                    all_sr.append(meta["sr"])

    # Also include raw VLM CKNNA range
    all_cknna_raw = []
    for rm in RAW_MODELS:
        if rm not in data:
            continue
        for fa in FA_VARIANTS:
            if fa not in data[rm]:
                continue
            for h in HORIZONS:
                if h in data[rm][fa]:
                    all_cknna_raw.append(data[rm][fa][h])

    all_cknna = all_cknna_vla + all_cknna_raw
    if not all_cknna:
        print("WARNING: No data found for scatter plot.")
        return

    cknna_min, cknna_max = min(all_cknna), max(all_cknna)
    cknna_pad = (cknna_max - cknna_min) * 0.08
    x_lo, x_hi = cknna_min - cknna_pad, cknna_max + cknna_pad

    sr_min, sr_max = min(all_sr), max(all_sr)
    sr_pad = (sr_max - sr_min) * 0.08
    y_lo, y_hi = sr_min - sr_pad, sr_max + sr_pad

    n_h = len(HORIZONS)
    fig, axes = plt.subplots(n_h, 3, figsize=(13, 4 * n_h))
    if n_h == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"CKNNA [DTW] vs WidowX SR  (Group L, H_max=40, N=7762)\n"
        f"Colored markers = VLA fine-tuned  |  Cyan diamonds = raw pretrained VLM",
        fontsize=12, y=1.005,
    )

    for ri, h in enumerate(HORIZONS):
        for ci, fa in enumerate(FA_VARIANTS):
            ax = axes[ri, ci]

            xs_vla, ys_vla = [], []
            for model_key, meta in model_meta.items():
                if model_key not in data or fa not in data[model_key]:
                    continue
                if h not in data[model_key][fa]:
                    continue
                val = data[model_key][fa][h]
                c = COLOR_CARD[meta["group"]]["color"]
                m = COLOR_CARD[meta["group"]]["marker"]
                ax.scatter(val, meta["sr"], c=c, marker=m, s=80,
                           edgecolors="k", linewidths=0.4, zorder=3)
                ax.annotate(meta["label"], (val, meta["sr"]),
                            textcoords="offset points", xytext=(4, 2),
                            fontsize=5.5, color=c)
                xs_vla.append(val)
                ys_vla.append(meta["sr"])

            # Add raw VLMs as cyan diamonds (no SR -- plotted at y midpoint)
            sr_mid = (y_lo + y_hi) / 2
            raw_xs = []
            for rm in RAW_MODELS:
                if rm not in data or fa not in data[rm] or h not in data[rm][fa]:
                    continue
                val = data[rm][fa][h]
                ax.scatter(val, sr_mid, c=COLOR_CARD["RawVLM"]["color"],
                           marker=COLOR_CARD["RawVLM"]["marker"], s=100,
                           edgecolors="k", linewidths=0.6, zorder=5,
                           label=display_name(rm) if ri == 0 and ci == 0 else None)
                ax.annotate(display_name(rm), (val, sr_mid),
                            textcoords="offset points", xytext=(4, 2),
                            fontsize=5.5, color=COLOR_CARD["RawVLM"]["color"])
                raw_xs.append(val)

            if len(xs_vla) >= 3:
                rho, pval = spearmanr(xs_vla, ys_vla)
                p_str = f"{pval:.3f}" if pval >= 0.001 else f"{pval:.2e}"
                ax.text(0.04, 0.97, f"rho={rho:.3f}  p={p_str}\nn={len(xs_vla)}",
                        transform=ax.transAxes, fontsize=7, va="top",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                  edgecolor="#aaaaaa", alpha=0.85))

            if raw_xs:
                ax.axvline(np.mean(raw_xs), color=COLOR_CARD["RawVLM"]["color"],
                           linestyle=":", alpha=0.5, linewidth=1.2)

            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.set_xlabel("CKNNA k=10", fontsize=8)
            ax.set_ylabel("WidowX SR (%)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25)

            parts = []
            if ri == 0:
                parts.append(FA_LABELS[fa])
            parts.append(f"t+1 to t+{h}")
            ax.set_title("  |  ".join(parts), fontsize=8, pad=3)

    shown_groups = {meta["group"] for meta in model_meta.values()}
    patches = [mpatches.Patch(color=COLOR_CARD[g]["color"], label=g)
               for g in COLOR_CARD if g in shown_groups]
    patches.append(mpatches.Patch(color=COLOR_CARD["RawVLM"]["color"],
                                  label="Raw VLM (no SR, plotted at y midpoint)"))
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=7, bbox_to_anchor=(0.5, 0.0), framealpha=0.9)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_cknna_vs_sr_L_with_raw.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def print_raw_vlm_stats(rows):
    """Print CKNNA values for raw VLMs at all horizons."""
    print("\n=== Raw VLM CKNNA (imgtext, k=10, Group L H_max=40) ===")
    print(f"{'Model':<30s} " + " ".join(f"{'h='+str(h):>7s}" for h in HORIZONS))
    for rm in RAW_MODELS:
        hs, vals = get_curve(rows, rm)
        h_to_v = dict(zip(hs, vals))
        row_vals = [h_to_v.get(h, float("nan")) for h in HORIZONS]
        print(f"{display_name(rm):<30s} " + " ".join(f"{v:7.3f}" for v in row_vals))

    print("\n=== Comparison: raw VLM vs mean of paired FT models ===")
    for raw_m, ft_m in PAIRS:
        hs_raw, vals_raw = get_curve(rows, raw_m)
        hs_ft, vals_ft = get_curve(rows, ft_m)
        print(f"\n  {display_name(raw_m)} vs {display_name(ft_m)}")
        print(f"  {'h':>5s}  {'raw':>8s}  {'ft':>8s}  {'diff':>8s}")
        for h, v_raw in zip(hs_raw, vals_raw):
            v_ft = dict(zip(hs_ft, vals_ft)).get(h, float("nan"))
            diff = v_ft - v_raw
            print(f"  {h:>5d}  {v_raw:8.3f}  {v_ft:8.3f}  {diff:+8.3f}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_csv(CSV_PATH)

    models_in_csv = sorted(set(r["model"] for r in rows))
    print(f"Models in CSV ({len(models_in_csv)}): {models_in_csv}")

    missing_raw = [rm for rm in RAW_MODELS if rm not in models_in_csv]
    if missing_raw:
        print(f"WARNING: Raw VLM models missing from CSV: {missing_raw}")
        print("Run DTW CKNNA first. Aborting.")
        return

    model_meta = load_model_meta()

    # Figure 1: 2x3 pair comparison grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 10),
                             gridspec_kw={"hspace": 0.42, "wspace": 0.32})
    pair_axes = [axes[r][c] for r, c in
                 [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]]

    for ax, (raw_model, ft_model) in zip(pair_axes, PAIRS):
        ax.set_title(f"{display_name(ft_model)}  vs  {display_name(raw_model)}",
                     fontsize=10, fontweight="bold")
        plot_pair(ax, rows, raw_model, ft_model)

    fig.suptitle(
        "Frozen-VLM Hypothesis: Group L (H_max=40, N=7762)\n"
        "Each panel: same base architecture, only E2E fine-tuning differs.  "
        "Two VLM families: Qwen + Prismatic.",
        fontsize=13, fontweight="bold", y=1.01,
    )
    path = os.path.join(OUT_DIR, "frozen_vlm_hypothesis_confirmed.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Figure 2: absolute summary
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    plot_absolute_summary(ax2, rows)
    ax2.set_title(
        "Frozen-VLM: Group L (H_max=40, N=7762)  |  All Models Shared Y-axis\n"
        "Blue = frozen/raw backbone  |  Red = VLA fine-tuned",
        fontsize=12, fontweight="bold",
    )
    fig2.tight_layout()
    p2 = os.path.join(OUT_DIR, "frozen_vlm_summary.png")
    fig2.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {p2}")

    # Figure 3: CKNNA vs SR scatter with raw VLMs overlaid
    fig_cknna_vs_sr_with_raw(rows, model_meta)

    # Print stats
    print_raw_vlm_stats(rows)

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
