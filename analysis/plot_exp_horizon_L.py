"""
Plot Exp 1v2 results for the Bridge V2 subset (Group L only).
Re-draws fig_cknna_vs_sr_L.png and fig_cknna_vs_horizon.png
with fixed global axis ranges across all subplots.

No mention of "Group L" anywhere - referred to as "Bridge V2 subset".
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
EXP_DIR = os.path.join(CKNNA_ROOT, "record", "exp_horizon")

DTW_VERSION = "sym2_cuda_nowin"
K = 10

CSV_PATH = os.path.join(EXP_DIR, "cknna_data_exp1v2_full_L", "finetune",
                         f"dtw_{DTW_VERSION}",
                         f"results_groups_dtw_{DTW_VERSION}.csv")

HORIZONS = [1, 3, 7, 15, 25, 40]

FIG_DIR = SCRIPT_DIR

META_CSV = os.path.join(CKNNA_ROOT, "cknna_data", "performance",
                        "model_metadata.csv")


def _load_display_names(csv_path):
    names = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["model_key"] not in names:
                names[row["model_key"]] = row["figure_label"]
    return names


_DISPLAY_NAMES = _load_display_names(META_CSV)


def display_name(key):
    return _DISPLAY_NAMES.get(key, key)


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
}

FAMILY_COLOR = {
    "Qwen-GR00T-Bridge":          "#1f6eb5",
    "Qwen-GR00T-Bridge-RT-1":     "#4a90d9",
    "Qwen3VL-GR00T-Bridge-RT-1":  "#7bb8f0",
    "Qwen-OFT-Bridge-RT-1":       "#e07c00",
    "Qwen3VL-OFT-Bridge-RT-1":    "#f5a840",
    "spatialvla-sft-bridge":       "#6b3a9e",
    "pi0-lerobot-bridge":          "#c9436a",
    "openvla-7b-bridge":           "#2e7d46",
    "openvla-7b-bridge-ft-200k":   "#5ab070",
    "rt1x-bridge":                 "#555555",
    "octo-base-bridge":            "#999999",
    "cogact-small-bridge":         "#8b0000",
    "cogact-base-bridge":          "#c62828",
    "cogact-large-bridge":         "#ef5350",
    "groot-n15-bridge":            "#5d4037",
    "groot-n16-bridge":            "#a1887f",
}

FA_VARIANTS = ["imgtext", "img", "txt"]
FA_LABELS = {"imgtext": "img+txt", "img": "image only", "txt": "text only"}


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


def load_data():
    """Return dict[model][variant][horizon] = cknna_k10."""
    data = defaultdict(lambda: defaultdict(dict))
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            data[r["model"]][r["feats_A_variant"]][int(r["horizon"])] = float(r["cknna_k10"])
    return data


def _compute_global_ranges(data, model_meta):
    """Compute global min/max for CKNNA (x) and SR (y) across ALL horizons and variants."""
    all_cknna = []
    all_sr = []
    for fa in FA_VARIANTS:
        for h in HORIZONS:
            for model_key, meta in model_meta.items():
                if model_key not in data:
                    continue
                if fa not in data[model_key]:
                    continue
                if h not in data[model_key][fa]:
                    continue
                all_cknna.append(data[model_key][fa][h])
                all_sr.append(meta["sr"])

    if not all_cknna:
        return (0, 1), (0, 100)

    cknna_min, cknna_max = min(all_cknna), max(all_cknna)
    sr_min, sr_max = min(all_sr), max(all_sr)

    # Extend by 5% on each side
    cknna_pad = (cknna_max - cknna_min) * 0.08
    sr_pad = (sr_max - sr_min) * 0.08

    return (cknna_min - cknna_pad, cknna_max + cknna_pad), (sr_min - sr_pad, sr_max + sr_pad)


def fig_cknna_vs_sr(data, model_meta):
    """Scatter: CKNNA vs success rate, rows=horizons, cols=variants. Fixed axes."""
    (x_lo, x_hi), (y_lo, y_hi) = _compute_global_ranges(data, model_meta)

    n_h = len(HORIZONS)
    fig, axes = plt.subplots(n_h, 3, figsize=(13, 4 * n_h))
    if n_h == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f"proprio_seq CKNNA [DTW] vs WidowX SR  "
                 f"(Bridge V2 subset, H_max={HORIZONS[-1]}, N=7762)",
                 fontsize=12, y=1.005)

    for ri, h in enumerate(HORIZONS):
        for ci, fa in enumerate(FA_VARIANTS):
            ax = axes[ri, ci]
            xs, ys = [], []
            for model_key, meta in model_meta.items():
                if model_key not in data:
                    continue
                if fa not in data[model_key]:
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
                xs.append(val)
                ys.append(meta["sr"])

            if len(xs) >= 3:
                rho, pval = spearmanr(xs, ys)
                p_str = f"{pval:.3f}" if pval >= 0.001 else f"{pval:.2e}"
                ax.text(0.04, 0.97, f"rho={rho:.3f}  p={p_str}\nn={len(xs)}",
                        transform=ax.transAxes, fontsize=7, va="top",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                  edgecolor="#aaaaaa", alpha=0.85))

            # Fixed axis ranges
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)

            ax.set_xlabel(f"CKNNA k={K}", fontsize=8)
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
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=7, bbox_to_anchor=(0.5, 0.0), framealpha=0.9)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_cknna_vs_sr_L.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig_cknna_vs_horizon(data):
    """Single line plot (Group L only, one subplot per variant), fixed y-axis."""
    # Compute global y range across all variants
    all_cknna = []
    models = sorted(data.keys())
    for fa in FA_VARIANTS:
        for m in models:
            if fa not in data[m]:
                continue
            for h in HORIZONS:
                v = data[m][fa].get(h, None)
                if v is not None:
                    all_cknna.append(v)

    if all_cknna:
        y_min, y_max = min(all_cknna), max(all_cknna)
        y_pad = (y_max - y_min) * 0.08
        y_lo, y_hi = y_min - y_pad, y_max + y_pad
    else:
        y_lo, y_hi = 0, 0.3

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for ax, fa in zip(axes, FA_VARIANTS):
        xs_all, ys_all = [], []
        for m in models:
            if fa not in data[m]:
                continue
            ys = [data[m][fa].get(h, np.nan) for h in HORIZONS]
            color = FAMILY_COLOR.get(m, "#aaaaaa")
            ax.plot(HORIZONS, ys, marker="o", markersize=4, linewidth=1.3,
                    color=color, alpha=0.7, label=display_name(m))
            for h, y in zip(HORIZONS, ys):
                if not np.isnan(y):
                    xs_all.append(h)
                    ys_all.append(y)

        # Mean line
        means = []
        for h in HORIZONS:
            vals = [data[m][fa][h]
                    for m in models
                    if fa in data[m] and h in data[m][fa]]
            means.append(np.mean(vals) if vals else np.nan)
        ax.plot(HORIZONS, means, marker="D", markersize=7, linewidth=2.8,
                color="black", zorder=10, label="Mean", linestyle="--")

        if len(xs_all) >= 3:
            rho, p = spearmanr(xs_all, ys_all)
            p_str = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        else:
            rho, p_str, sig = 0, "N/A", "N/A"

        ax.set_title(f"{FA_LABELS[fa]}  (H_max={HORIZONS[-1]})\n"
                     f"Spearman rho={rho:.3f}  p={p_str}  {sig}",
                     fontsize=10.5)
        ax.set_xlabel("Horizon h", fontsize=11)
        ax.set_ylabel("CKNNA  (k=10)", fontsize=11)
        ax.set_xticks(HORIZONS)
        ax.set_xticklabels([f"t+1\nto\nt+{h}" for h in HORIZONS], fontsize=8)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.98), loc="upper left",
               fontsize=8, framealpha=0.9, ncol=1)
    fig.suptitle(
        "proprio_seq CKNNA [DTW] vs horizon\n"
        "(Bridge V2 subset, N=7762, 16 models)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_cknna_vs_horizon_groups.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def print_statistics(data, model_meta):
    """Print CKNNA-vs-SR Spearman correlations for paper updates."""
    print("\n=== CKNNA vs SR Spearman correlations (BridgeV2 subset, all 16 models) ===\n")

    for fa in FA_VARIANTS:
        print(f"\n--- {FA_LABELS[fa]} ---")
        for h in HORIZONS:
            xs, ys = [], []
            for model_key, meta in model_meta.items():
                if model_key not in data or fa not in data[model_key]:
                    continue
                if h not in data[model_key][fa]:
                    continue
                xs.append(data[model_key][fa][h])
                ys.append(meta["sr"])
            if len(xs) >= 3:
                rho, pval = spearmanr(xs, ys)
                p_str = f"{pval:.3f}" if pval >= 0.001 else f"{pval:.2e}"
                sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "n.s."))
                print(f"  h={h:2d}: rho={rho:.3f}  p={p_str}  {sig}  n={len(xs)}")

    # Table 1 data at h=15
    print("\n=== Table 1 data: CKNNA at h=15 (all 16 models) ===")
    print(f"{'Model':<35s} {'imgtext':>8s} {'img':>8s} {'txt':>8s} {'SR':>7s}")
    for model_key, meta in sorted(model_meta.items(), key=lambda x: -x[1]["sr"]):
        if model_key not in data:
            continue
        vals = []
        for fa in FA_VARIANTS:
            if fa in data[model_key] and 15 in data[model_key][fa]:
                vals.append(f"{data[model_key][fa][15]:8.3f}")
            else:
                vals.append(f"{'N/A':>8s}")
        print(f"{meta['label']:<35s} " + " ".join(vals) + f" {meta['sr']:7.1f}")

    # Overall Spearman across all horizons (for exp1-spearman table)
    print("\n=== Overall Spearman (CKNNA vs horizon, all models x all horizons) ===")
    models = sorted(data.keys())
    for fa in ["imgtext"]:
        xs_all, ys_all = [], []
        for m in models:
            if fa not in data[m]:
                continue
            for h in HORIZONS:
                if h in data[m][fa]:
                    xs_all.append(h)
                    ys_all.append(data[m][fa][h])
        if len(xs_all) >= 3:
            rho, p = spearmanr(xs_all, ys_all)
            p_str = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
            print(f"  {FA_LABELS[fa]:>10s}: rho={rho:.3f}  p={p_str}  n={len(xs_all)}")

    # Mean CKNNA across models
    print("\n=== Mean CKNNA (imgtext) across 16 models per horizon ===")
    for h in HORIZONS:
        vals = [data[m]["imgtext"][h] for m in models
                if "imgtext" in data[m] and h in data[m]["imgtext"]]
        print(f"  h={h:2d}: mean={np.mean(vals):.3f} (n={len(vals)})")

    # All horizons for all models
    print("\n=== Per-model CKNNA at all horizons (imgtext) ===")
    print(f"{'Model':<35s} " + " ".join(f"{'h='+str(h):>7s}" for h in HORIZONS))
    for model_key, meta in sorted(model_meta.items(), key=lambda x: -x[1]["sr"]):
        if model_key not in data or "imgtext" not in data[model_key]:
            continue
        vals = [data[model_key]["imgtext"].get(h, float('nan')) for h in HORIZONS]
        print(f"{meta['label']:<35s} " + " ".join(f"{v:7.3f}" for v in vals))


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    model_meta = load_model_meta()
    data = load_data()

    print(f"Models loaded: {len(data)}")
    print(f"Horizons: {HORIZONS}")

    fig_cknna_vs_sr(data, model_meta)
    fig_cknna_vs_horizon(data)
    print_statistics(data, model_meta)
    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
