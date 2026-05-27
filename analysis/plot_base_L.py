"""
plot_base_L.py

Plot CKNNA (proprio_seq DTW) vs WidowX success rate for BASE models,
using the Group-L Bridge V2 subset (N=7762, H_max=40).

Reads:  record/exp_horizon_L/base/results_groups_dtw_sym2_cuda_nowin.csv
Writes: record/exp_horizon_L/base/fig_cknna_vs_sr_base_L.png
        record/exp_horizon_L/base/fig_cknna_vs_horizon_base_L.png

Usage:
    python cknna/record/exp_horizon_L/base/plot_base_L.py
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

CSV_PATH = os.path.join(SCRIPT_DIR, "results_groups_dtw_sym2_cuda_nowin.csv")

HORIZONS = [1, 3, 7, 15, 25, 40]

META_CSV = os.path.join(CKNNA_ROOT, "cknna_data", "performance", "model_metadata.csv")

K = 10

FA_VARIANTS = ["imgtext", "img", "txt"]
FA_LABELS   = {"imgtext": "img+txt", "img": "image only", "txt": "text only"}

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
    "cogact-small-bridge":  "#8b0000",
    "cogact-base-bridge":   "#c62828",
    "cogact-large-bridge":  "#ef5350",
    "groot-n15-base":       "#5d4037",
    "groot-n16-base":       "#a1887f",
    "openvla-7b-bridge":    "#2e7d46",
    "pi0-base":             "#c9436a",
    "spatialvla-pt-base":   "#6b3a9e",
    "rt1x-bridge":          "#555555",
    "octo-base-bridge":     "#999999",
}


def _parse_bool(s):
    return s.strip().lower() in ("true", "1", "yes")


def load_model_meta():
    """Load base models (finetune_on_bridge=no) from model_metadata.csv."""
    meta = {}
    with open(META_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["finetune_on_bridge"] != "no":
                continue
            meta[row["model_key"]] = {
                "sr":    float(row["success_rate"]),
                "group": row["group"],
                "label": row["figure_label"],
            }
    return meta


def _load_display_names():
    names = {}
    with open(META_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["model_key"] not in names:
                names[row["model_key"]] = row["figure_label"]
    return names


_DISPLAY_NAMES = _load_display_names()


def display_name(key):
    return _DISPLAY_NAMES.get(key, key)


def load_data():
    """Return dict[model][variant][horizon] = cknna_k10."""
    data = defaultdict(lambda: defaultdict(dict))
    if not os.path.exists(CSV_PATH):
        print(f"WARNING: CSV not found: {CSV_PATH}")
        return data
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            data[r["model"]][r["feats_A_variant"]][int(r["horizon"])] = float(r["cknna_k10"])
    return data


def _compute_global_ranges(data, model_meta):
    all_cknna = []
    all_sr    = []
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
    sr_min,    sr_max    = min(all_sr),    max(all_sr)

    cknna_pad = (cknna_max - cknna_min) * 0.08
    sr_pad    = (sr_max    - sr_min)    * 0.08

    return (cknna_min - cknna_pad, cknna_max + cknna_pad), \
           (sr_min    - sr_pad,    sr_max    + sr_pad)


def fig_cknna_vs_sr(data, model_meta):
    """Scatter: CKNNA vs success rate, rows=horizons, cols=variants."""
    (x_lo, x_hi), (y_lo, y_hi) = _compute_global_ranges(data, model_meta)

    n_h = len(HORIZONS)
    fig, axes = plt.subplots(n_h, 3, figsize=(13, 4 * n_h))
    if n_h == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"proprio_seq CKNNA [DTW] vs WidowX SR  "
        f"(Base models, Bridge V2 subset, H_max={HORIZONS[-1]}, N=7762)",
        fontsize=12, y=1.005,
    )

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
    out = os.path.join(SCRIPT_DIR, "fig_cknna_vs_sr_base_L.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig_cknna_vs_horizon(data):
    """Line plot: CKNNA vs horizon, one subplot per variant."""
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
    n_models = len(models)
    fig.suptitle(
        f"proprio_seq CKNNA [DTW] vs horizon\n"
        f"(Base models, Bridge V2 subset, N=7762, {n_models} models)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = os.path.join(SCRIPT_DIR, "fig_cknna_vs_horizon_base_L.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def print_statistics(data, model_meta):
    print("\n=== CKNNA vs SR Spearman correlations (base models, Group L) ===\n")
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

    print("\n=== Per-model CKNNA at all horizons (imgtext) ===")
    models = sorted(data.keys())
    print(f"{'Model':<35s} " + " ".join(f"{'h='+str(h):>7s}" for h in HORIZONS))
    for model_key, meta in sorted(model_meta.items(), key=lambda x: -x[1]["sr"]):
        if model_key not in data or "imgtext" not in data[model_key]:
            continue
        vals = [data[model_key]["imgtext"].get(h, float("nan")) for h in HORIZONS]
        print(f"{meta['label']:<35s} " + " ".join(f"{v:7.3f}" for v in vals))


def main():
    os.makedirs(SCRIPT_DIR, exist_ok=True)
    model_meta = load_model_meta()
    data = load_data()

    print(f"Models in CSV: {len(data)}")
    print(f"Models in meta (base): {len(model_meta)}")
    print(f"Horizons: {HORIZONS}")

    if not data:
        print("ERROR: no data loaded. Run compute_base_L.py first.")
        return

    fig_cknna_vs_sr(data, model_meta)
    fig_cknna_vs_horizon(data)
    print_statistics(data, model_meta)
    print(f"\nAll figures saved to: {SCRIPT_DIR}")


if __name__ == "__main__":
    main()
