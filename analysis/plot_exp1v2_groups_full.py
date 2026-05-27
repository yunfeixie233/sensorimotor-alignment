"""
Plot Exp 1v2 results (group-specific H_max, DTW CKNNA).

Figure A: CKNNA vs Success Rate scatter per group (rows=horizons, cols=img+txt/img/txt)
Figure B: CKNNA vs Horizon line plot (3 subplots S/M/L, all models)
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
EXP_DIR    = SCRIPT_DIR
CKNNA_ROOT = os.environ.get("DATA_STORE", "/home/ubuntu/verl/starVLA/cknna")

DTW_VERSION = "sym2_cuda_nowin"
K = 10

GROUP_CSVS = {
    "S": os.path.join(EXP_DIR, "cknna_data_exp1v2_full_S", "finetune",
                      f"dtw_{DTW_VERSION}",
                      f"results_groups_dtw_{DTW_VERSION}.csv"),
    "M": os.path.join(EXP_DIR, "cknna_data_exp1v2_full_M", "finetune",
                      f"dtw_{DTW_VERSION}",
                      f"results_groups_dtw_{DTW_VERSION}.csv"),
    "L": os.path.join(EXP_DIR, "cknna_data_exp1v2_full_L", "finetune",
                      f"dtw_{DTW_VERSION}",
                      f"results_groups_dtw_{DTW_VERSION}.csv"),
}

GROUP_HORIZONS = {
    "S": [1, 3, 7, 15],
    "M": [1, 3, 7, 15, 25],
    "L": [1, 3, 7, 15, 25, 40],
}

FIG_DIR = os.path.join(EXP_DIR, "exp1v2_full_figures")

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

MODEL_SHORT = _DISPLAY_NAMES

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

GROUP_COLOR = {"S": "#e74c3c", "M": "#2980b9", "L": "#27ae60"}

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


def load_all_data():
    """Return dict[group][model][variant][horizon] = cknna_k10."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for group, csv_path in GROUP_CSVS.items():
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                data[group][r["model"]][r["feats_A_variant"]][int(r["horizon"])] = float(r["cknna_k10"])
    return data


def fig_a_cknna_vs_sr(data, model_meta):
    """Per-group scatter: CKNNA vs success rate, rows=horizons, cols=variants."""
    for group in ["S", "M", "L"]:
        horizons = GROUP_HORIZONS[group]
        n_h = len(horizons)
        fig, axes = plt.subplots(n_h, 3, figsize=(13, 4 * n_h))
        if n_h == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(f"Group {group}: proprio_seq CKNNA [DTW] vs WidowX SR  "
                     f"(H_max={horizons[-1]})", fontsize=12, y=1.005)

        for ri, h in enumerate(horizons):
            for ci, fa in enumerate(FA_VARIANTS):
                ax = axes[ri, ci]
                xs, ys = [], []
                for model_key, meta in model_meta.items():
                    if model_key not in data[group]:
                        continue
                    if fa not in data[group][model_key]:
                        continue
                    if h not in data[group][model_key][fa]:
                        continue
                    val = data[group][model_key][fa][h]
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
        out = os.path.join(FIG_DIR, f"fig_cknna_vs_sr_{group}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


def fig_b_cknna_vs_horizon(data):
    """3 subplots (S/M/L), all model lines, imgtext only."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)

    for ax, group in zip(axes, ["S", "M", "L"]):
        horizons = GROUP_HORIZONS[group]
        models = sorted(data[group].keys())

        xs_all, ys_all = [], []
        for m in models:
            if "imgtext" not in data[group][m]:
                continue
            ys = [data[group][m]["imgtext"].get(h, np.nan) for h in horizons]
            color = FAMILY_COLOR.get(m, "#aaaaaa")
            ax.plot(horizons, ys, marker="o", markersize=4, linewidth=1.3,
                    color=color, alpha=0.7, label=display_name(m))
            for h, y in zip(horizons, ys):
                if not np.isnan(y):
                    xs_all.append(h)
                    ys_all.append(y)

        means = []
        for h in horizons:
            vals = [data[group][m]["imgtext"][h]
                    for m in models
                    if "imgtext" in data[group][m] and h in data[group][m]["imgtext"]]
            means.append(np.mean(vals) if vals else np.nan)
        ax.plot(horizons, means, marker="D", markersize=7, linewidth=2.8,
                color="black", zorder=10, label="Mean", linestyle="--")

        if len(xs_all) >= 3:
            rho, p = spearmanr(xs_all, ys_all)
            p_str = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        else:
            rho, p_str, sig = 0, "N/A", "N/A"

        ax.set_title(f"Group {group}  (H_max={horizons[-1]})\n"
                     f"Spearman rho={rho:.3f}  p={p_str}  {sig}",
                     fontsize=10.5)
        ax.set_xlabel("Horizon h", fontsize=11)
        ax.set_ylabel("CKNNA  (k=10)", fontsize=11)
        ax.set_xticks(horizons)
        ax.set_xticklabels([f"t+1\nto\nt+{h}" for h in horizons], fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=9)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.98), loc="upper left",
               fontsize=8, framealpha=0.9, ncol=1)
    fig.suptitle(
        "Exp 1v2: proprio_seq CKNNA [DTW] vs horizon, by episode-length group\n"
        "(N=7762/group, 16 models, feats_A=imgtext, group-specific H_max)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_cknna_vs_horizon_groups.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    model_meta = load_model_meta()
    data = load_all_data()

    print(f"Groups loaded: {sorted(data.keys())}")
    for g in ["S", "M", "L"]:
        print(f"  Group {g}: {len(data[g])} models, horizons={GROUP_HORIZONS[g]}")

    fig_a_cknna_vs_sr(data, model_meta)
    fig_b_cknna_vs_horizon(data)
    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
