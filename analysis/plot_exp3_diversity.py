"""
Exp 3-Diversity: CKNNA vs Horizon on diverse dataset (H_max=45, N=6784, 4684 tasks).

Compares three conditions:
  Left:   Exp1 Group L   (H_max=40, N=10619, diverse, 9% sweep)
  Middle: Exp3-Diversity  (H_max=45, N=6784,  diverse, 14% sweep)  [new]
  Right:  Exp2           (H_max=75, N=985,   sweep-dominated, 79%)

Scientific question: does saturation at h=25 persist with diverse tasks?
"""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR    = SCRIPT_DIR
CKNNA_ROOT = os.environ.get("DATA_STORE", "/home/ubuntu/verl/starVLA/cknna")

META_CSV = os.path.join(CKNNA_ROOT, "cknna_data", "performance",
                        "model_metadata.csv")

DTW_VER = "sym2_cuda_nowin"

EXP1L_CSV   = os.path.join(EXP_DIR, "cknna_data_exp1v2_full_L", "finetune",
                            f"dtw_{DTW_VER}", f"results_groups_dtw_{DTW_VER}.csv")
EXP3D_CSV   = os.path.join(CKNNA_ROOT, "record", "cknna_data_exp3_diversity", "finetune",
                            f"dtw_{DTW_VER}", f"results_groups_dtw_{DTW_VER}.csv")
EXP2_CSV    = os.path.join(EXP_DIR, "cknna_data_exp2", "finetune",
                            f"dtw_{DTW_VER}", f"results_groups_dtw_{DTW_VER}.csv")

EXP1L_HORIZONS  = [1, 3, 7, 15, 25, 40]
EXP3D_HORIZONS  = [1, 3, 7, 15, 25, 35, 40, 45]
EXP2_HORIZONS   = [1, 3, 7, 15, 25, 40, 50, 60, 75]

FIG_DIR = os.path.join(EXP_DIR, "exp3_diversity_figures")

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
    "qwen25vl-3b-raw":             "#1abc9c",
    "qwen3vl-4b-raw":              "#16a085",
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
    return _DISPLAY_NAMES.get(key, key)


def load_csv(path):
    data = defaultdict(lambda: defaultdict(dict))
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["feats_B_type"] != "proprio_seq":
                continue
            if r["feats_A_variant"] != "imgtext":
                continue
            data[r["model"]][int(r["horizon"])] = float(r["cknna_k10"])
    return data


def _spearman(xs, ys):
    if len(xs) < 3:
        return 0.0, "N/A", "N/A"
    rho, p = spearmanr(xs, ys)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
    p_str = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
    return rho, p_str, sig


def plot_one(ax, data, horizons, title, color_fn=None):
    models = sorted(data.keys())
    xs_all, ys_all = [], []

    for m in models:
        ys = [data[m].get(h, np.nan) for h in horizons]
        color = FAMILY_COLOR.get(m, "#aaaaaa")
        ax.plot(horizons, ys, marker="o", markersize=3.5, linewidth=1.2,
                color=color, alpha=0.65, label=display_name(m))
        for h, y in zip(horizons, ys):
            if not np.isnan(y):
                xs_all.append(h)
                ys_all.append(y)

    means = []
    for h in horizons:
        vals = [data[m][h] for m in models if h in data[m]]
        means.append(np.mean(vals) if vals else np.nan)
    ax.plot(horizons, means, marker="D", markersize=6, linewidth=2.5,
            color="black", zorder=10, label="Mean", linestyle="--")

    rho, p_str, sig = _spearman(xs_all, ys_all)
    ax.set_title(f"{title}\nSpearman rho={rho:.3f}  p={p_str}  {sig}", fontsize=9.5)
    ax.set_xlabel("Horizon h", fontsize=10)
    ax.set_ylabel("CKNNA (imgtext, k=10)", fontsize=10)
    ax.set_xticks(horizons)
    ax.set_xticklabels([f"t+{h}" for h in horizons], fontsize=7, rotation=30)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    data1L  = load_csv(EXP1L_CSV)
    data3d  = load_csv(EXP3D_CSV)
    data2   = load_csv(EXP2_CSV)

    print(f"Exp1 Group L: {len(data1L)} models")
    print(f"Exp3-Diversity: {len(data3d)} models")
    print(f"Exp2: {len(data2)} models")

    # ----------------------------------------------------------------
    # Figure A: 3-panel CKNNA vs horizon comparison
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=False)

    plot_one(axes[0], data1L,  EXP1L_HORIZONS,
             "Exp1 Group L\n(H_max=40, N=10619, diverse, 9% sweep)")
    plot_one(axes[1], data3d,  EXP3D_HORIZONS,
             "Exp3-Diversity\n(H_max=45, N=6784, diverse, 14% sweep)")
    plot_one(axes[2], data2,   EXP2_HORIZONS,
             "Exp2\n(H_max=75, N=985, sweep-dominated, 79%)")

    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.98), loc="upper left",
               fontsize=7.5, framealpha=0.9, ncol=1)

    fig.suptitle(
        "Exp 3-Diversity vs Exp1 vs Exp2: Does CKNNA Saturation Persist with Diverse Tasks?\n"
        "(feats_A=imgtext, all models, DTW CKNNA k=10)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_exp3_diversity_vs_horizon.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # ----------------------------------------------------------------
    # Figure B: Mean curves overlay (clean comparison)
    # ----------------------------------------------------------------
    fig2, ax2 = plt.subplots(1, 1, figsize=(9, 5.5))

    def mean_curve(data, horizons, label, color, ls="-"):
        models = sorted(data.keys())
        means = []
        for h in horizons:
            vals = [data[m][h] for m in models if h in data[m]]
            means.append(np.mean(vals) if vals else np.nan)
        ax2.plot(horizons, means, marker="o", markersize=6, linewidth=2.5,
                 color=color, label=label, linestyle=ls, zorder=5)

    mean_curve(data1L, EXP1L_HORIZONS,
               "Exp1 Group L (H=40, diverse, 9% sweep)", "#1f77b4", "-")
    mean_curve(data3d, EXP3D_HORIZONS,
               "Exp3-Diversity (H=45, diverse, 14% sweep)", "#2ca02c", "-")
    mean_curve(data2,  EXP2_HORIZONS,
               "Exp2 (H=75, sweep-dominated, 79%)", "#d62728", "--")

    ax2.set_xlabel("Horizon h", fontsize=12)
    ax2.set_ylabel("Mean CKNNA (imgtext, k=10)", fontsize=12)
    ax2.set_title(
        "Mean CKNNA vs Horizon: Diverse vs Sweep-dominated Datasets\n"
        "Saturation check: does peak at h=25 persist with diverse tasks?",
        fontsize=11,
    )
    ax2.legend(fontsize=10, loc="lower right")
    ax2.grid(True, alpha=0.25)
    ax2.tick_params(labelsize=10)
    fig2.tight_layout()
    out2 = os.path.join(FIG_DIR, "fig_exp3_diversity_mean_comparison.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out2}")

    # ----------------------------------------------------------------
    # Figure C: Exp3-Diversity individual model lines (standalone)
    # ----------------------------------------------------------------
    fig3, ax3 = plt.subplots(1, 1, figsize=(11, 6))
    plot_one(ax3, data3d, EXP3D_HORIZONS,
             "Exp3-Diversity: CKNNA vs Horizon\n"
             "(H_max=45, N=6784, 4684 tasks, 14% sweep)")
    handles3, labels3 = ax3.get_legend_handles_labels()
    fig3.legend(handles3, labels3, bbox_to_anchor=(1.01, 0.98), loc="upper left",
                fontsize=8, framealpha=0.9, ncol=1)
    fig3.tight_layout()
    out3 = os.path.join(FIG_DIR, "fig_exp3_diversity_all_models.png")
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {out3}")

    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
