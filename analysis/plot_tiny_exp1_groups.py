"""
Plot Tiny Exp 1 results: CKNNA vs horizon for groups S / M / L.

Produces two figures saved to record/finetune/tiny_exp1/:
  fig1_group_lines.png  -- 3 subplots, one per group, all 16 model lines
  fig2_group_summary.png -- single panel, mean ± std per group
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
DTW_VERSION = "sym2_cuda_nowin"

# DTW results: per-group computation within each group (correct)
CSV_PATH = os.path.abspath(
    os.path.join(
        SCRIPT_DIR, "..",
        "cknna_data_exp1_tiny", "finetune",
        f"dtw_{DTW_VERSION}",
        f"results_groups_dtw_{DTW_VERSION}.csv",
    )
)
FIG_DIR = os.path.dirname(CSV_PATH)
os.makedirs(FIG_DIR, exist_ok=True)

HORIZONS = [1, 3, 7, 15]
K = 10

SHORT = {
    "Qwen-GR00T-Bridge":          "GR00T",
    "Qwen-GR00T-Bridge-RT-1":     "GR00T-RT1",
    "Qwen3VL-GR00T-Bridge-RT-1":  "Q3-GR00T",
    "Qwen-OFT-Bridge-RT-1":       "OFT",
    "Qwen3VL-OFT-Bridge-RT-1":    "Q3-OFT",
    "spatialvla-sft-bridge":       "SpatialVLA",
    "pi0-lerobot-bridge":          "Pi0",
    "openvla-7b-bridge":           "OpenVLA",
    "openvla-7b-bridge-ft-200k":   "OpenVLA-FT",
    "rt1x-bridge":                 "RT-1-X",
    "octo-base-bridge":            "Octo",
    "cogact-small-bridge":         "CogACT-S",
    "cogact-base-bridge":          "CogACT-B",
    "cogact-large-bridge":         "CogACT-L",
    "groot-n15-bridge":            "GR00T-N1.5",
    "groot-n16-bridge":            "GR00T-N1.6",
}

# Color families
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
GROUP_LABEL = {
    "S": "Group S  (L=16–25, short episodes)",
    "M": "Group M  (L=26–40, medium episodes)",
    "L": "Group L  (L>40,    long episodes)",
}


def load_data():
    """Return dict[model][group][horizon] = cknna_k10."""
    data = defaultdict(lambda: defaultdict(dict))
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            if r["feats_B_type"] == "proprio_seq" and r["feats_A_variant"] == "imgtext":
                data[r["model"]][r["group"]][int(r["horizon"])] = float(r["cknna_k10"])
    return data


def spearman_group(data, group):
    xs, ys = [], []
    for model_d in data.values():
        if group not in model_d:
            continue
        for h in HORIZONS:
            if h in model_d[group]:
                xs.append(h)
                ys.append(model_d[group][h])
    rho, p = spearmanr(xs, ys)
    return rho, p


def fig1_group_lines(data):
    """3 subplots, one per group, all 16 model lines."""
    models = sorted(data.keys())
    groups = ["S", "M", "L"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=False)

    for ax, g in zip(axes, groups):
        rho, p = spearman_group(data, g)
        p_str = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))

        # Individual model lines
        for m in models:
            if g not in data[m]:
                continue
            ys = [data[m][g].get(h, np.nan) for h in HORIZONS]
            color = FAMILY_COLOR.get(m, "#aaaaaa")
            ax.plot(HORIZONS, ys, marker="o", markersize=4, linewidth=1.3,
                    color=color, alpha=0.7, label=SHORT.get(m, m))

        # Mean line
        means = []
        for h in HORIZONS:
            vals = [data[m][g][h] for m in models if g in data[m] and h in data[m][g]]
            means.append(np.mean(vals) if vals else np.nan)
        ax.plot(HORIZONS, means, marker="D", markersize=7, linewidth=2.8,
                color="black", zorder=10, label="Mean", linestyle="--")

        ax.set_title(f"{GROUP_LABEL[g]}\nSpearman ρ={rho:.3f}  p={p_str}  {sig}",
                     fontsize=10.5)
        ax.set_xlabel("Horizon h", fontsize=11)
        ax.set_ylabel("CKNNA  (k=10)", fontsize=11)
        ax.set_xticks(HORIZONS)
        ax.set_xticklabels([f"t+1\nto\nt+{h}" for h in HORIZONS], fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=9)

    # Shared legend from last axis
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.98), loc="upper left",
               fontsize=8, framealpha=0.9, ncol=1)
    fig.suptitle(
        "Tiny Exp 1: proprio_seq CKNNA vs horizon, by episode-length group\n"
        "(N=1000 per group, 16 models, feats_A=VLM imgtext, feats_B=proprio t+1..t+h)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig1_group_lines.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig2_group_summary(data):
    """Single panel: mean ± std per group, horizon on x-axis."""
    models = sorted(data.keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: mean ± std lines
    ax = axes[0]
    for g in ["S", "M", "L"]:
        rho, p = spearman_group(data, g)
        p_str = f"p={p:.2e}" if p < 0.001 else f"p={p:.3f}"
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        color = GROUP_COLOR[g]

        means, stds = [], []
        for h in HORIZONS:
            vals = [data[m][g][h] for m in models if g in data[m] and h in data[m][g]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        means = np.array(means)
        stds = np.array(stds)

        label = f"Group {g}  (ρ={rho:.2f}, {p_str} {sig})"
        ax.plot(HORIZONS, means, marker="o", markersize=7, linewidth=2.2,
                color=color, label=label)
        ax.fill_between(HORIZONS, means - stds, means + stds,
                        color=color, alpha=0.15)

    ax.set_xlabel("Horizon h", fontsize=12)
    ax.set_ylabel("CKNNA  (k=10)  mean ± std across 16 models", fontsize=11)
    ax.set_title("Mean CKNNA per group  (proprio_seq, imgtext)", fontsize=12)
    ax.set_xticks(HORIZONS)
    ax.set_xticklabels([f"t+1\nto\nt+{h}" for h in HORIZONS], fontsize=10)
    ax.legend(fontsize=9.5, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Right: per-model h=1 vs h=15 slope (delta CKNNA)
    ax2 = axes[1]
    for g in ["S", "M", "L"]:
        color = GROUP_COLOR[g]
        deltas = []
        for m in models:
            if g not in data[m]:
                continue
            v1  = data[m][g].get(1,  np.nan)
            v15 = data[m][g].get(15, np.nan)
            if not (np.isnan(v1) or np.isnan(v15)):
                deltas.append(v15 - v1)
        x_pos = {"S": 0, "M": 1, "L": 2}[g]
        ax2.boxplot(deltas, positions=[x_pos], widths=0.5,
                    patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.55),
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(color=color),
                    capprops=dict(color=color),
                    flierprops=dict(marker="o", markersize=4, color=color, alpha=0.7))
        # jitter overlay
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(deltas))
        ax2.scatter([x_pos + j for j in jitter], deltas, s=22,
                    color=color, alpha=0.7, zorder=5)

    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["Group S", "Group M", "Group L"], fontsize=11)
    ax2.set_ylabel("ΔCKNNA  (h=15 minus h=1)", fontsize=11)
    ax2.set_title("Horizon gain: CKNNA(h=15) − CKNNA(h=1)\nper model per group", fontsize=11)
    ax2.grid(True, alpha=0.25, axis="y")

    fig.suptitle(
        "Tiny Exp 1: proprio_seq CKNNA  —  group-level summary\n"
        "(N=1000/group, 16 models, seed=42)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig2_group_summary.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    data = load_data()
    print(f"Loaded {len(data)} models, groups: {sorted({g for m in data.values() for g in m})}")
    fig1_group_lines(data)
    fig2_group_summary(data)
    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
