"""Plot 3-way CKNNA comparison (imgtext / img / txt) vs WidowX success rate.

Generates scatter plots comparing how CKNNA_proprio and CKNNA_action correlate
with success rate when feats_A represents image+text, image-only, or text-only.

Usage:
    python plot_3way_cknna.py
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.environ.get("RECORD_DIR", "/home/ubuntu/verl/starVLA/cknna/record")
CSV_PATH = os.path.join(RECORD_DIR, "cknna_3way_5k.csv")
OUT_DIR = os.path.join(RECORD_DIR, "runs", "3way_5k")

GROUPS = {
    "StarVLA-GR00T": {
        "models": [
            "Qwen2.5-GR00T-Bridge",
            "Qwen2.5-GR00T-Bridge-RT-1",
            "Qwen3-GR00T-Bridge-RT-1",
        ],
        "cmap": plt.cm.Blues,
    },
    "StarVLA-Other": {
        "models": [
            "Qwen2.5-FAST-Bridge-RT-1",
            "Qwen2.5-OFT-Bridge-RT-1",
            "Qwen3-OFT-Bridge-RT-1",
        ],
        "cmap": plt.cm.Oranges,
    },
    "SpatialVLA/Pi0/OpenVLA": {
        "models": [
            "spatialvla-sft-bridge",
            "pi0-lerobot-bridge",
            "openvla-7b-bridge",
            "openvla-7b-bridge-ft-200k",
        ],
        "cmap": plt.cm.Greens,
    },
    "CogACT": {
        "models": [
            "CogACT-Small",
            "CogACT-Base",
            "CogACT-Large",
        ],
        "cmap": plt.cm.Purples,
    },
    "GR00T-NVIDIA": {
        "models": [
            "GR00T-N1.5-Bridge",
            "GR00T-N1.6-Bridge",
        ],
        "cmap": plt.cm.Reds,
    },
    "RT-1-X/Octo": {
        "models": [
            "RT-1-X",
            "Octo-base",
        ],
        "cmap": plt.cm.Greys,
    },
}

SHORT = {
    "Qwen2.5-GR00T-Bridge": "GR00T",
    "Qwen2.5-GR00T-Bridge-RT-1": "GR00T-RT1",
    "Qwen3-GR00T-Bridge-RT-1": "Q3-GR00T",
    "Qwen2.5-FAST-Bridge-RT-1": "FAST",
    "Qwen2.5-OFT-Bridge-RT-1": "OFT",
    "Qwen3-OFT-Bridge-RT-1": "Q3-OFT",
    "spatialvla-sft-bridge": "SpatialVLA",
    "pi0-lerobot-bridge": "Pi0",
    "openvla-7b-bridge": "OpenVLA",
    "openvla-7b-bridge-ft-200k": "OpenVLA-FT",
    "CogACT-Small": "CogACT-S",
    "CogACT-Base": "CogACT-B",
    "CogACT-Large": "CogACT-L",
    "GR00T-N1.5-Bridge": "GR00T-1.5",
    "GR00T-N1.6-Bridge": "GR00T-1.6",
    "RT-1-X": "RT-1-X",
    "Octo-base": "Octo",
}


def load_csv():
    rows = {}
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["Model"]] = row
    return rows


def plot_scatter(data, metric_cols, title, ylabel, out_path):
    """Plot 3-panel scatter (imgtext / img / txt) sharing y-axis."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    variant_labels = ["Image + Text", "Image Only", "Text Only"]

    for ax_idx, (col, vlabel) in enumerate(zip(metric_cols, variant_labels)):
        ax = axes[ax_idx]
        all_x, all_y = [], []

        for gname, ginfo in GROUPS.items():
            cmap = ginfo["cmap"]
            models = ginfo["models"]
            n = len(models)
            colors = [cmap(0.4 + 0.5 * i / max(n - 1, 1)) for i in range(n)]

            for model, color in zip(models, colors):
                row = data.get(model)
                if row is None:
                    continue
                x_val = row.get(col, "")
                y_val = row.get("WidowX_avg", "")
                if x_val == "" or y_val == "":
                    continue
                x = float(x_val)
                y = float(y_val)
                label_txt = SHORT.get(model, model)
                ax.scatter(x, y, c=[color], s=80, zorder=3, edgecolors="black", linewidths=0.5)
                ax.annotate(label_txt, (x, y), fontsize=7, ha="left", va="bottom",
                            xytext=(4, 4), textcoords="offset points")
                all_x.append(x)
                all_y.append(y)

        if len(all_x) >= 3:
            rho, pval = stats.spearmanr(all_x, all_y)
            ax.set_title(f"{vlabel}\nSpearman rho={rho:.3f}, p={pval:.4f}", fontsize=11)
        else:
            ax.set_title(vlabel, fontsize=11)

        ax.set_xlabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("WidowX Success Rate (%)", fontsize=10)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_bar_comparison(data, metric_prefix, title, out_path):
    """Bar chart comparing imgtext/img/txt CKNNA for each model."""
    models_ordered = []
    for ginfo in GROUPS.values():
        models_ordered.extend(ginfo["models"])

    names = []
    vals_imgtext = []
    vals_img = []
    vals_txt = []

    for m in models_ordered:
        row = data.get(m)
        if row is None:
            continue
        v1 = row.get(f"{metric_prefix}_imgtext_k10", "")
        v2 = row.get(f"{metric_prefix}_img_k10", "")
        v3 = row.get(f"{metric_prefix}_txt_k10", "")
        if v1 == "" and v2 == "" and v3 == "":
            continue
        names.append(SHORT.get(m, m))
        vals_imgtext.append(float(v1) if v1 else 0)
        vals_img.append(float(v2) if v2 else 0)
        vals_txt.append(float(v3) if v3 else 0)

    x = np.arange(len(names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w, vals_imgtext, w, label="Image+Text", color="#4477AA")
    ax.bar(x, vals_img, w, label="Image Only", color="#EE6677")
    ax.bar(x + w, vals_txt, w, label="Text Only", color="#228833")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Score (k=10)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_csv()

    plot_scatter(
        data,
        ["CKNNA_proprio_imgtext_k10", "CKNNA_proprio_img_k10", "CKNNA_proprio_txt_k10"],
        "CKNNA_proprio vs WidowX Success (k=10, N=5K)",
        "CKNNA_proprio (k=10)",
        os.path.join(OUT_DIR, "scatter_CKNNA_proprio_3way.png"),
    )

    plot_scatter(
        data,
        ["CKNNA_action_imgtext_k10", "CKNNA_action_img_k10", "CKNNA_action_txt_k10"],
        "CKNNA_action vs WidowX Success (k=10, N=5K)",
        "CKNNA_action (k=10)",
        os.path.join(OUT_DIR, "scatter_CKNNA_action_3way.png"),
    )

    plot_scatter(
        data,
        ["MutualKNN_proprio_imgtext_k10", "MutualKNN_proprio_img_k10", "MutualKNN_proprio_txt_k10"],
        "MutualKNN_proprio vs WidowX Success (k=10, N=5K)",
        "MutualKNN_proprio (k=10)",
        os.path.join(OUT_DIR, "scatter_MutualKNN_proprio_3way.png"),
    )

    plot_scatter(
        data,
        ["MutualKNN_action_imgtext_k10", "MutualKNN_action_img_k10", "MutualKNN_action_txt_k10"],
        "MutualKNN_action vs WidowX Success (k=10, N=5K)",
        "MutualKNN_action (k=10)",
        os.path.join(OUT_DIR, "scatter_MutualKNN_action_3way.png"),
    )

    plot_bar_comparison(
        data,
        "CKNNA_proprio",
        "CKNNA_proprio (k=10): Image+Text vs Image vs Text",
        os.path.join(OUT_DIR, "bar_CKNNA_proprio_3way.png"),
    )

    plot_bar_comparison(
        data,
        "CKNNA_action",
        "CKNNA_action (k=10): Image+Text vs Image vs Text",
        os.path.join(OUT_DIR, "bar_CKNNA_action_3way.png"),
    )

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
