"""
Plot combined Exp 1v2 FULL results (all groups pooled, N=23286, DTW CKNNA).

Generates fig_cknna_vs_sr_ALL.png: CKNNA vs SR scatter,
rows = horizons {1, 3, 7, 15}, cols = {img+txt, img, txt}.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKNNA_ROOT = os.environ.get("DATA_STORE", "/home/ubuntu/verl/starVLA/cknna")

DTW_VERSION = "sym2_cuda_nowin"
K = 10

COMBINED_CSV = os.path.join(
    SCRIPT_DIR, "..",
    "cknna_data_exp1v2_full_ALL", "finetune",
    f"dtw_{DTW_VERSION}",
    f"cknna_dtw_{DTW_VERSION}_results.csv",
)

HORIZONS = [1, 3, 7, 15]

FIG_DIR = os.path.join(SCRIPT_DIR, "exp1v2_full_figures")

META_CSV = os.path.join(CKNNA_ROOT, "cknna_data", "performance",
                        "model_metadata.csv")

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


def load_combined_data():
    """Return dict[model][variant][horizon] = cknna_k10."""
    from collections import defaultdict
    data = defaultdict(lambda: defaultdict(dict))
    with open(COMBINED_CSV) as f:
        for r in csv.DictReader(f):
            if r["feats_B_type"] != "proprio_seq":
                continue
            data[r["model"]][r["feats_A_variant"]][int(r["horizon"])] = float(r["cknna_k10"])
    return data


def fig_cknna_vs_sr(data, model_meta):
    n_h = len(HORIZONS)
    fig, axes = plt.subplots(n_h, 3, figsize=(13, 4 * n_h))

    fig.suptitle(
        "All groups combined: proprio_seq CKNNA [DTW] vs WidowX SR\n"
        "(N=23286, S+M+L pooled, shared horizons {1,3,7,15})",
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
    out = os.path.join(FIG_DIR, "fig_cknna_vs_sr_ALL.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    model_meta = load_model_meta()
    data = load_combined_data()
    print(f"Models loaded: {len(data)}")
    print(f"Horizons: {HORIZONS}")
    fig_cknna_vs_sr(data, model_meta)


if __name__ == "__main__":
    main()
