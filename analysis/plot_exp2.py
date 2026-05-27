"""
Plot Exp 2 results: extended horizon on long episodes (H_max=75, L>=76).

Figure A: CKNNA vs SR scatter for Exp 2 Group L
          rows = horizons {1,3,7,15,25,40,50,60,75}, cols = imgtext/img/txt

Figure B: CKNNA vs Horizon line plot
          Left subplot:  Exp 1 Group L (h up to 40)
          Right subplot: Exp 2       (h up to 75)
          Both show individual model lines + mean line, imgtext variant only.
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

EXP2_CSV = os.path.join(
    EXP_DIR, "cknna_data_exp2", "finetune",
    f"dtw_{DTW_VERSION}", f"results_groups_dtw_{DTW_VERSION}.csv",
)

# Exp 1 Group L full for comparison in Figure B
EXP1L_CSV = os.path.join(
    EXP_DIR, "cknna_data_exp1v2_full_L", "finetune",
    f"dtw_{DTW_VERSION}", f"results_groups_dtw_{DTW_VERSION}.csv",
)

EXP2_HORIZONS = [1, 3, 7, 15, 25, 40, 50, 60, 75]
EXP1L_HORIZONS = [1, 3, 7, 15, 25, 40]

FIG_DIR = os.path.join(EXP_DIR, "exp2_figures")

META_CSV = os.path.join(CKNNA_ROOT, "cknna_data", "performance", "model_metadata.csv")


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


FA_VARIANTS = ["imgtext", "img", "txt"]
FA_LABELS   = {"imgtext": "img+txt", "img": "image only", "txt": "text only"}

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

MODEL_SHORT = _DISPLAY_NAMES

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


def load_csv(path):
    """Return dict[model][variant][horizon] = cknna_k10 from a groups CSV."""
    data = defaultdict(lambda: defaultdict(dict))
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["feats_B_type"] != "proprio_seq":
                continue
            data[r["model"]][r["feats_A_variant"]][int(r["horizon"])] = float(r["cknna_k10"])
    return data


def _spearman_label(xs, ys):
    if len(xs) < 3:
        return "n<3", "N/A"
    rho, p = spearmanr(xs, ys)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
    p_str = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
    return f"rho={rho:.3f}  p={p_str}  {sig}", rho


def fig_a_cknna_vs_sr(data2, model_meta):
    """9 rows x 3 cols scatter: Exp 2 Group L CKNNA vs WidowX SR."""
    horizons = EXP2_HORIZONS
    n_h = len(horizons)
    fig, axes = plt.subplots(n_h, 3, figsize=(13, 4 * n_h))

    fig.suptitle(
        "Exp 2: Group L (L>=76, H_max=75)  proprio_seq CKNNA [DTW] vs WidowX SR\n"
        f"N=985 unique episodes, 79% sweep-into-pile",
        fontsize=11, y=1.005,
    )

    for ri, h in enumerate(horizons):
        for ci, fa in enumerate(FA_VARIANTS):
            ax = axes[ri, ci]
            xs, ys = [], []
            for model_key, meta in model_meta.items():
                if model_key not in data2:
                    continue
                if fa not in data2[model_key]:
                    continue
                if h not in data2[model_key][fa]:
                    continue
                val = data2[model_key][fa][h]
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
                sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "n.s."))
                ax.text(0.04, 0.97,
                        f"rho={rho:.3f}  p={p_str}  {sig}\nn={len(xs)}",
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
    out = os.path.join(FIG_DIR, "fig_exp2_cknna_vs_sr.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _load_bridge_keys():
    with open(META_CSV, newline="") as f:
        return {r["model_key"] for r in csv.DictReader(f)
                if r["finetune_on_bridge"] == "yes"}


def fig_b_cknna_vs_horizon(data1L, data2):
    """Single panel: Exp 2 only (h=1..75), imgtext variant.
    Only shows the 16 VLA fine-tuned-on-bridge models.
    """
    horizons = EXP2_HORIZONS
    vla_keys = _load_bridge_keys()
    models = sorted(m for m in data2.keys() if m in vla_keys)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))

    xs_all, ys_all = [], []
    for m in models:
        if "imgtext" not in data2[m]:
            continue
        ys = [data2[m]["imgtext"].get(h, np.nan) for h in horizons]
        color = FAMILY_COLOR.get(m, "#aaaaaa")
        ax.plot(horizons, ys, marker="o", markersize=4, linewidth=1.3,
                color=color, alpha=0.8, label=display_name(m))
        for h, y in zip(horizons, ys):
            if not np.isnan(y):
                xs_all.append(h)
                ys_all.append(y)

    means = []
    for h in horizons:
        vals = [data2[m]["imgtext"][h]
                for m in models
                if "imgtext" in data2[m] and h in data2[m]["imgtext"]]
        means.append(np.mean(vals) if vals else np.nan)
    ax.plot(horizons, means, marker="D", markersize=7, linewidth=2.8,
            color="black", zorder=10, label="Mean", linestyle="--")

    spearman_str, _ = _spearman_label(xs_all, ys_all)
    ax.set_title(
        f"Exp 2  (L=[76,119], N=985, $H_{{\\mathrm{{max}}}}$=75, 79% sweep-into-pile)\n"
        f"{spearman_str}",
        fontsize=11,
    )
    ax.set_xlabel("Prediction horizon $h$", fontsize=12)
    ax.set_ylabel("CKNNA  ($k$=10)", fontsize=12)
    ax.set_xticks(horizons)
    ax.set_xticklabels([f"t+1\nto\nt+{h}" for h in horizons], fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=10)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.99), loc="upper left",
               fontsize=8.5, framealpha=0.9, ncol=1)
    fig.suptitle(
        "CKNNA vs. Horizon -- Exp 2  (feats_A = imgtext, 16 models)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_exp2_cknna_vs_horizon.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print(f"Exp 2 CSV:      {EXP2_CSV}")
    print(f"Exp 1 Group L:  {EXP1L_CSV}")

    model_meta = load_model_meta()
    data2  = load_csv(EXP2_CSV)
    data1L = load_csv(EXP1L_CSV)

    print(f"Exp 2 models: {len(data2)}, horizons in data: "
          f"{sorted({h for m in data2.values() for v in m.values() for h in v})}")
    print(f"Exp 1 Group L models: {len(data1L)}")

    fig_a_cknna_vs_sr(data2, model_meta)
    fig_b_cknna_vs_horizon(data1L, data2)

    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
