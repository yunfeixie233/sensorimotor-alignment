"""
Plot Discovery 4: "Last Mile Bottleneck" — feats_action vs real_action CKNNA.

Reads:
  record/exp_horizon/cknna_data_exp1v2_full_L/feats_action/results_feats_action_groupL.csv

Produces a 2-row × 3-col figure:
  Row 1: CKNNA(VLM_t, feats_action_t) vs Success Rate
  Row 2: CKNNA(VLM_t, real_action_t)  vs Success Rate
  Cols:   img+text | image only | text only

Output:
  record/exp_horizon/discovery4_figures/fig_feats_action_vs_sr_groupL.png
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKNNA_ROOT = os.environ.get("DATA_STORE", "/home/ubuntu/verl/starVLA/cknna")

RESULTS_CSV = os.path.join(SCRIPT_DIR, "cknna_data_exp1v2_full_L", "feats_action",
                           "results_feats_action_groupL.csv")
META_CSV    = os.path.join(CKNNA_ROOT, "cknna_data", "performance", "model_metadata.csv")
FIG_DIR     = os.path.join(SCRIPT_DIR, "discovery4_figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Color card (matches plot_exp1v2_groups_full.py)
# ---------------------------------------------------------------------------
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
FA_LABELS   = {"imgtext": "img + text", "img": "image only", "txt": "text only"}

FEATS_B_TYPES = ["feats_action_t", "real_action_t"]
FEATS_B_LABELS = {
    "feats_action_t": "CKNNA(VLM, action-head features)",
    "real_action_t":  "CKNNA(VLM, real robot actions)",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_model_meta():
    """Returns dict[model_key] = {sr, group, label} for finetune_on_bridge=yes models."""
    meta = {}
    with open(META_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["finetune_on_bridge"] != "yes":
                continue
            if row["exclude_always"].lower() == "true":
                continue
            key = row["model_key"]
            if key not in meta:  # keep first occurrence
                meta[key] = {
                    "sr":    float(row["success_rate"]),
                    "group": row["group"],
                    "label": row["figure_label"],
                }
    return meta


def load_results():
    """Returns dict[model][feats_A_variant][feats_B_type] = cknna_k10."""
    data = {}
    with open(RESULTS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            m = row["model"]
            fa = row["feats_A_variant"]
            fb = row["feats_B_type"]
            val = float(row["cknna_k10"])
            data.setdefault(m, {}).setdefault(fa, {})[fb] = val
    return data


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_figure(cknna_data, model_meta):
    n_rows = len(FEATS_B_TYPES)   # 2
    n_cols = len(FA_VARIANTS)     # 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10))

    for ri, fb_type in enumerate(FEATS_B_TYPES):
        for ci, fa_var in enumerate(FA_VARIANTS):
            ax = axes[ri, ci]

            xs, ys = [], []
            for model_key, meta in model_meta.items():
                if model_key not in cknna_data:
                    continue
                if fa_var not in cknna_data[model_key]:
                    continue
                if fb_type not in cknna_data[model_key][fa_var]:
                    continue
                val  = cknna_data[model_key][fa_var][fb_type]
                sr   = meta["sr"]
                grp  = meta["group"]
                card = COLOR_CARD.get(grp, {"color": "#333333", "marker": "o"})
                ax.scatter(val, sr, c=card["color"], marker=card["marker"],
                           s=90, edgecolors="k", linewidths=0.5, zorder=3)
                ax.annotate(meta["label"], (val, sr),
                            textcoords="offset points", xytext=(4, 2),
                            fontsize=6, color=card["color"])
                xs.append(val)
                ys.append(sr)

            if len(xs) >= 3:
                xs_arr = np.array(xs)
                ys_arr = np.array(ys)
                rho, pval = spearmanr(xs_arr, ys_arr)
                sig = ("***" if pval < 0.001 else
                       "**"  if pval < 0.01  else
                       "*"   if pval < 0.05  else "n.s.")
                slope, intercept = np.polyfit(xs_arr, ys_arr, 1)
                xline = np.linspace(xs_arr.min() - 0.005, xs_arr.max() + 0.005, 50)
                ax.plot(xline, slope * xline + intercept, "--", color="gray",
                        linewidth=1.0, zorder=2)
                ax.set_title(
                    f"{FA_LABELS[fa_var]}\nSpearman ρ={rho:.3f}  p={pval:.4f} {sig}  N={len(xs)}",
                    fontsize=10,
                )
            else:
                ax.set_title(FA_LABELS[fa_var], fontsize=10)

            if ci == 0:
                ax.set_ylabel(FEATS_B_LABELS[fb_type] + "\n\nWidowX SR (%)", fontsize=9)
            if ri == n_rows - 1:
                ax.set_xlabel("CKNNA (k=10)  [Group L, N=7762]", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.2)

    fig.suptitle(
        "Discovery 4 — Last Mile Bottleneck\n"
        "CKNNA(VLM_t, feats_action_t)  vs  CKNNA(VLM_t, real_action_t)  ·  Group L",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_feats_action_vs_sr_groupL.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def make_gap_figure(cknna_data, model_meta):
    """Bar chart showing feats_action vs real_action CKNNA side-by-side per model (imgtext)."""
    models_ordered = sorted(
        [k for k in model_meta if k in cknna_data],
        key=lambda k: -model_meta[k]["sr"],
    )

    labels, fa_vals, ra_vals, colors = [], [], [], []
    for mk in models_ordered:
        if "imgtext" not in cknna_data[mk]:
            continue
        d = cknna_data[mk]["imgtext"]
        if "feats_action_t" not in d or "real_action_t" not in d:
            continue
        labels.append(model_meta[mk]["label"])
        fa_vals.append(d["feats_action_t"])
        ra_vals.append(d["real_action_t"])
        grp = model_meta[mk]["group"]
        colors.append(COLOR_CARD.get(grp, {"color": "#333333"})["color"])

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    bars1 = ax.bar(x - w / 2, fa_vals, w, label="feats_action_t", color=colors, alpha=0.9,
                   edgecolor="k", linewidth=0.5)
    bars2 = ax.bar(x + w / 2, ra_vals, w, label="real_action_t",   color=colors, alpha=0.35,
                   edgecolor="k", linewidth=0.5, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("CKNNA (k=10, imgtext)", fontsize=10)
    ax.set_title(
        "Discovery 4: Action-Feature vs Real-Action CKNNA  [Group L, imgtext]\n"
        "Sorted by WidowX success rate (high → low)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_feats_action_gap_bar_groupL.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def print_summary(cknna_data, model_meta):
    print("\n=== Summary: feats_action_t vs real_action_t CKNNA (imgtext) ===")
    print(f"{'Model':<35} {'SR%':>5}  {'feats_action':>13}  {'real_action':>11}  {'ratio':>7}")
    print("-" * 80)
    models_sorted = sorted(
        [k for k in model_meta if k in cknna_data],
        key=lambda k: -model_meta[k]["sr"],
    )
    for mk in models_sorted:
        d = cknna_data[mk].get("imgtext", {})
        fa = d.get("feats_action_t", float("nan"))
        ra = d.get("real_action_t",  float("nan"))
        ratio = fa / ra if ra > 1e-6 else float("inf")
        sr = model_meta[mk]["sr"]
        print(f"{model_meta[mk]['label']:<35} {sr:>5.1f}  {fa:>13.5f}  {ra:>11.5f}  {ratio:>7.1f}x")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model_meta = load_model_meta()
    cknna_data  = load_results()
    print(f"Models in metadata (non-excluded): {len(model_meta)}")
    print(f"Models in results:                 {len(cknna_data)}")
    print()

    print_summary(cknna_data, model_meta)
    make_figure(cknna_data, model_meta)
    make_gap_figure(cknna_data, model_meta)
    print("\nDone.")
