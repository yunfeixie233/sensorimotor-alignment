"""
Plot alignment metrics vs success rate under different temporal horizons.

Produces:
1. Summary plot: Spearman rho vs horizon for each metric
2. Scatter grid: metric vs success rate at selected horizons
"""

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.environ.get("RECORD_DIR", "/home/ubuntu/verl/starVLA/cknna/record")
TEMPORAL_CSV = os.path.join(RECORD_DIR, "cknna_temporal_sweep.csv")
ORIG_CSV = os.path.join(RECORD_DIR, "cknna_action_proprio_simplerenv.csv")
FIG_DIR = os.path.join(RECORD_DIR, "runs", "temporal_sweep")
os.makedirs(FIG_DIR, exist_ok=True)

HORIZONS = [0, 1, 2, 3, 4, 5, 10, 15]
K = 10

NAME_MAP = {
    "Qwen2.5-GR00T-Bridge":        "Qwen-GR00T-Bridge",
    "Qwen2.5-GR00T-Bridge-RT-1":   "Qwen-GR00T-Bridge-RT-1",
    "Qwen3-GR00T-Bridge-RT-1":     "Qwen3VL-GR00T-Bridge-RT-1",
    "Qwen2.5-FAST-Bridge-RT-1":    "Qwen-FAST-Bridge-RT-1",
    "Qwen2.5-OFT-Bridge-RT-1":     "Qwen-OFT-Bridge-RT-1",
    "Qwen3-OFT-Bridge-RT-1":       "Qwen3VL-OFT-Bridge-RT-1",
    "spatialvla-sft-bridge":        "spatialvla-sft-bridge",
    "pi0-lerobot-bridge":           "pi0-lerobot-bridge",
    "openvla-7b-bridge":            "openvla-7b-bridge",
    "openvla-7b-bridge-ft-200k":    "openvla-7b-bridge-ft-200k",
    "RT-1-X":                       "rt1x-bridge",
    "CogACT-Small":                 "cogact-small-bridge",
    "CogACT-Base":                  "cogact-base-bridge",
    "CogACT-Large":                 "cogact-large-bridge",
    "GR00T-N1.5-Bridge":            "groot-n15-bridge",
    "Octo-base":                    "octo-base-bridge",
    "GR00T-N1.6-Bridge":            "groot-n16-bridge",
}

SHORT = {
    "Qwen-GR00T-Bridge":          "GR00T",
    "Qwen-GR00T-Bridge-RT-1":     "GR00T-RT1",
    "Qwen3VL-GR00T-Bridge-RT-1":  "Q3-GR00T",
    "Qwen-FAST-Bridge-RT-1":      "FAST",
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

DROP_ACTION = {"rt1x-bridge", "octo-base-bridge"}


def load_success_rates():
    """Load model -> success_rate from original CSV."""
    sr = {}
    with open(ORIG_CSV) as f:
        for r in csv.DictReader(f):
            orig_name = r["Model"]
            if orig_name not in NAME_MAP:
                continue
            val = r["WidowX_avg_entire"].strip()
            if val in ("", "N/A"):
                continue
            sr[NAME_MAP[orig_name]] = float(val)
    return sr


def load_temporal():
    """Load temporal sweep CSV into nested dict: data[(metric_type, horizon, model)] = {col: val}."""
    data = {}
    with open(TEMPORAL_CSV) as f:
        for r in csv.DictReader(f):
            key = (r["MetricType"], int(r["Horizon"]), r["Model"])
            data[key] = r
    return data


def get_xy(temporal, success_rates, metric_type, horizon, col, drop_models=None):
    """Get aligned (success_rate, metric_value, label) arrays."""
    xs, ys, labels = [], [], []
    for model, sr in success_rates.items():
        if drop_models and model in drop_models:
            continue
        key = (metric_type, horizon, model)
        if key not in temporal:
            continue
        val = temporal[key][col]
        if val.strip() in ("", "N/A"):
            continue
        xs.append(sr)
        ys.append(float(val))
        labels.append(SHORT.get(model, model))
    return np.array(xs), np.array(ys), labels


def plot_spearman_vs_horizon(temporal, success_rates):
    """Summary plot: Spearman rho vs horizon for each metric."""
    metrics = [
        ("proprio", f"CKNNA_k{K}", "CKNNA_proprio", None, "tab:blue", "o"),
        ("action",  f"CKNNA_k{K}", "CKNNA_action",  DROP_ACTION, "tab:red", "s"),
        ("proprio", f"MutualKNN_k{K}", "MutualKNN_proprio", None, "tab:cyan", "^"),
        ("action",  f"MutualKNN_k{K}", "MutualKNN_action",  DROP_ACTION, "tab:orange", "D"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for mt, col, label, drop, color, marker in metrics:
        rhos, pvals = [], []
        for h in HORIZONS:
            xs, ys, _ = get_xy(temporal, success_rates, mt, h, col, drop)
            if len(xs) < 3:
                rhos.append(np.nan)
                pvals.append(np.nan)
                continue
            rho, pval = spearmanr(xs, ys)
            rhos.append(rho)
            pvals.append(pval)

        axes[0].plot(HORIZONS, rhos, marker=marker, label=label, color=color, linewidth=2, markersize=6)
        axes[1].plot(HORIZONS, [-np.log10(p) if p > 0 else 5.0 for p in pvals],
                     marker=marker, label=label, color=color, linewidth=2, markersize=6)

    axes[0].set_xlabel("Horizon a", fontsize=12)
    axes[0].set_ylabel("Spearman rho", fontsize=12)
    axes[0].set_title(f"Correlation with Success Rate vs Horizon (k={K})", fontsize=13)
    axes[0].set_xticks(HORIZONS)
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Horizon a", fontsize=12)
    axes[1].set_ylabel("-log10(p-value)", fontsize=12)
    axes[1].set_title(f"Statistical Significance vs Horizon (k={K})", fontsize=13)
    axes[1].set_xticks(HORIZONS)
    axes[1].axhline(y=-np.log10(0.05), color="red", linestyle="--", alpha=0.5, label="p=0.05")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"spearman_vs_horizon_k{K}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_scatter_grid(temporal, success_rates):
    """Grid of scatter plots: rows = metrics, cols = horizons."""
    selected_horizons = [0, 1, 3, 5, 10, 15]
    metric_specs = [
        ("proprio", f"CKNNA_k{K}", "CKNNA_proprio", None),
        ("action",  f"CKNNA_k{K}", "CKNNA_action", DROP_ACTION),
        ("proprio", f"MutualKNN_k{K}", "MutualKNN_proprio", None),
        ("action",  f"MutualKNN_k{K}", "MutualKNN_action", DROP_ACTION),
    ]

    n_rows = len(metric_specs)
    n_cols = len(selected_horizons)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))

    for ri, (mt, col, ylabel, drop) in enumerate(metric_specs):
        for ci, h in enumerate(selected_horizons):
            ax = axes[ri, ci]
            xs, ys, labels = get_xy(temporal, success_rates, mt, h, col, drop)

            if len(xs) < 3:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center")
                continue

            ax.scatter(xs, ys, s=30, zorder=5, edgecolors="k", linewidths=0.4, alpha=0.8)

            rho, pval = spearmanr(xs, ys)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."

            slope, intercept = np.polyfit(xs, ys, 1)
            xline = np.linspace(xs.min() - 2, xs.max() + 2, 50)
            ax.plot(xline, slope * xline + intercept, "--", color="gray", linewidth=0.8, zorder=3)

            for x, y, lab in zip(xs, ys, labels):
                ax.annotate(lab, (x, y), textcoords="offset points", xytext=(3, 3),
                            fontsize=5, color="0.4")

            ax.set_title(f"a={h}  rho={rho:.3f} {sig}", fontsize=8)

            if ci == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            if ri == n_rows - 1:
                ax.set_xlabel("Success Rate (%)", fontsize=8)

            ax.tick_params(labelsize=6)

    fig.suptitle(f"Alignment vs Success Rate at Different Horizons (k={K})", fontsize=14, y=1.01)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"scatter_grid_vs_success_k{K}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_scatter_per_metric(temporal, success_rates):
    """Individual scatter plots per metric, showing all horizons as color-coded subplots."""
    metric_specs = [
        ("proprio", f"CKNNA_k{K}", "CKNNA_proprio", None),
        ("action",  f"CKNNA_k{K}", "CKNNA_action", DROP_ACTION),
        ("proprio", f"MutualKNN_k{K}", "MutualKNN_proprio", None),
        ("action",  f"MutualKNN_k{K}", "MutualKNN_action", DROP_ACTION),
    ]

    for mt, col, metric_label, drop in metric_specs:
        n_h = len(HORIZONS)
        n_cols = 4
        n_rows = (n_h + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
        axes_flat = axes.flatten()

        for hi, h in enumerate(HORIZONS):
            ax = axes_flat[hi]
            xs, ys, labels = get_xy(temporal, success_rates, mt, h, col, drop)

            if len(xs) < 3:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center")
                continue

            ax.scatter(xs, ys, s=50, zorder=5, edgecolors="k", linewidths=0.5, alpha=0.85)

            rho, pval = spearmanr(xs, ys)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."

            slope, intercept = np.polyfit(xs, ys, 1)
            xline = np.linspace(xs.min() - 2, xs.max() + 2, 50)
            ax.plot(xline, slope * xline + intercept, "--", color="gray", linewidth=1, zorder=3)

            for x, y, lab in zip(xs, ys, labels):
                ax.annotate(lab, (x, y), textcoords="offset points", xytext=(4, 4),
                            fontsize=6.5, color="0.3")

            ax.set_title(f"horizon a={h}\nrho={rho:.3f}  p={pval:.4f} {sig}", fontsize=10)
            ax.set_xlabel("Success Rate (%)", fontsize=9)
            ax.set_ylabel(f"{metric_label} (k={K})", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)

        for hi in range(len(HORIZONS), len(axes_flat)):
            axes_flat[hi].set_visible(False)

        fig.suptitle(f"{metric_label} vs Success Rate at Different Horizons (k={K})", fontsize=14)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, f"scatter_{metric_label}_vs_success_k{K}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


def main():
    success_rates = load_success_rates()
    temporal = load_temporal()

    print(f"Models with success rates: {len(success_rates)}")
    for m, sr in sorted(success_rates.items(), key=lambda x: -x[1]):
        print(f"  {SHORT.get(m, m):15s}  {sr:.1f}%")
    print()

    plot_spearman_vs_horizon(temporal, success_rates)
    plot_scatter_grid(temporal, success_rates)
    plot_scatter_per_metric(temporal, success_rates)

    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
