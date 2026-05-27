"""
Compare inclusive vs future-only proprio CKNNA:
  inclusive:   CKNNA(VLM_t, [proprio_t, ..., proprio_{t+a}])   -- previous sweep
  future-only: CKNNA(VLM_t, [proprio_{t+1}, ..., proprio_{t+a}]) -- new sweep

Produces:
  1. Line plot: CKNNA vs horizon for both modes, k=10 (all models)
  2. Line plot: CKNNA vs horizon normalized to a=0 inclusive (future starts at a=1)
  3. Spearman rho vs horizon for both modes vs success rate
  4. Scatter grid: future-only vs success rate at selected horizons
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
RESULTS_DIR = os.path.join(RECORD_DIR, "..", "cknna_data", "temporal_cknna_results")
ORIG_CSV = os.path.join(RECORD_DIR, "cknna_action_proprio_simplerenv.csv")
FIG_DIR = os.path.join(RECORD_DIR, "runs", "temporal_sweep")
os.makedirs(FIG_DIR, exist_ok=True)

K = 10
HORIZONS_INC = [0, 1, 2, 3, 4, 5, 10, 15]   # inclusive sweep
HORIZONS_FUT = [1, 2, 3, 4, 5, 10, 15]       # future-only (no a=0 since empty)

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

GROUPS = {
    "StarVLA-GR00T": (["Qwen-GR00T-Bridge", "Qwen-GR00T-Bridge-RT-1", "Qwen3VL-GR00T-Bridge-RT-1"], plt.cm.Blues),
    "StarVLA-Other": (["Qwen-FAST-Bridge-RT-1", "Qwen-OFT-Bridge-RT-1", "Qwen3VL-OFT-Bridge-RT-1"], plt.cm.Oranges),
    "SpatialVLA/Pi0/OpenVLA": (["spatialvla-sft-bridge", "pi0-lerobot-bridge",
                                  "openvla-7b-bridge", "openvla-7b-bridge-ft-200k"], plt.cm.Greens),
    "CogACT": (["cogact-small-bridge", "cogact-base-bridge", "cogact-large-bridge"], plt.cm.Purples),
    "GR00T-NVIDIA": (["groot-n15-bridge", "groot-n16-bridge"], plt.cm.Reds),
    "Other": (["rt1x-bridge", "octo-base-bridge"], plt.cm.Greys),
}


def load_json_results(prefix, horizons):
    """Load per-horizon JSON into dict[horizon][model] = entry."""
    data = {}
    for h in horizons:
        path = os.path.join(RESULTS_DIR, f"{prefix}{h}.json")
        with open(path) as f:
            data[h] = json.load(f)["models"]
    return data


def load_success_rates():
    sr = {}
    with open(ORIG_CSV) as f:
        for r in csv.DictReader(f):
            orig = r["Model"]
            if orig not in NAME_MAP:
                continue
            val = r["WidowX_avg_entire"].strip()
            if val in ("", "N/A"):
                continue
            sr[NAME_MAP[orig]] = float(val)
    return sr


def get_models(data, horizons):
    return sorted(data[horizons[0]].keys())


def plot_inclusive_vs_future(inc_data, fut_data):
    """Side-by-side line plots: inclusive vs future-only per model."""
    models = get_models(inc_data, HORIZONS_INC)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)

    for ax, (data, horizons, title) in zip(axes, [
        (inc_data, HORIZONS_INC, f"Inclusive: CKNNA(VLM_t, proprio_{{t..t+a}})"),
        (fut_data, HORIZONS_FUT, f"Future-only: CKNNA(VLM_t, proprio_{{t+1..t+a}})"),
    ]):
        for group_name, (group_models, cmap) in GROUPS.items():
            n = len(group_models)
            for idx, m in enumerate(group_models):
                if m not in models:
                    continue
                color = cmap(0.4 + 0.5 * idx / max(n - 1, 1))
                ys = [data[h][m][f"cknna_k{K}"] for h in horizons]
                ax.plot(horizons, ys, marker="o", markersize=4, label=SHORT.get(m, m),
                        color=color, linewidth=1.5)

        ax.set_xlabel("Horizon a", fontsize=12)
        ax.set_ylabel(f"CKNNA (k={K})", fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(horizons)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.02, 0.9), loc="upper left", fontsize=8)
    fig.suptitle("Inclusive vs Future-only Proprio CKNNA (k=10, N=5000)", fontsize=14)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"future_proprio_inclusive_vs_future_k{K}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_delta(inc_data, fut_data):
    """
    For each model and each a in HORIZONS_FUT, compute:
      delta = future_only(a) - inclusive(0)
    Shows how much extra predictive power comes purely from future states,
    over and above knowing the current state.
    """
    models = get_models(inc_data, HORIZONS_INC)
    fig, ax = plt.subplots(figsize=(12, 7))

    for group_name, (group_models, cmap) in GROUPS.items():
        n = len(group_models)
        for idx, m in enumerate(group_models):
            if m not in models:
                continue
            color = cmap(0.4 + 0.5 * idx / max(n - 1, 1))
            baseline = inc_data[0][m][f"cknna_k{K}"]
            ys = [fut_data[h][m][f"cknna_k{K}"] - baseline for h in HORIZONS_FUT]
            ax.plot(HORIZONS_FUT, ys, marker="o", markersize=4, label=SHORT.get(m, m),
                    color=color, linewidth=1.5)

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Horizon a", fontsize=12)
    ax.set_ylabel(f"CKNNA_future(a) - CKNNA_inclusive(0)  k={K}", fontsize=12)
    ax.set_title("Future-only CKNNA gain over current-state baseline (k=10, N=5000)", fontsize=13)
    ax.set_xticks(HORIZONS_FUT)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"future_proprio_delta_from_t0_k{K}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_spearman_comparison(inc_data, fut_data, success_rates):
    """Spearman rho vs horizon for inclusive and future-only."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for mode, data, horizons, color, label in [
        ("inclusive", inc_data, HORIZONS_INC, "tab:blue", "Inclusive [t..t+a]"),
        ("future",    fut_data, HORIZONS_FUT, "tab:red",  "Future-only [t+1..t+a]"),
    ]:
        rhos, pvals = [], []
        for h in horizons:
            xs, ys = [], []
            for m, sr in success_rates.items():
                if m not in data[h]:
                    continue
                xs.append(sr)
                ys.append(data[h][m][f"cknna_k{K}"])
            if len(xs) < 3:
                rhos.append(np.nan); pvals.append(np.nan); continue
            rho, pval = spearmanr(xs, ys)
            rhos.append(rho); pvals.append(pval)

        axes[0].plot(horizons, rhos, marker="o", label=label, color=color, linewidth=2, markersize=6)
        axes[1].plot(horizons, [-np.log10(p) if p > 0 else 5.0 for p in pvals],
                     marker="o", label=label, color=color, linewidth=2, markersize=6)

    axes[0].set_xlabel("Horizon a", fontsize=12)
    axes[0].set_ylabel("Spearman rho", fontsize=12)
    axes[0].set_title(f"CKNNA_proprio vs Success Rate  (k={K})", fontsize=13)
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Horizon a", fontsize=12)
    axes[1].set_ylabel("-log10(p-value)", fontsize=12)
    axes[1].set_title(f"Statistical Significance  (k={K})", fontsize=13)
    axes[1].axhline(y=-np.log10(0.05), color="red", linestyle="--", alpha=0.5, label="p=0.05")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Inclusive vs Future-only: Spearman Correlation with Success Rate", fontsize=14)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"future_proprio_spearman_comparison_k{K}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_scatter_future(fut_data, success_rates):
    """Scatter: future-only CKNNA vs success rate at each horizon (all 7 horizons, 2 rows)."""
    all_horizons = HORIZONS_FUT  # [1,2,3,4,5,10,15]
    n_cols = 4
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5.5 * n_rows))
    axes_flat = axes.flatten()

    for i, h in enumerate(all_horizons):
        ax = axes_flat[i]
        xs, ys, labels = [], [], []
        for m, sr in success_rates.items():
            if m not in fut_data[h]:
                continue
            xs.append(sr)
            ys.append(fut_data[h][m][f"cknna_k{K}"])
            labels.append(SHORT.get(m, m))

        xs = np.array(xs); ys = np.array(ys)
        ax.scatter(xs, ys, s=80, zorder=5, edgecolors="k", linewidths=0.6, alpha=0.85)
        rho, pval = spearmanr(xs, ys)
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
        slope, intercept = np.polyfit(xs, ys, 1)
        xline = np.linspace(xs.min() - 2, xs.max() + 2, 50)
        ax.plot(xline, slope * xline + intercept, "--", color="gray", linewidth=1.2)
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 5),
                        fontsize=8, color="0.25")
        ax.set_title(f"a={h}  (future t+1 to t+{h})\nrho={rho:.3f}  p={pval:.4f}  {sig}", fontsize=11)
        ax.set_xlabel("SimplerEnv WidowX Success Rate (%)", fontsize=10)
        ax.set_ylabel(f"CKNNA_future_proprio (k={K})", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.25)

    # hide the unused 8th panel
    axes_flat[-1].set_visible(False)

    fig.suptitle(f"Future-only Proprio CKNNA vs Success Rate (k={K}, N=5000)\n"
                 f"feats_B = concat[proprio_{{t+1}}, ..., proprio_{{t+a}}]", fontsize=14)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"scatter_future_proprio_vs_success_k{K}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_head_to_head(inc_data, fut_data, success_rates):
    """
    Head-to-head at each shared horizon:
    x-axis = inclusive CKNNA, y-axis = future-only CKNNA, one point per model.
    Shows where models gain or lose when removing t from the window.
    """
    shared = [1, 3, 5, 10, 15]
    n = len(shared)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))

    for ax, h in zip(axes, shared):
        xs, ys, labels = [], [], []
        for m in inc_data[h]:
            if m not in fut_data[h]:
                continue
            xs.append(inc_data[h][m][f"cknna_k{K}"])
            ys.append(fut_data[h][m][f"cknna_k{K}"])
            labels.append(SHORT.get(m, m))

        xs = np.array(xs); ys = np.array(ys)
        ax.scatter(xs, ys, s=50, zorder=5, edgecolors="k", linewidths=0.5, alpha=0.85)
        mn = min(xs.min(), ys.min()) - 0.002
        mx = max(xs.max(), ys.max()) + 0.002
        ax.plot([mn, mx], [mn, mx], "--", color="gray", linewidth=0.8, label="y=x")
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(3, 3), fontsize=6, color="0.3")
        frac_above = (ys > xs).mean()
        ax.set_title(f"a={h}: future vs inclusive\n{frac_above*100:.0f}% above y=x", fontsize=10)
        ax.set_xlabel(f"CKNNA inclusive [t..t+{h}]", fontsize=8)
        ax.set_ylabel(f"CKNNA future [t+1..t+{h}]", fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"Head-to-head: Inclusive vs Future-only CKNNA_proprio (k={K})", fontsize=14)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"future_proprio_head_to_head_k{K}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    inc_data = load_json_results("cknna_proprio_horizon_", HORIZONS_INC)
    fut_data = load_json_results("cknna_proprio_future_horizon_", HORIZONS_FUT)
    success_rates = load_success_rates()

    print(f"Inclusive horizons: {HORIZONS_INC}")
    print(f"Future-only horizons: {HORIZONS_FUT}")
    print(f"Models: {len(get_models(inc_data, HORIZONS_INC))}")
    print(f"Success rates loaded: {len(success_rates)}")
    print()

    plot_inclusive_vs_future(inc_data, fut_data)
    plot_delta(inc_data, fut_data)
    plot_spearman_comparison(inc_data, fut_data, success_rates)
    plot_scatter_future(fut_data, success_rates)
    plot_head_to_head(inc_data, fut_data, success_rates)

    print(f"\nAll saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
