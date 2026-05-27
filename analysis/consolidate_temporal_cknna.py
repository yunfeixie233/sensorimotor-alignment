"""
Consolidate temporal CKNNA sweep results into a single CSV and produce
line plots showing CKNNA vs horizon for each model.

Reads JSON files from temporal_cknna_results/ and outputs:
  - record/cknna_temporal_sweep.csv
  - record/runs/temporal_cknna_proprio_k10.png
  - record/runs/temporal_cknna_action_k10.png

Usage:
  python record/scripts/consolidate_temporal_cknna.py
"""

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.environ.get("RECORD_DIR", "/home/ubuntu/verl/starVLA/cknna/record")
RESULTS_DIR = os.path.join(RECORD_DIR, "..", "cknna_data", "temporal_cknna_results")
CSV_OUT = os.path.join(RECORD_DIR, "cknna_temporal_sweep.csv")
FIG_DIR = os.path.join(RECORD_DIR, "runs", "temporal_sweep")
os.makedirs(FIG_DIR, exist_ok=True)

HORIZONS = [0, 1, 2, 3, 4, 5, 10, 15]
K_VALUES = [5, 10, 20]

SHORT = {
    "Qwen-FAST-Bridge-RT-1":      "FAST",
    "Qwen-GR00T-Bridge":          "GR00T",
    "Qwen-GR00T-Bridge-RT-1":     "GR00T-RT1",
    "Qwen-OFT-Bridge-RT-1":       "OFT",
    "Qwen3VL-GR00T-Bridge-RT-1":  "Q3-GR00T",
    "Qwen3VL-OFT-Bridge-RT-1":    "Q3-OFT",
    "spatialvla-sft-bridge":       "SpatialVLA",
    "pi0-lerobot-bridge":          "Pi0",
    "openvla-7b-bridge":           "OpenVLA",
    "openvla-7b-bridge-ft-200k":   "OpenVLA-FT",
    "rt1x-bridge":                 "RT-1-X",
    "octo-base-bridge":            "Octo",
    "cogact-base-bridge":          "CogACT-B",
    "cogact-large-bridge":         "CogACT-L",
    "cogact-small-bridge":         "CogACT-S",
    "groot-n15-bridge":            "GR00T-N1.5",
    "groot-n16-bridge":            "GR00T-N1.6",
}

GROUPS = {
    "StarVLA-GR00T": ["Qwen-GR00T-Bridge", "Qwen-GR00T-Bridge-RT-1", "Qwen3VL-GR00T-Bridge-RT-1"],
    "StarVLA-Other": ["Qwen-FAST-Bridge-RT-1", "Qwen-OFT-Bridge-RT-1", "Qwen3VL-OFT-Bridge-RT-1"],
    "SpatialVLA/Pi0/OpenVLA": ["spatialvla-sft-bridge", "pi0-lerobot-bridge",
                                "openvla-7b-bridge", "openvla-7b-bridge-ft-200k"],
    "CogACT": ["cogact-small-bridge", "cogact-base-bridge", "cogact-large-bridge"],
    "GR00T-NVIDIA": ["groot-n15-bridge", "groot-n16-bridge"],
    "Other": ["rt1x-bridge", "octo-base-bridge"],
}

GROUP_COLORS = {
    "StarVLA-GR00T": plt.cm.Blues,
    "StarVLA-Other": plt.cm.Oranges,
    "SpatialVLA/Pi0/OpenVLA": plt.cm.Greens,
    "CogACT": plt.cm.Purples,
    "GR00T-NVIDIA": plt.cm.Reds,
    "Other": plt.cm.Greys,
}


def load_all_results():
    """Load all JSON results into a nested dict: data[metric_type][horizon][model] = {...}."""
    data = {"proprio": {}, "action": {}}
    for metric_type in ("proprio", "action"):
        for h in HORIZONS:
            path = os.path.join(RESULTS_DIR, f"cknna_{metric_type}_horizon_{h}.json")
            with open(path) as f:
                j = json.load(f)
            data[metric_type][h] = j["models"]
    return data


def write_csv(data):
    models = sorted(data["proprio"][0].keys())
    fieldnames = ["Model", "ShortName", "MetricType", "Horizon"]
    for k in K_VALUES:
        fieldnames.append(f"CKNNA_k{k}")
        fieldnames.append(f"MutualKNN_k{k}")

    rows = []
    for metric_type in ("proprio", "action"):
        for h in HORIZONS:
            for m in models:
                entry = data[metric_type][h][m]
                row = {
                    "Model": m,
                    "ShortName": SHORT.get(m, m),
                    "MetricType": metric_type,
                    "Horizon": h,
                }
                for k in K_VALUES:
                    row[f"CKNNA_k{k}"] = f"{entry[f'cknna_k{k}']:.6f}"
                    row[f"MutualKNN_k{k}"] = f"{entry[f'mutual_knn_k{k}']:.6f}"
                rows.append(row)

    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"CSV saved: {CSV_OUT} ({len(rows)} rows)")


def plot_temporal(data, metric_type, k):
    models = sorted(data[metric_type][0].keys())
    fig, ax = plt.subplots(figsize=(12, 7))

    for group_name, group_models in GROUPS.items():
        cmap = GROUP_COLORS[group_name]
        n = len(group_models)
        for idx, m in enumerate(group_models):
            if m not in models:
                continue
            color = cmap(0.4 + 0.5 * idx / max(n - 1, 1))
            ys = [data[metric_type][h][m][f"cknna_k{k}"] for h in HORIZONS]
            label = SHORT.get(m, m)
            ax.plot(HORIZONS, ys, marker="o", markersize=4, label=label, color=color, linewidth=1.5)

    ax.set_xlabel("Horizon a (number of future timesteps)", fontsize=12)
    ylabel = f"CKNNA (VLM_t, {metric_type}_{{t:t+a}})  k={k}"
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"Temporal CKNNA -- {metric_type.capitalize()} (k={k}, N=5000)", fontsize=13)
    ax.set_xticks(HORIZONS)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, ncol=1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(FIG_DIR, f"temporal_cknna_{metric_type}_k{k}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def plot_normalized(data, metric_type, k):
    """Plot CKNNA normalized to a=0 baseline (shows relative change)."""
    models = sorted(data[metric_type][0].keys())
    fig, ax = plt.subplots(figsize=(12, 7))

    for group_name, group_models in GROUPS.items():
        cmap = GROUP_COLORS[group_name]
        n = len(group_models)
        for idx, m in enumerate(group_models):
            if m not in models:
                continue
            color = cmap(0.4 + 0.5 * idx / max(n - 1, 1))
            ys = [data[metric_type][h][m][f"cknna_k{k}"] for h in HORIZONS]
            baseline = ys[0]
            ys_norm = [y / baseline for y in ys]
            label = SHORT.get(m, m)
            ax.plot(HORIZONS, ys_norm, marker="o", markersize=4, label=label, color=color, linewidth=1.5)

    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Horizon a (number of future timesteps)", fontsize=12)
    ax.set_ylabel(f"CKNNA / CKNNA(a=0)  (k={k})", fontsize=12)
    ax.set_title(f"Temporal CKNNA Relative Change -- {metric_type.capitalize()} (k={k}, N=5000)", fontsize=13)
    ax.set_xticks(HORIZONS)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, ncol=1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(FIG_DIR, f"temporal_cknna_{metric_type}_k{k}_normalized.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def main():
    data = load_all_results()
    write_csv(data)

    for metric_type in ("proprio", "action"):
        for k in [10]:
            plot_temporal(data, metric_type, k)
            plot_normalized(data, metric_type, k)

    print("\nDone.")


if __name__ == "__main__":
    main()
