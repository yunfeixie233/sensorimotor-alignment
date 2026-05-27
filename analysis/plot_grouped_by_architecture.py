"""
Scatter: alignment metric vs WidowX SR, models grouped by architecture family.

Groups (defined by model architecture, NOT training data):
  1. StarVLA  -- Qwen VLM directly converted to VLA
  2. OpenVLA / CogACT -- Prismatic (DINOv2+SigLIP) + Llama2-7B backbone
  3. PaLiGemma VLA -- PaLiGemma/PaLiGemma2 backbone (SpatialVLA, Pi0)
  4. Pre-VLM Generalist -- lightweight non-VLM (RT-1-X, Octo)
  5. NVIDIA GR00T N1.5/N1.6 -- NVEagle VLM
"""
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

csv_path = os.path.join(os.path.dirname(__file__), "..", "cknna_action_proprio_simplerenv.csv")
with open(csv_path) as f:
    rows = list(csv.DictReader(f))

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
    "RT-1-X": "RT-1-X",
    "CogACT-Small": "CogACT-S",
    "CogACT-Base": "CogACT-B",
    "CogACT-Large": "CogACT-L",
    "GR00T-N1.5-Bridge": "GR00T-N1.5",
    "GR00T-N1.6-Bridge": "GR00T-N1.6",
    "Octo-base": "Octo",
}

GROUPS = [
    {
        "label": "StarVLA (Qwen VLM2VLA)",
        "color": "#1f77b4",
        "marker": "o",
        "size": 60,
        "models": {
            "Qwen2.5-GR00T-Bridge",
            "Qwen2.5-GR00T-Bridge-RT-1",
            "Qwen3-GR00T-Bridge-RT-1",
            "Qwen2.5-FAST-Bridge-RT-1",
            "Qwen2.5-OFT-Bridge-RT-1",
            "Qwen3-OFT-Bridge-RT-1",
        },
    },
    {
        "label": "OpenVLA / CogACT (Prismatic+Llama2)",
        "color": "#ff7f0e",
        "marker": "D",
        "size": 70,
        "models": {
            "openvla-7b-bridge",
            "CogACT-Small",
            "CogACT-Base",
            "CogACT-Large",
        },
    },
    {
        "label": "PaLiGemma VLA",
        "color": "#2ca02c",
        "marker": "^",
        "size": 80,
        "models": {
            "spatialvla-sft-bridge",
            "pi0-lerobot-bridge",
        },
    },
    {
        "label": "Pre-VLM Generalist",
        "color": "#d62728",
        "marker": "s",
        "size": 65,
        "models": {
            "RT-1-X",
            "Octo-base",
        },
    },
    {
        "label": "NVIDIA GR00T (NVEagle)",
        "color": "#9467bd",
        "marker": "*",
        "size": 120,
        "models": {
            "GR00T-N1.5-Bridge",
            "GR00T-N1.6-Bridge",
        },
    },
]

ALL_MODELS = set()
for g in GROUPS:
    ALL_MODELS |= g["models"]

rows = [r for r in rows if r["Model"] in ALL_MODELS]

k = 10

panels = [
    ("CKNNA_proprio_k%d" % k, "CKNNA (VLM, Proprio)  k=%d" % k),
    ("CKNNA_action_k%d" % k, "CKNNA (VLM, Action)  k=%d" % k),
    ("MutualKNN_proprio_k%d" % k, "Mutual k-NN (VLM, Proprio)  k=%d" % k),
    ("MutualKNN_action_k%d" % k, "Mutual k-NN (VLM, Action)  k=%d" % k),
]

timestamp = time.strftime("%Y%m%d_%H%M%S")
outdir = os.path.join(os.path.dirname(__file__), "..", "runs", timestamp)
os.makedirs(outdir, exist_ok=True)


def _sig(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


for col, ylabel in panels:
    fig, ax = plt.subplots(figsize=(8.5, 6))

    xs_all, ys_all = [], []

    for grp in GROUPS:
        xs, ys, labs = [], [], []
        for r in rows:
            name = r["Model"]
            if name not in grp["models"]:
                continue
            x = float(r["WidowX_avg_entire"])
            y = float(r[col])
            xs.append(x)
            ys.append(y)
            labs.append(SHORT[name])

        if not xs:
            continue

        ax.scatter(xs, ys, s=grp["size"], zorder=5,
                   marker=grp["marker"], edgecolors="k", linewidths=0.5,
                   color=grp["color"], label=grp["label"])

        for x, y, lab in zip(xs, ys, labs):
            ax.annotate(lab, (x, y), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, color="0.3")

        xs_all.extend(xs)
        ys_all.extend(ys)

    xs_all = np.array(xs_all)
    ys_all = np.array(ys_all)

    rho, pval = spearmanr(xs_all, ys_all)
    slope, intercept = np.polyfit(xs_all, ys_all, 1)
    xline = np.linspace(xs_all.min() - 2, xs_all.max() + 2, 100)
    ax.plot(xline, slope * xline + intercept, "--", color="gray",
            linewidth=1, zorder=3)

    title_lines = [ylabel]
    title_lines.append("Spearman rho=%.3f  p=%.4f %s  (N=%d)"
                       % (rho, pval, _sig(pval), len(xs_all)))
    ax.set_title("\n".join(title_lines), fontsize=9)
    ax.set_xlabel("SimplerEnv WidowX Success Rate (%)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="best", framealpha=0.9)
    fig.tight_layout()

    fname = "grouped_%s_k%d.png" % (col, k)
    out = os.path.join(outdir, fname)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)

print("All figures saved to:", outdir)
