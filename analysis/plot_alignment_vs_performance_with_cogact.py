"""
Plot 5a: Alignment vs WidowX performance for all models including CogACT + GR00T N1.5/N1.6.

Based on plot_alignment_vs_performance_qwen.py.
Adds CogACT-Small, CogACT-Base, CogACT-Large as OXE-pretrained (orange diamonds).
Bridge-finetuned models (including GR00T-N1.5 and GR00T-N1.6) shown as blue circles.
Reports Spearman correlation for All and Bridge-FT-only.
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

INCLUDE_MODELS = {
    "Qwen2.5-GR00T-Bridge",
    "Qwen2.5-GR00T-Bridge-RT-1",
    "Qwen3-GR00T-Bridge-RT-1",
    "Qwen2.5-FAST-Bridge-RT-1",
    "Qwen2.5-OFT-Bridge-RT-1",
    "Qwen3-OFT-Bridge-RT-1",
    "pi0-lerobot-bridge",
    "spatialvla-sft-bridge",
    "CogACT-Small",
    "CogACT-Base",
    "CogACT-Large",
    "GR00T-N1.5-Bridge",
    "GR00T-N1.6-Bridge",
}

rows = [r for r in rows if r["Model"] in INCLUDE_MODELS]

OXE_PRETRAINED = {"CogACT-Small", "CogACT-Base", "CogACT-Large"}

k = 10

panels = [
    ("CKNNA_proprio_k%d" % k, "CKNNA (VLM, Proprio)  k=%d" % k),
    ("CKNNA_action_k%d" % k, "CKNNA (VLM, Action)  k=%d" % k),
    ("MutualKNN_proprio_k%d" % k, "Mutual k-NN (VLM, Proprio)  k=%d" % k),
    ("MutualKNN_action_k%d" % k, "Mutual k-NN (VLM, Action)  k=%d" % k),
]

SHORT = {
    "Qwen2.5-GR00T-Bridge": "GR00T",
    "Qwen2.5-GR00T-Bridge-RT-1": "GR00T-RT1",
    "Qwen3-GR00T-Bridge-RT-1": "Q3-GR00T",
    "Qwen2.5-FAST-Bridge-RT-1": "FAST",
    "Qwen2.5-OFT-Bridge-RT-1": "OFT",
    "Qwen3-OFT-Bridge-RT-1": "Q3-OFT",
    "pi0-lerobot-bridge": "Pi0",
    "spatialvla-sft-bridge": "SpatialVLA",
    "CogACT-Small": "CogACT-S",
    "CogACT-Base": "CogACT-B",
    "CogACT-Large": "CogACT-L",
    "GR00T-N1.5-Bridge": "GR00T-N1.5",
    "GR00T-N1.6-Bridge": "GR00T-N1.6",
}

timestamp = time.strftime("%Y%m%d_%H%M%S")
outdir = os.path.join(os.path.dirname(__file__), "..", "runs", timestamp)
os.makedirs(outdir, exist_ok=True)

for col, ylabel in panels:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    xs_ft, ys_ft, labs_ft = [], [], []
    xs_oxe, ys_oxe, labs_oxe = [], [], []

    for r in rows:
        name = r["Model"]
        x = float(r["WidowX_avg_entire"])
        y = float(r[col])
        lab = SHORT[name]
        if name in OXE_PRETRAINED:
            xs_oxe.append(x)
            ys_oxe.append(y)
            labs_oxe.append(lab)
        else:
            xs_ft.append(x)
            ys_ft.append(y)
            labs_ft.append(lab)

    xs_ft = np.array(xs_ft)
    ys_ft = np.array(ys_ft)
    xs_oxe = np.array(xs_oxe)
    ys_oxe = np.array(ys_oxe)
    xs_all = np.concatenate([xs_ft, xs_oxe])
    ys_all = np.concatenate([ys_ft, ys_oxe])

    ax.scatter(xs_ft, ys_ft, s=60, zorder=5, marker="o",
               edgecolors="k", linewidths=0.5, color="tab:blue", label="Bridge-finetuned")
    if len(xs_oxe) > 0:
        ax.scatter(xs_oxe, ys_oxe, s=80, zorder=5, marker="D",
                   edgecolors="k", linewidths=0.5, color="tab:orange",
                   label="OXE-pretrained (CogACT)")

    rho_all, pval_all = spearmanr(xs_all, ys_all)
    slope, intercept = np.polyfit(xs_all, ys_all, 1)
    xline = np.linspace(xs_all.min() - 2, xs_all.max() + 2, 100)
    ax.plot(xline, slope * xline + intercept, "--", color="gray", linewidth=1, zorder=3)

    if len(xs_ft) >= 3:
        rho_ft, pval_ft = spearmanr(xs_ft, ys_ft)
    else:
        rho_ft, pval_ft = float("nan"), float("nan")

    for x, y, lab in list(zip(xs_ft, ys_ft, labs_ft)) + list(zip(xs_oxe, ys_oxe, labs_oxe)):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 5),
                    fontsize=7, color="0.3")

    def _sig(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "n.s."

    n_all = len(xs_all)
    n_ft = len(xs_ft)
    title_lines = [ylabel]
    title_lines.append("All: rho=%.3f p=%.4f %s (N=%d)" % (rho_all, pval_all, _sig(pval_all), n_all))
    if not np.isnan(rho_ft):
        title_lines.append("Bridge-FT only: rho=%.3f p=%.4f %s (N=%d)" % (rho_ft, pval_ft, _sig(pval_ft), n_ft))
    ax.set_title("\n".join(title_lines), fontsize=9)
    ax.set_xlabel("SimplerEnv WidowX Success Rate (%)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()

    fname = "all-cogact_%s_k%d.png" % (col, k)
    out = os.path.join(outdir, fname)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)

print("All figures saved to:", outdir)
