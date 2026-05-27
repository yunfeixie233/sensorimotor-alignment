"""
Scatter: alignment metric vs WidowX SR, models grouped by VLM backbone.

Produces TWO figure sets:
  (A) "all"     -- every model except OpenVLA
  (B) "vla-only" -- excludes OpenVLA + pre-VLM generalists (RT-1-X, Octo)

Groups (by backbone):
  1. Qwen2.5-VL-3B         (StarVLA Qwen 2.5 variants)
  2. Qwen3-VL-4B           (StarVLA Qwen 3 variants)
  3. Prismatic+Llama2-7B   (CogACT family)
  4. PaLiGemma             (SpatialVLA, Pi0)
  5. NVEagle               (GR00T N1.5, N1.6)
  6. Non-VLM               (RT-1-X, Octo) -- only in figure set A

Uses adjustText for automatic label repositioning to avoid overlaps.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from scipy.stats import spearmanr


csv_path = os.path.join(os.path.dirname(__file__), "..", "cknna_action_proprio_simplerenv.csv")
with open(csv_path) as f:
    rows_all = list(csv.DictReader(f))

EXCLUDE_ALWAYS = set()
PRE_VLM = {"RT-1-X", "Octo-base"}

GROUPS = [
    {
        "label": "Qwen2.5-VL-3B",
        "color": "#1f77b4",
        "marker": "o",
        "size": 80,
        "models": {
            "Qwen2.5-GR00T-Bridge",
            "Qwen2.5-GR00T-Bridge-RT-1",
            "Qwen2.5-FAST-Bridge-RT-1",
            "Qwen2.5-OFT-Bridge-RT-1",
        },
    },
    {
        "label": "Qwen3-VL-4B",
        "color": "#17becf",
        "marker": "^",
        "size": 90,
        "models": {
            "Qwen3-GR00T-Bridge-RT-1",
            "Qwen3-OFT-Bridge-RT-1",
        },
    },
    {
        "label": "Prismatic+Llama2-7B (OpenVLA / CogACT)",
        "color": "#ff7f0e",
        "marker": "D",
        "size": 85,
        "models": {
            "openvla-7b-bridge",
            "openvla-7b-bridge-ft-200k",
            "CogACT-Small",
            "CogACT-Base",
            "CogACT-Large",
        },
    },
    {
        "label": "PaLiGemma",
        "color": "#2ca02c",
        "marker": "v",
        "size": 95,
        "models": {
            "spatialvla-sft-bridge",
            "pi0-lerobot-bridge",
        },
    },
    {
        "label": "NVEagle",
        "color": "#9467bd",
        "marker": "*",
        "size": 140,
        "models": {
            "GR00T-N1.5-Bridge",
            "GR00T-N1.6-Bridge",
        },
    },
    {
        "label": "Non-VLM",
        "color": "#d62728",
        "marker": "s",
        "size": 80,
        "models": PRE_VLM,
    },
]

DISPLAY_NAME = {
    "Qwen2.5-GR00T-Bridge": "Qwen2.5-GR00T-Bridge",
    "Qwen2.5-GR00T-Bridge-RT-1": "Qwen2.5-GR00T-Bridge-RT-1",
    "Qwen2.5-FAST-Bridge-RT-1": "Qwen2.5-FAST-Bridge-RT-1",
    "Qwen2.5-OFT-Bridge-RT-1": "Qwen2.5-OFT-Bridge-RT-1",
    "Qwen3-GR00T-Bridge-RT-1": "Qwen3-GR00T-Bridge-RT-1",
    "Qwen3-OFT-Bridge-RT-1": "Qwen3-OFT-Bridge-RT-1",
    "spatialvla-sft-bridge": "SpatialVLA",
    "pi0-lerobot-bridge": "Pi0",
    "openvla-7b-bridge": "OpenVLA-Base",
    "openvla-7b-bridge-ft-200k": "OpenVLA-FT-200k",
    "CogACT-Small": "CogACT-Small",
    "CogACT-Base": "CogACT-Base",
    "CogACT-Large": "CogACT-Large",
    "GR00T-N1.5-Bridge": "GR00T-N1.5",
    "GR00T-N1.6-Bridge": "GR00T-N1.6",
    "RT-1-X": "RT-1-X",
    "Octo-base": "Octo",
}

LABEL_FONTSIZE = 10
LABEL_FONTWEIGHT = "bold"

k = 10

panels = [
    ("CKNNA_proprio_k%d" % k, "CKNNA (VLM, Proprio)  k=%d" % k),
    ("CKNNA_action_k%d" % k, "CKNNA (VLM, Action)  k=%d" % k),
    ("MutualKNN_proprio_k%d" % k, "Mutual k-NN (VLM, Proprio)  k=%d" % k),
    ("MutualKNN_action_k%d" % k, "Mutual k-NN (VLM, Action)  k=%d" % k),
]

outdir = os.path.join(os.path.dirname(__file__), "..", "runs", "20260224_212656")
os.makedirs(outdir, exist_ok=True)


def _sig(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


FIGURE_SETS = [
    ("all", EXCLUDE_ALWAYS),
    ("vla-only", EXCLUDE_ALWAYS | PRE_VLM),
]

for fig_tag, exclude_set in FIGURE_SETS:
    rows = [r for r in rows_all if r["Model"] not in exclude_set]

    for col, ylabel in panels:
        fig, ax = plt.subplots(figsize=(10, 7))

        xs_all, ys_all = [], []
        texts = []

        for grp in GROUPS:
            if grp["models"] <= exclude_set:
                continue

            xs, ys, labs = [], [], []
            for r in rows:
                name = r["Model"]
                if name not in grp["models"]:
                    continue
                if r[col] == "N/A" or r["WidowX_avg_entire"] == "N/A":
                    continue
                x = float(r["WidowX_avg_entire"])
                y = float(r[col])
                xs.append(x)
                ys.append(y)
                labs.append(DISPLAY_NAME[name])

            if not xs:
                continue

            ax.scatter(xs, ys, s=grp["size"], zorder=5,
                       marker=grp["marker"], edgecolors="k", linewidths=0.6,
                       color=grp["color"], label=grp["label"])

            for x, y, lab in zip(xs, ys, labs):
                t = ax.text(x, y, lab,
                            fontsize=LABEL_FONTSIZE,
                            fontweight=LABEL_FONTWEIGHT,
                            color="0.25", zorder=10)
                texts.append(t)

            xs_all.extend(xs)
            ys_all.extend(ys)

        xs_all = np.array(xs_all)
        ys_all = np.array(ys_all)

        rho, pval = spearmanr(xs_all, ys_all)
        slope, intercept = np.polyfit(xs_all, ys_all, 1)
        xline = np.linspace(xs_all.min() - 2, xs_all.max() + 2, 100)
        ax.plot(xline, slope * xline + intercept, "--", color="gray",
                linewidth=1, zorder=3)

        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="0.4", lw=0.8),
                    expand=(1.2, 1.3),
                    force_text=(0.4, 0.4),
                    force_points=(0.8, 0.8),
                    min_arrow_len=8,
                    ensure_inside_axes=True)

        title_lines = [ylabel]
        title_lines.append("Spearman rho=%.3f  p=%.4f %s  (N=%d)"
                           % (rho, pval, _sig(pval), len(xs_all)))
        ax.set_title("\n".join(title_lines), fontsize=13, fontweight="bold")
        ax.set_xlabel("SimplerEnv WidowX Success Rate (%)",
                       fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=11, loc="best", framealpha=0.9,
                  prop={"weight": "bold"})
        fig.tight_layout()

        fname = "%s_%s_k%d.png" % (fig_tag, col, k)
        out = os.path.join(outdir, fname)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out)

print("\nAll figures saved to:", outdir)
