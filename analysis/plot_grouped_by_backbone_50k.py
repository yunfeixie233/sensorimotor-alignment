"""
Scatter: alignment metric vs WidowX SR for N=50K CKNNA evaluation.

Two versions of each panel:
  (1) N=50K only (6 StarVLA models completed so far)
  (2) N=5K vs N=50K comparison (arrows from 5K to 50K values)

Reads from:
  - cknna_action_proprio_simplerenv.csv      (N=5K, full set)
  - cknna_action_proprio_simplerenv_50k.csv  (N=50K, StarVLA only)
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

csv_dir = os.path.join(os.path.dirname(__file__), "..")
csv_5k = os.path.join(csv_dir, "cknna_action_proprio_simplerenv.csv")
csv_50k = os.path.join(csv_dir, "cknna_action_proprio_simplerenv_50k.csv")

with open(csv_5k) as f:
    rows_5k = {r["Model"]: r for r in csv.DictReader(f)}
with open(csv_50k) as f:
    rows_50k = {r["Model"]: r for r in csv.DictReader(f)}

GROUPS = [
    {
        "label": "Qwen2.5-VL-3B",
        "color": "#1f77b4",
        "marker": "o",
        "size": 100,
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
        "size": 110,
        "models": {
            "Qwen3-GR00T-Bridge-RT-1",
            "Qwen3-OFT-Bridge-RT-1",
        },
    },
]

DISPLAY_NAME = {
    "Qwen2.5-GR00T-Bridge": "GR00T-Bridge",
    "Qwen2.5-GR00T-Bridge-RT-1": "GR00T-Bridge-RT-1",
    "Qwen2.5-FAST-Bridge-RT-1": "FAST-Bridge-RT-1",
    "Qwen2.5-OFT-Bridge-RT-1": "OFT-Bridge-RT-1",
    "Qwen3-GR00T-Bridge-RT-1": "Qwen3-GR00T-RT-1",
    "Qwen3-OFT-Bridge-RT-1": "Qwen3-OFT-RT-1",
}

outdir = os.path.join(csv_dir, "runs", "20260227_50k")
os.makedirs(outdir, exist_ok=True)

k = 10


def _sig(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# --- Figure 1: N=50K standalone (proprio only, since action not yet computed) ---
for col, ylabel in [
    ("CKNNA_proprio_k%d" % k, "CKNNA (VLM, Proprio)  k=%d  [N=50K]" % k),
    ("MutualKNN_proprio_k%d" % k, "Mutual k-NN (VLM, Proprio)  k=%d  [N=50K]" % k),
]:
    fig, ax = plt.subplots(figsize=(10, 7))
    xs_all, ys_all = [], []

    for grp in GROUPS:
        xs, ys, labs = [], [], []
        for name in sorted(grp["models"]):
            r = rows_50k.get(name)
            if r is None or r[col] == "N/A":
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
            ax.annotate(lab, (x, y), fontsize=10, fontweight="bold",
                        color="0.25", xytext=(6, 6),
                        textcoords="offset points", zorder=10)

        xs_all.extend(xs)
        ys_all.extend(ys)

    xs_all = np.array(xs_all)
    ys_all = np.array(ys_all)

    rho, pval = spearmanr(xs_all, ys_all)
    slope, intercept = np.polyfit(xs_all, ys_all, 1)
    xline = np.linspace(xs_all.min() - 2, xs_all.max() + 2, 100)
    ax.plot(xline, slope * xline + intercept, "--", color="gray",
            linewidth=1, zorder=3)

    title = "%s\nSpearman rho=%.3f  p=%.4f %s  (N_models=%d, N_samples=50K)" % (
        ylabel, rho, pval, _sig(pval), len(xs_all))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("SimplerEnv WidowX Success Rate (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11, loc="best", framealpha=0.9, prop={"weight": "bold"})
    fig.tight_layout()

    fname = "50k_%s_k%d.png" % (col, k)
    out = os.path.join(outdir, fname)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


# --- Figure 2: N=5K vs N=50K comparison (arrows showing the shift) ---
for col, ylabel_base in [
    ("CKNNA_proprio_k%d" % k, "CKNNA (VLM, Proprio) k=%d" % k),
    ("MutualKNN_proprio_k%d" % k, "Mutual k-NN (VLM, Proprio) k=%d" % k),
]:
    fig, ax = plt.subplots(figsize=(11, 7))

    for grp in GROUPS:
        for name in sorted(grp["models"]):
            r5 = rows_5k.get(name)
            r50 = rows_50k.get(name)
            if r5 is None or r50 is None:
                continue
            if r5[col] == "N/A" or r50[col] == "N/A":
                continue
            x = float(r5["WidowX_avg_entire"])
            y5 = float(r5[col])
            y50 = float(r50[col])

            ax.scatter(x, y5, s=grp["size"] * 0.6, zorder=4,
                       marker=grp["marker"], edgecolors="k", linewidths=0.5,
                       color=grp["color"], alpha=0.4)
            ax.scatter(x, y50, s=grp["size"], zorder=5,
                       marker=grp["marker"], edgecolors="k", linewidths=0.6,
                       color=grp["color"])
            ax.annotate("", xy=(x, y50), xytext=(x, y5),
                        arrowprops=dict(arrowstyle="->", color=grp["color"],
                                        lw=1.5, alpha=0.7),
                        zorder=3)
            ax.annotate(DISPLAY_NAME[name], (x, y50), fontsize=9,
                        fontweight="bold", color="0.25",
                        xytext=(6, 6), textcoords="offset points", zorder=10)

    ax.scatter([], [], s=60, color="gray", alpha=0.4, label="N=5K")
    ax.scatter([], [], s=100, color="gray", label="N=50K")
    for grp in GROUPS:
        ax.scatter([], [], s=grp["size"], marker=grp["marker"],
                   color=grp["color"], edgecolors="k", linewidths=0.6,
                   label=grp["label"])

    title = "%s: N=5K -> N=50K shift\n(arrows show direction; x-axis = WidowX SR, same for both)" % ylabel_base
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("SimplerEnv WidowX Success Rate (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel_base, fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=10, loc="best", framealpha=0.9, prop={"weight": "bold"})
    fig.tight_layout()

    fname = "compare_5k_vs_50k_%s_k%d.png" % (col, k)
    out = os.path.join(outdir, fname)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


# --- Figure 3: Ranking comparison bar chart ---
fig, ax = plt.subplots(figsize=(12, 5))

col = "CKNNA_proprio_k%d" % k
models_sorted = sorted(rows_50k.keys(), key=lambda n: float(rows_50k[n][col]), reverse=True)

x_pos = np.arange(len(models_sorted))
bar_w = 0.35
vals_5k = [float(rows_5k[m][col]) for m in models_sorted]
vals_50k = [float(rows_50k[m][col]) for m in models_sorted]

bars5 = ax.bar(x_pos - bar_w / 2, vals_5k, bar_w, label="N=5K", color="#aec7e8", edgecolor="k", linewidth=0.5)
bars50 = ax.bar(x_pos + bar_w / 2, vals_50k, bar_w, label="N=50K", color="#1f77b4", edgecolor="k", linewidth=0.5)

ax.set_xticks(x_pos)
ax.set_xticklabels([DISPLAY_NAME[m] for m in models_sorted], rotation=30, ha="right", fontsize=11, fontweight="bold")
ax.set_ylabel("CKNNA (VLM, Proprio) k=%d" % k, fontsize=13, fontweight="bold")
ax.set_title("CKNNA_proprio k=%d: N=5K vs N=50K (ranking preserved)" % k, fontsize=14, fontweight="bold")
ax.legend(fontsize=12, prop={"weight": "bold"})
ax.tick_params(labelsize=11)

for b5, b50 in zip(bars5, bars50):
    ratio = b50.get_height() / b5.get_height()
    ax.text(b50.get_x() + b50.get_width() / 2, b50.get_height() + 0.002,
            "%.1fx" % ratio, ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1f77b4")

fig.tight_layout()
out = os.path.join(outdir, "ranking_5k_vs_50k_CKNNA_proprio_k%d.png" % k)
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", out)

print("\nAll figures saved to:", outdir)
