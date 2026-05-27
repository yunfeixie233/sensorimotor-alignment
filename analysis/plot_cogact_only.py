"""
Plot 5b: CogACT-only alignment vs WidowX performance.

Shows only CogACT-Small, CogACT-Base, CogACT-Large.
Highlights the scaling behavior: same VLM backbone trained end-to-end
with different DiT action head sizes (13M, 89M, 308M).
"""
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

csv_path = os.path.join(os.path.dirname(__file__), "..", "cknna_action_proprio_simplerenv.csv")
with open(csv_path) as f:
    rows = list(csv.DictReader(f))

COGACT_MODELS = ["CogACT-Small", "CogACT-Base", "CogACT-Large"]
rows = [r for r in rows if r["Model"] in COGACT_MODELS]
model_order = {m: i for i, m in enumerate(COGACT_MODELS)}
rows.sort(key=lambda r: model_order[r["Model"]])

SHORT = {
    "CogACT-Small": "CogACT-S\n(DiT-S 13M)",
    "CogACT-Base": "CogACT-B\n(DiT-B 89M)",
    "CogACT-Large": "CogACT-L\n(DiT-L 308M)",
}
COLORS = {
    "CogACT-Small": "tab:green",
    "CogACT-Base": "tab:orange",
    "CogACT-Large": "tab:red",
}

k = 10

panels = [
    ("CKNNA_proprio_k%d" % k, "CKNNA (VLM, Proprio)  k=%d" % k),
    ("MutualKNN_proprio_k%d" % k, "Mutual k-NN (VLM, Proprio)  k=%d" % k),
]

timestamp = time.strftime("%Y%m%d_%H%M%S")
outdir = os.path.join(os.path.dirname(__file__), "..", "runs", timestamp + "_cogact_only")
os.makedirs(outdir, exist_ok=True)

for col, ylabel in panels:
    fig, ax = plt.subplots(figsize=(6, 5))

    for r in rows:
        name = r["Model"]
        x = float(r["WidowX_avg_entire"])
        y = float(r[col])
        ax.scatter(x, y, s=120, zorder=5, marker="D",
                   edgecolors="k", linewidths=0.8, color=COLORS[name],
                   label=SHORT[name].replace("\n", " "))
        ax.annotate(SHORT[name], (x, y), textcoords="offset points",
                    xytext=(8, -8), fontsize=8, color="0.2",
                    ha="left", va="top")

    xs = np.array([float(r["WidowX_avg_entire"]) for r in rows])
    ys = np.array([float(r[col]) for r in rows])

    slope, intercept = np.polyfit(xs, ys, 1)
    xline = np.linspace(xs.min() - 1, xs.max() + 1, 100)
    ax.plot(xline, slope * xline + intercept, "--", color="gray",
            linewidth=1, zorder=3)

    ax.set_title("CogACT Scaling: %s\n(Same Prismatic VLM, different DiT action head)" % ylabel,
                 fontsize=9)
    ax.set_xlabel("SimplerEnv WidowX Success Rate (%)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)

    xpad = (xs.max() - xs.min()) * 0.3 if xs.max() != xs.min() else 2
    ax.set_xlim(xs.min() - xpad, xs.max() + xpad)

    fig.tight_layout()

    fname = "cogact-only_%s_k%d.png" % (col, k)
    out = os.path.join(outdir, fname)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)

summary_fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

k_values = [5, 10, 20]
for i, kv in enumerate(k_values):
    ax = axes[i]
    col = "CKNNA_proprio_k%d" % kv
    for r in rows:
        name = r["Model"]
        x = float(r["WidowX_avg_entire"])
        y = float(r[col])
        ax.scatter(x, y, s=100, zorder=5, marker="D",
                   edgecolors="k", linewidths=0.8, color=COLORS[name])
        ax.annotate(name.replace("CogACT-", ""), (x, y),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color="0.2")

    ax.set_title("CKNNA proprio k=%d" % kv, fontsize=9)
    ax.set_xlabel("WidowX Success Rate (%)", fontsize=9)
    if i == 0:
        ax.set_ylabel("CKNNA (VLM, Proprio)", fontsize=9)
    ax.tick_params(labelsize=8)
    xs = np.array([float(r["WidowX_avg_entire"]) for r in rows])
    xpad = (xs.max() - xs.min()) * 0.3 if xs.max() != xs.min() else 2
    ax.set_xlim(xs.min() - xpad, xs.max() + xpad)

summary_fig.suptitle("CogACT Scaling: VLM-Proprio Alignment vs Performance\n"
                     "(Prismatic VLM trained end-to-end with DiT-S/B/L; OXE-pretrained, 120 ep/task)",
                     fontsize=10)
summary_fig.tight_layout()
out = os.path.join(outdir, "cogact-only_CKNNA_proprio_all_k.png")
summary_fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(summary_fig)
print("Saved:", out)

print("All figures saved to:", outdir)
