import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

csv_path = os.path.join(os.path.dirname(__file__), "..", "cknna_action_proprio_simplerenv.csv")
with open(csv_path) as f:
    rows = list(csv.DictReader(f))

k = 10
drop_action = {"RT-1-X", "Octo-base"}

panels = [
    ("CKNNA_proprio_k%d" % k,    "CKNNA (VLM, Proprio)  k=%d" % k,    False),
    ("CKNNA_action_k%d" % k,     "CKNNA (VLM, Action)  k=%d" % k,     True),
    ("MutualKNN_proprio_k%d" % k, "Mutual k-NN (VLM, Proprio)  k=%d" % k, False),
    ("MutualKNN_action_k%d" % k,  "Mutual k-NN (VLM, Action)  k=%d" % k,  True),
]

SHORT = {
    "Qwen2.5-GR00T-Bridge":        "GR00T",
    "Qwen2.5-GR00T-Bridge-RT-1":   "GR00T-RT1",
    "Qwen3-GR00T-Bridge-RT-1":     "Q3-GR00T",
    "Qwen2.5-FAST-Bridge-RT-1":    "FAST",
    "Qwen2.5-OFT-Bridge-RT-1":     "OFT",
    "Qwen3-OFT-Bridge-RT-1":       "Q3-OFT",
    "spatialvla-sft-bridge":       "SpatialVLA",
    "pi0-lerobot-bridge":          "Pi0",
    "openvla-7b-bridge":           "OpenVLA",
    "openvla-7b-bridge-ft-200k":   "OpenVLA-FT",
    "RT-1-X":                      "RT-1-X",
    "Octo-base":                   "Octo",
    "CogACT-Small":                "CogACT-S",
    "CogACT-Base":                 "CogACT-B",
    "CogACT-Large":                "CogACT-L",
    "GR00T-N1.5-Bridge":           "GR00T-N1.5",
    "GR00T-N1.6-Bridge":           "GR00T-N1.6",
}

# two filter sets: all models, and VLA-only (no Octo / RT-1-X)
FILTER_SETS = [
    ("all",      lambda name: True),
    ("vla-only", lambda name: name not in {"RT-1-X", "Octo-base"}),
]

# Fixed output directory (not timestamped so figures overwrite in place)
outdir = os.path.join(os.path.dirname(__file__), "..", "runs", "20260224_212656")
os.makedirs(outdir, exist_ok=True)

for prefix, keep in FILTER_SETS:
    for col, ylabel, do_drop in panels:
        xs, ys, labels = [], [], []
        for r in rows:
            name = r["Model"]
            if not keep(name):
                continue
            if do_drop and name in drop_action:
                continue
            if name not in SHORT:
                continue
            # skip rows where either metric is N/A
            if r["WidowX_avg_entire"].strip() in ("", "N/A") or r[col].strip() in ("", "N/A"):
                continue
            xs.append(float(r["WidowX_avg_entire"]))
            ys.append(float(r[col]))
            labels.append(SHORT[name])

        if len(xs) < 3:
            continue

        xs = np.array(xs)
        ys = np.array(ys)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(xs, ys, s=60, zorder=5, edgecolors="k", linewidths=0.5)

        rho, pval = spearmanr(xs, ys)
        slope, intercept = np.polyfit(xs, ys, 1)
        xline = np.linspace(xs.min() - 2, xs.max() + 2, 100)
        ax.plot(xline, slope * xline + intercept, "--", color="gray", linewidth=1, zorder=3)

        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color="0.3")

        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
        ax.set_title("%s\nSpearman rho=%.3f  p=%.4f %s  (N=%d)" % (ylabel, rho, pval, sig, len(xs)),
                     fontsize=10)
        ax.set_xlabel("SimplerEnv WidowX Success Rate (%)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        fig.tight_layout()

        fname = "%s_%s_k%d.png" % (prefix, col, k)
        out = os.path.join(outdir, fname)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out)

print("All figures saved to:", outdir)
