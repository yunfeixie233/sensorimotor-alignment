"""
Plot multi-frame CKNNA results: CKNNA vs m for multiple horizons.

Usage:
  python analysis/plot_multiframe_cknna.py
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "/home/ubuntu/vla/cknna_project/record/multiframe_cknna"

# Load all available results
H_CONFIGS = [
    (1,  "#e67e22", "h=1 (next step)"),
    (75, "#2980b9", "h=75 (75-step trajectory)"),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, (h, color, label) in zip(axes, H_CONFIGS):
    json_path = os.path.join(OUT_DIR, f"cknna_vs_m_h{h}_k10.json")
    if not os.path.exists(json_path):
        print(f"Missing: {json_path}")
        continue

    with open(json_path) as f:
        data = json.load(f)

    d = data["results"][str(h)]
    ms = sorted(int(k) for k in d.keys())
    scores = [d[str(m)] for m in ms]

    ax.plot(ms, scores, "o-", color=color, linewidth=2.5, markersize=9,
            label="LingBot-VA (joint, Exp A)")
    for m, s in zip(ms, scores):
        ax.annotate(f"{s:.4f}", (m, s), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, color=color)

    # Shade region between min and max to show scale of variation
    ax.axhline(y=scores[0], color="gray", linestyle="--", alpha=0.4,
               linewidth=1, label=f"m=0 baseline ({scores[0]:.4f})")

    ax.set_xlabel("Context frames m\n(m=0: single-frame baseline)", fontsize=11)
    ax.set_ylabel(f"CKNNA (k=10)", fontsize=11)
    ax.set_title(f"CKNNA vs Context Frames — {label}", fontsize=12)
    ax.set_xticks(ms)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    # Report % change from m=0
    print(f"\nh={h}:")
    for m, s in zip(ms, scores):
        pct = (s - scores[0]) / scores[0] * 100
        print(f"  m={m:2d}: {s:.6f}  ({pct:+.1f}% vs m=0)")

fig.suptitle("Multi-Frame CKNNA: LingBot-VA Exp A\n"
             "DROID dataset, N=4890, joint temporal attention", fontsize=13)
fig.tight_layout()

out_png = os.path.join(OUT_DIR, "cknna_vs_m_combined.png")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nCombined figure → {out_png}")
