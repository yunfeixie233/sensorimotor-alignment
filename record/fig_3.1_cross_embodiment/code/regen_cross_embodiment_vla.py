
# Resolve paths relative to this script's location so the release works from any clone path.
import os as _os
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_DATA = _HERE.parent / "data"
_FIG_DIR = _HERE.parent
# Add record/shared/ to sys.path for google_palette / scatter_labels helpers.
_sys.path.insert(0, str(_HERE.parent.parent / "shared"))
#!/usr/bin/env python3
"""§3.2 figure: 1×2 scatter showing the VLA \\PA result holds across two
reference embodiments. Same n=12 finetuned VLAs, same SimplerEnv-WidowX
target, but \\PA is computed on different offline reference datasets:

    Left  panel: DROID    reference (Franka Panda, cross-embodiment),  h=7,
                 Pearson r ≈ +0.717.
    Right panel: BridgeDataV2 reference (WidowX,        in-embodiment),    h=15,
                 Pearson r ≈ +0.700.

Both panels share the y axis. Color is the Google-red VLA palette in both
panels (single cohort), with a small reference-name tag in the bottom-right.

Outputs:
    figures/cross_embodiment_vla.pdf  (vector, used in paper)
    figures/cross_embodiment_vla.png  (preview)
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import pearsonr

from google_palette import GOOGLE_CMAP, GOOGLE_BLUE, GOOGLE_RED
from scatter_labels import add_labels, add_fit_with_ci, stats_box_text, format_r

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 13,
    "mathtext.fontset": "cm",
    "mathtext.default": "regular",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FIG_DIR = str(_FIG_DIR)

VLA_DROID_CKNNA  = str(_DATA / "cknna_vla_droid_complete.csv")
VLA_BRIDGE_CKNNA = str(_DATA / "cknna_vla_bridgev2_complete.csv")
VLA_SR_CSV       = str(_DATA / "widowx_sr.csv")

FEAT = "imgtext"  # LLM-out vision-text


def load_sr(path):
    # Drop the VLM4VLA-style finetuned rows (vlm4vla-qwen25vl3b-...): those
    # belong to the §4 frozen-vs-unfrozen ablation, not the headline n=12
    # finetuned-VLA cohort plotted here.
    sr = {}
    for r in csv.DictReader(open(path)):
        if r.get("excluded", "").strip().lower() == "true":
            continue
        if r["model_key"].startswith("vlm4vla"):
            continue
        sr[r["model_key"]] = float(r["success_rate"])
    return sr


def load_pairs(cknna_csv, sr_dict, feat, horizon):
    xs, ys, ks = [], [], []
    for r in csv.DictReader(open(cknna_csv)):
        if r["feats_A_variant"] != feat: continue
        if int(float(r["horizon"])) != int(horizon): continue
        if r["model"] not in sr_dict: continue
        xs.append(float(r["cknna_k10"]))
        ys.append(sr_dict[r["model"]])
        ks.append(r["model"])
    return np.array(xs), np.array(ys), ks


def panel(ax, x, y, cmap, title, x_lim, y_lim, keys=None):
    norm = (x - x.min()) / (x.max() - x.min()) if len(x) > 1 and x.max() > x.min() else np.full_like(x, 0.5)
    add_fit_with_ci(ax, x, y, color=cmap(0.85))
    ax.scatter(x, y, s=180, c=cmap(0.30 + 0.65 * norm),
               edgecolors="black", linewidths=0.7, zorder=3)
    r, p2 = pearsonr(x, y)
    p1 = p2 / 2 if r > 0 else 1 - p2 / 2
    stats = stats_box_text(r, p1)
    # Per-figure font scaling: 1x2 panel renders at ~0.29x source
    # width (12.5in vs 0.65 textwidth). Sizes bumped slightly so axis
    # labels match Fig 3's rendered size at ~0.327x.
    ax.text(0.04, 0.96, stats, transform=ax.transAxes, fontsize=23,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#888888", alpha=0.92))
    # Reference dataset is now in the panel title (above the axes)
    # rather than as a bottom-right text box, per 2026-05-06 review.
    ax.set_title(title, fontsize=24, fontweight="bold", pad=10)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_xlabel(r"Sensorimotor Alignment ($\mathrm{S}^2$)", fontsize=23)
    ax.tick_params(axis="both", labelsize=18)
    # Standardise S^2 x-tick labels to 3 decimal places across all
    # Pearson scatters (Figs 3, 4-left, 5a, 8, 9).
    from matplotlib.ticker import FormatStrFormatter
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.grid(True, alpha=0.25)
    if keys is not None:
        add_labels(ax, x, y, keys, fontsize=15)


def main():
    sr = load_sr(VLA_SR_CSV)
    dx, dy, dk = load_pairs(VLA_DROID_CKNNA,  sr, FEAT, 7)
    bx, by, bk = load_pairs(VLA_BRIDGE_CKNNA, sr, FEAT, 15)
    print(f"DROID    n={len(dx)} r={format_r(pearsonr(dx, dy)[0])}")
    print(f"BridgeDataV2 n={len(bx)} r={format_r(pearsonr(bx, by)[0])}")

    # Per-panel x range (different absolute SiMA scales). Shared y range.
    # Generous padding (15% on x, 18% on y) gives the n=12 labels room
    # to fan out from clustered points (CogACT-S/B/L especially) without
    # leader lines having to run all the way back to the data band.
    def x_pad(arr, frac=0.15):
        lo, hi = float(arr.min()), float(arr.max())
        pad = frac * max(hi - lo, 1e-9)
        return (lo - pad, hi + pad)

    droid_x_lim  = x_pad(dx)
    bridge_x_lim = x_pad(bx)
    y_all = np.concatenate([dy, by])
    y_pad = 0.18 * (y_all.max() - y_all.min())
    y_lim = (y_all.min() - y_pad, y_all.max() + y_pad)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True)
    panel(axL, dx, dy, GOOGLE_CMAP["blue"],
          r"$\mathrm{S}^2$ computed on DROID",
          droid_x_lim, y_lim, keys=dk)
    panel(axR, bx, by, GOOGLE_CMAP["red"],
          r"$\mathrm{S}^2$ computed on BridgeDataV2",
          bridge_x_lim, y_lim, keys=bk)
    axL.set_ylabel("SimplerEnv success rate (\\%)", fontsize=23)

    fig.tight_layout()
    out_pdf = os.path.join(FIG_DIR, "cross_embodiment_vla.pdf")
    out_png = os.path.join(FIG_DIR, "cross_embodiment_vla.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
