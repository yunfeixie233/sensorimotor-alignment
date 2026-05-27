
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
"""Regenerate the WAM Sensorimotor Similarity vs.\ RoboTwin success-rate
scatter that contrasts the ALOHA and DROID reference datasets, in the
same minimalist style as figures/rawvlm_cknna_best.pdf:
    - Single combined vector PDF (figures/va_sr_aloha_vs_droid.pdf).
    - Two subplots side-by-side, no per-panel matplotlib title, no linear
      fit, no Spearman.
    - Each panel has a Pearson stats box (r, p one-tailed, n) at the
      top-left and a small bold "(a)/(b) <reference> reference"
      identifier at the top-right.
    - Color is per-reference: ALOHA panel uses a blue palette, DROID
      uses red. Within each panel the color intensity rank-deepens
      with the Sensorimotor Similarity value.
    - Shared y axis ("RoboTwin Success Rate (%)"). Per-panel x ranges
      because the absolute \PA values differ ~10x between ALOHA and
      DROID; a global x range would crowd each panel into a thin strip.
    - Single bordered legend at the bottom with two entries:
      "ALOHA" (blue) and "DROID" (red).
    - All chart text in the Helvetica family. PDF embeds Type 42.

Three World-Action Models, the same three used in
Section~\\ref{sec:finding-va-sr}: LingBot-VA, Motus, and Vidar.
Feature: V-B7 (video transformer block 7 / 30) at horizon h=15.

Inputs (relative to this script's `code/` dir, resolved at runtime):
    ALOHA \\PA + RoboTwin SR: ../data/cknna_wam_aloha_complete.csv
    DROID \\PA + RoboTwin SR: ../data/cknna_wam_droid_complete.csv

Usage:
    cd /home/ubuntu/vla/vla_idea/writing/cknna_vla/figures
    /opt/miniconda/bin/python regen_va_sr_aloha_vs_droid.py
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from google_palette import GOOGLE_CMAP
from scatter_labels import add_labels, add_fit_with_ci, stats_box_text, format_r

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"
]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

FIG_DIR = str(_FIG_DIR)

ALOHA_CSV = str(_DATA / "cknna_wam_aloha_complete.csv")
DROID_CSV = str(_DATA / "cknna_wam_droid_complete.csv")

FEATURE = "V-B7"
HORIZON = 15

# The three World-Action Models that appear in the main-text caption. The
# same model can have a different row name in each CSV because the master
# tables encode the inference layout used for that reference (e.g.,
# ALOHA-native vs.\ 3-cam DROID), so we map by friendly name.
ALOHA_MODELS = {
    "LingBot-VA": "lingbot-va-posttrain-robotwin",
    "Motus":      "motus-robotwin2-aloha-native",
    "Vidar":      "vidar_aloha_native",
}
DROID_MODELS = {
    "LingBot-VA": "lingbot-va-posttrain-robotwin",
    "Motus":      "motus-robotwin2-lingbot",
    "Vidar":      "vidar_3cam_modeB",
}


def load_pairs(csv_path, model_map, feature, horizon):
    """Return ([cknna], [robotwin_sr_avg], [keys])."""
    by_row = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["feature_source"] != feature:
                continue
            if int(r["horizon"]) != int(horizon):
                continue
            by_row[r["model"]] = (
                float(r["cknna_k10"]), float(r["robotwin_sr_avg"]))
    xs, ys, ks = [], [], []
    for friendly, row_name in model_map.items():
        if row_name not in by_row:
            raise KeyError(f"missing row {row_name!r} in {csv_path}")
        cknna, sr = by_row[row_name]
        xs.append(cknna)
        ys.append(sr)
        ks.append(row_name)
    return np.array(xs), np.array(ys), ks


def panel(ax, x, y, cmap, title, x_lim, y_lim, keys=None,
          stats_loc="top-left"):
    if len(x) > 1 and x.max() > x.min():
        norm = (x - x.min()) / (x.max() - x.min())
    else:
        norm = np.full_like(x, 0.5)
    facecolors = cmap(0.35 + 0.6 * norm)

    add_fit_with_ci(ax, x, y, color=cmap(0.85))
    # n=3 sparse panels can comfortably afford bigger dots (s=300 vs
    # the 200 default that came from a denser-cohort design). Heavier
    # edges (1.0 vs 0.8) keep dots distinct against the regression line.
    ax.scatter(x, y, s=300, c=facecolors,
               edgecolors="black", linewidths=1.0, zorder=3)

    r, p_two = pearsonr(x, y)
    p_one = p_two / 2 if r > 0 else 1 - p_two / 2
    stats = stats_box_text(r, p_one)
    # Per-figure font scaling: 1x2 panel renders at ~0.29x source
    # width (12.5in vs 0.65 textwidth). Sizes bumped slightly so axis
    # labels match Fig 3's rendered size at ~0.327x.
    # `stats_loc` is per-panel because the data layout differs:
    # ALOHA panel has its dots along a SW->NE diagonal (top-left
    # corner free); DROID panel has LingBot-VA at top-left, so the
    # stats box has to move to bottom-right.
    POS = {"top-left":     (0.04, 0.96, "top",    "left"),
           "bottom-right": (0.96, 0.04, "bottom", "right")}
    sx, sy, sva, sha = POS[stats_loc]
    # Same stats-box size on both panels (per 2026-05-06 review).
    # 19 is the largest value that fits in the DROID panel without
    # the bottom-right box brushing Vidar's right-anchored label;
    # we use the same size on the ALOHA panel for visual symmetry.
    ax.text(sx, sy, stats, transform=ax.transAxes, fontsize=19,
            va=sva, ha=sha,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#888888", alpha=0.92))

    # Reference dataset is now in the panel title (Fig 6 style),
    # replacing the previous bottom-right "(a) ALOHA reference"
    # text box and the figure-level ALOHA/DROID legend.
    ax.set_title(title, fontsize=24, fontweight="bold", pad=10)

    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_xlabel(r"Sensorimotor Alignment ($\mathrm{S}^2$)", fontsize=23)
    ax.tick_params(axis="both", labelsize=18)
    # Standardise S^2 x-tick labels to 3 decimal places across all
    # Pearson scatters (Figs 3, 4-left, 5a, 6, 9).
    from matplotlib.ticker import FormatStrFormatter
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.grid(True, alpha=0.25)

    if keys is not None:
        # Bigger model labels (22 vs 15): with only n=3 dots per panel
        # the labels have plenty of room and the previous 15pt was
        # disproportionately small next to 23pt axis labels.
        add_labels(ax, x, y, keys, fontsize=22)


def main():
    for p in [ALOHA_CSV, DROID_CSV]:
        if not os.path.isfile(p):
            print(f"ERROR: missing input CSV: {p}", file=sys.stderr)
            sys.exit(1)

    ax_x, ax_y, ax_k = load_pairs(ALOHA_CSV, ALOHA_MODELS, FEATURE, HORIZON)
    dx_x, dx_y, dx_k = load_pairs(DROID_CSV, DROID_MODELS, FEATURE, HORIZON)

    # Per-reference x ranges (different absolute scales), shared y range
    # because both panels show the same RoboTwin success rate.
    # Padding is asymmetric on the y axis to keep the rightmost data point
    # clear of the top-left stats box and the bottom-right panel-ID tag:
    #   x: 0.20 both sides
    #   y: 0.15 below, 0.40 above
    def x_range_with_pad(arr, frac=0.20):
        lo, hi = float(arr.min()), float(arr.max())
        pad = frac * max(hi - lo, 1e-9)
        return (lo - pad, hi + pad)

    def y_range_with_pad(arr, bottom_frac=0.15, top_frac=0.40):
        lo, hi = float(arr.min()), float(arr.max())
        rng = max(hi - lo, 1e-9)
        return (lo - bottom_frac * rng, hi + top_frac * rng)

    aloha_x_lim = x_range_with_pad(ax_x)
    droid_x_lim = x_range_with_pad(dx_x)
    y_all = np.concatenate([ax_y, dx_y])
    y_lim = y_range_with_pad(y_all)

    print(f"  ALOHA: x range {aloha_x_lim[0]:.4f} .. {aloha_x_lim[1]:.4f}")
    print(f"  DROID: x range {droid_x_lim[0]:.4f} .. {droid_x_lim[1]:.4f}")
    print(f"  shared y range: {y_lim[0]:.2f} .. {y_lim[1]:.2f}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True)
    panel(axL, ax_x, ax_y, GOOGLE_CMAP["blue"],
          r"$\mathrm{S}^2$ computed on ALOHA",
          aloha_x_lim, y_lim, keys=ax_k, stats_loc="top-left")
    panel(axR, dx_x, dx_y, GOOGLE_CMAP["red"],
          r"$\mathrm{S}^2$ computed on DROID",
          droid_x_lim, y_lim, keys=dx_k, stats_loc="bottom-right")
    axL.set_ylabel("RoboTwin success rate (\\%)", fontsize=23)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "va_sr_aloha_vs_droid.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print("  wrote", os.path.basename(out))
    r_a, _ = pearsonr(ax_x, ax_y)
    r_d, _ = pearsonr(dx_x, dx_y)
    print(f"    (a) ALOHA n={len(ax_x)} r={format_r(r_a)}")
    print(f"    (b) DROID n={len(dx_x)} r={format_r(r_d)}")


if __name__ == "__main__":
    main()
