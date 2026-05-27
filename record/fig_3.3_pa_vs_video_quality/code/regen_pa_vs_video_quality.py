
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
"""Regenerate the four panels that contrast \\PA's correlation with RoboTwin
success rate against three standard video-quality metrics that fail to
predict it. Each panel is its own PDF so the paper-side wrapper can lay
them out via \\subcaptionbox (LaTeX-supplied (a)(b)(c)(d) labels).

Style mimics iREPA-SSM Figure 1 (arXiv 2512.10794): each axis label has
a directional arrow indicating "direction of better" --
    → at the right of the x-axis label when higher = better,
    ← at the left of the x-axis label when lower = better,
    ↑ at the left of the y-axis label since success rate is higher = better.

Three World-Action Models, the same three used in
Section~\\ref{sec:finding-va-sr}: LingBot-VA, Motus, Vidar.

Outputs (one PDF per panel):
    figures/pa_vs_video_quality_a_sima.pdf
    figures/pa_vs_video_quality_b_ssim.pdf
    figures/pa_vs_video_quality_c_psnr.pdf
    figures/pa_vs_video_quality_d_fid.pdf
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
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

FIG_DIR = str(_FIG_DIR)

ALOHA_CSV = str(_DATA / "cknna_wam_aloha_complete.csv")
FEATURE   = "V-B7"
HORIZON   = 15

ALOHA_MODELS = {
    "LingBot-VA": "lingbot-va-posttrain-robotwin",
    "Motus":      "motus-robotwin2-aloha-native",
    "Vidar":      "vidar_aloha_native",
}

MODELS = ["LingBot-VA", "Motus", "Vidar"]
SR     = {"LingBot-VA": 91.4,  "Motus": 85.2,   "Vidar": 57.8}
SSIM   = {"LingBot-VA": 0.806, "Motus": 0.886,  "Vidar": 0.791}
PSNR   = {"LingBot-VA": 17.5,  "Motus": 21.5,   "Vidar": 17.3}
FID    = {"LingBot-VA": 36.0,  "Motus": 16.1,   "Vidar": 12.4}


def load_aloha_pa(csv_path, model_map, feature, horizon):
    by_row = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["feature_source"] != feature: continue
            if int(r["horizon"]) != int(horizon): continue
            by_row[r["model"]] = float(r["cknna_k10"])
    return {friendly: by_row[row] for friendly, row in model_map.items()}


def x_pad(arr, frac=0.20):
    lo, hi = float(arr.min()), float(arr.max())
    pad = frac * max(hi - lo, 1e-9)
    return (lo - pad, hi + pad)


def y_pad(arr, bottom=0.18, top=0.18):
    lo, hi = float(arr.min()), float(arr.max())
    rng = max(hi - lo, 1e-9)
    return (lo - bottom * rng, hi + top * rng)


def panel(out_path, x, y, cmap, x_label_text, x_higher_better, x_lim, y_lim, keys=None):
    """One panel; iREPA-style arrow on each axis indicating direction of better.

    x_higher_better: True  -> arrow at right (e.g., "SSIM →")
                     False -> arrow at left  (e.g., "← FID")

    NB: no CI band on any panel (paper-style decision 2026-05-06):
    just the dashed regression fit line. The band was distracting
    readers from the main slope-sign + magnitude message.
    """
    # figsize_h = 5.2 (was 4.2): the rotated y-label "RoboTwin success
    # rate (%)" at fontsize=22 needs ~310 pt of vertical extent, which
    # the previous 4.2in (302 pt) figure couldn't accommodate -- the
    # closing "%)" was getting clipped at the panel edge.
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 5.2))

    norm = (x - x.min()) / (x.max() - x.min()) if len(x) > 1 and x.max() > x.min() else np.full_like(x, 0.5)
    facecolors = cmap(0.30 + 0.65 * norm)
    add_fit_with_ci(ax, x, y, color=cmap(0.85))
    ax.scatter(x, y, s=260, c=facecolors, edgecolors="black", linewidths=0.9, zorder=3)

    r, p_two = pearsonr(x, y)
    p_one = p_two / 2 if r > 0 else 1 - p_two / 2
    # Place the stats box in the bottom-right corner; the 3 data points sit
    # along a SW->NE diagonal in every panel, leaving the SE corner clear.
    stats = stats_box_text(r, p_one)
    # Per-figure font scaling: each Fig-5 sub-panel renders at ~0.26x
    # source width (5in vs 0.24 textwidth). Sizes bumped slightly so
    # axis labels match Fig 3's rendered size at ~0.327x.
    # Stats box fontsize=16 (vs xlabel=24): the 4-inch sub-panels
    # cannot fit a wider stats box in any corner without overlapping
    # the data points (LingBot-VA at top-left, Motus at top-right,
    # Vidar at bottom-left -- only the bottom-right corner is free,
    # and a 24pt box spans wide enough to hit Vidar's label even
    # there). 16pt is the largest size that clears all dots in all
    # four sub-panels.
    ax.text(0.96, 0.04, stats, transform=ax.transAxes, fontsize=16,
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.40", facecolor="white",
                      edgecolor="#888888", alpha=0.92))

    # Arrow convention: direction of "better" on x-axis only.
    if x_higher_better:
        x_label = f"{x_label_text}  $\\rightarrow$"
    else:
        x_label = f"$\\leftarrow$  {x_label_text}"
    y_label = "RoboTwin success rate (\\%)"

    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_xlabel(x_label, fontsize=24)
    # y-label at fontsize=22 (vs xlabel=24) so the rotated 25-char
    # string "RoboTwin success rate (%)" fits within the figure
    # height without clipping the closing "(%)" -- the panel width is
    # generous enough that an asymmetric x/y axis label size is the
    # cleanest fix.
    ax.set_ylabel(y_label, fontsize=22)
    ax.tick_params(axis="both", labelsize=19)
    # Standardise S^2 x-tick labels to 3 decimal places (matches Figs
    # 3, 4-left, 6, 8, 9). Only the SiMA panel uses S^2 on x; the
    # SSIM/PSNR/FID panels use their natural metric scale.
    if r"\mathrm{S}^2" in x_label_text:
        from matplotlib.ticker import FormatStrFormatter
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.grid(True, alpha=0.25)

    if keys is not None:
        add_labels(ax, x, y, keys, fontsize=16)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.basename(out_path)}  r={format_r(r)}  p={p_one:.3f}")


def main():
    if not os.path.isfile(ALOHA_CSV):
        print(f"ERROR: missing input CSV: {ALOHA_CSV}", file=sys.stderr); sys.exit(1)

    pa = load_aloha_pa(ALOHA_CSV, ALOHA_MODELS, FEATURE, HORIZON)
    y    = np.array([SR[m]    for m in MODELS])
    x_pa = np.array([pa[m]    for m in MODELS])
    x_ss = np.array([SSIM[m]  for m in MODELS])
    x_ps = np.array([PSNR[m]  for m in MODELS])
    x_fi = np.array([FID[m]   for m in MODELS])

    y_lim = y_pad(y)

    panels = [
        ("a_sima", x_pa, GOOGLE_CMAP["blue"], r"Sensorimotor Alignment ($\mathrm{S}^2$)", True),
        ("b_ssim", x_ss, GOOGLE_CMAP["grey"], "SSIM",                                     True),
        ("c_psnr", x_ps, GOOGLE_CMAP["grey"], "PSNR (dB)",                                True),
        ("d_fid",  x_fi, GOOGLE_CMAP["grey"], "FID",                                      False),
    ]
    keys = [ALOHA_MODELS[m] for m in MODELS]
    for tag, x, cmap, label, higher_better in panels:
        out = os.path.join(FIG_DIR, f"pa_vs_video_quality_{tag}.pdf")
        panel(out, x, y, cmap, label, higher_better, x_pad(x), y_lim, keys=keys)

    composite_out = os.path.join(FIG_DIR, "pa_vs_video_quality.png")
    _save_composite(composite_out, panels, y, y_lim, keys)


def _save_composite(out_path, panels, y, y_lim, keys):
    """1x4 composite mirroring the LaTeX figures/pa-vs-video-quality.tex layout."""
    fig, axes = plt.subplots(1, 4, figsize=(14.4, 3.8), sharey=True)
    sub_labels = ["(a)", "(b)", "(c)", "(d)"]
    for ax, (tag, x, cmap, label, higher_better), sub in zip(axes, panels, sub_labels):
        x = np.asarray(x, dtype=float); ys = np.asarray(y, dtype=float)
        norm = (x - x.min()) / (x.max() - x.min()) if len(x) > 1 and x.max() > x.min() else np.full_like(x, 0.5)
        facecolors = cmap(0.30 + 0.65 * norm)
        r, p_two = pearsonr(x, ys)
        p_one = p_two / 2.0 if r > 0 else 1.0 - (p_two / 2.0)
        add_fit_with_ci(ax, x, ys, color=cmap(0.85))
        ax.scatter(x, ys, s=160, c=facecolors, edgecolors="black", linewidths=0.8, zorder=3)
        if keys is not None:
            add_labels(ax, x, ys, keys, fontsize=11)
        ax.set_xlim(*x_pad(x)); ax.set_ylim(*y_lim)
        arrow = " $\\rightarrow$" if higher_better else " $\\leftarrow$"
        ax.set_xlabel(label + arrow, fontsize=13)
        ax.set_title(f"{sub} r = {format_r(r)}, p = {p_one:.3f}", fontsize=12)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("RoboTwin success rate (\\%)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.basename(out_path)} (composite)")


if __name__ == "__main__":
    main()
