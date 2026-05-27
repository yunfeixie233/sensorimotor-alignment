
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
"""Appendix §F figure: S^2 vs ImageNet-1K top-1 across vision-encoder
paradigms.

Per-model view: every encoder is plotted as its own labeled dot, hue by
paradigm, intensity by model size within paradigm. Each paradigm family
is also wrapped in a transparent rounded-hull region (computed in
axes-normalised coordinates with a small margin), so the family extent
is visible without an extra marker for the mean.

X = S^2 on DROID at h=7 with N=6,000 anchors. V-JEPA models take a video
clip rather than a single image; we pad the anchor image into a 16-frame
clip before feeding them.

Y = ImageNet-1K top-1 under each source paper's headline protocol:
linear probe for SSL families (MAE, DINO v1, DINOv2, iBOT, I-JEPA),
zero-shot for contrastive (CLIP / OpenCLIP / EVA-CLIP / SigLIP-2),
4-layer attentive probe for V-JEPA / V-JEPA 2 / V-JEPA 2.1.

Two OLS lines:
  * per-model fit through all 22 encoders (dotted, headline)
  * per-family fit through the 4 paradigm means (dashed, secondary)

Colors (paradigm hue, also the legend):
  * Reconstruction (pixel MAE + tokenized BEiT)            blue
  * Self-distillation (DINO v1, DINOv2 family, iBOT)       green
  * Contrastive (CLIP / OpenCLIP / EVA-CLIP / SigLIP-2)    yellow
  * Joint-embedding predictive (I-JEPA + V-JEPA family)    red

VC-1-Base/Large, post-FT SigLIP-2, and V-JEPA 2.1 ViT-L have no
published ImageNet number (under a protocol consistent with the chart)
and are omitted.

Output:
    figures/paradigm_imagenet_scatter.pdf
    figures/paradigm_imagenet_scatter.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from google_palette import GOOGLE_CMAP
from scatter_labels import add_labels, stats_box_text

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
    "mathtext.fontset": "cm",
    "mathtext.default": "regular",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FIG_DIR = str(_FIG_DIR)


SIZE_PARAMS_M = {"S": 22, "B": 86, "L": 304, "SO": 400, "H": 632,
                 "g": 1035, "G": 1100}


# (display, paradigm, size, S^2, ImageNet headline)
COHORT = [
    # Reconstruction-based SSL (pixel MAE) — linear probe
    # BEiT temporarily removed pending an expanded BEiT-family enumeration.
    ("MAE-Base",        "Reconstruction", "B",  0.0205, 68.0),
    ("MAE-Large",       "Reconstruction", "L",  0.0208, 75.8),
    ("MAE-Huge",        "Reconstruction", "H",  0.0208, 76.6),
    # Self-distillation (DINO v1 + DINOv2 family + iBOT) — linear probe
    ("DINO v1-B",       "Self-distillation", "B",  0.0247, 78.2),
    ("DINOv2-Small",    "Self-distillation", "S",  0.0264, 81.1),
    ("DINOv2-Base",     "Self-distillation", "B",  0.0278, 84.5),
    ("DINOv2-Large",    "Self-distillation", "L",  0.0261, 86.3),
    ("DINOv2-Giant",    "Self-distillation", "G",  0.0257, 86.5),
    ("iBOT-Base",       "Self-distillation", "B",  0.0260, 79.5),
    # Contrastive vision-language — zero-shot
    ("CLIP-L/14",       "Contrastive",    "L",  0.0240, 75.3),
    ("OpenCLIP-L/14",   "Contrastive",    "L",  0.0251, 75.2),
    ("EVA-CLIP-L/14",   "Contrastive",    "L",  0.0248, 79.8),
    ("SigLIP-2-Base",   "Contrastive",    "B",  0.0236, 78.2),
    ("SigLIP-2-Large",  "Contrastive",    "L",  0.0250, 83.1),
    ("SigLIP-2-SO400M", "Contrastive",    "SO", 0.0254, 84.1),
    # Joint-embedding predictive (I-JEPA + V-JEPA family)
    # I-JEPA: linear probe; V-JEPA*: 4-layer attentive probe, tiled-frame S^2
    ("I-JEPA-H/14",     "JEPA",           "H",  0.0236, 79.3),
    ("V-JEPA-L/16",     "JEPA",           "L",  0.0228, 74.8),
    ("V-JEPA-H/16",     "JEPA",           "H",  0.0238, 75.9),
    ("V-JEPA 2-L",      "JEPA",           "L",  0.0228, 83.5),
    ("V-JEPA 2-H",      "JEPA",           "H",  0.0223, 83.8),
    ("V-JEPA 2-g",      "JEPA",           "g",  0.0245, 84.6),
    ("V-JEPA 2.1-g",    "JEPA",           "g",  0.0235, 84.8),
]


PARADIGM_ORDER = ["Reconstruction", "Self-distillation", "Contrastive", "JEPA"]
PARADIGM_CMAP = {
    "Reconstruction":    GOOGLE_CMAP["blue"],
    "Self-distillation": GOOGLE_CMAP["green"],
    "Contrastive":       GOOGLE_CMAP["yellow"],
    "JEPA":              GOOGLE_CMAP["red"],
}


def draw_family_region(ax, xs, ys, color, alpha=0.13, pad_px=14, lw=1.0):
    """Transparent on-screen circle enclosing a family of (x, y) points.

    The circle is constructed in display (pixel) coordinates so it
    renders as a true circle regardless of axes aspect ratio, then the
    256-vertex polygon is inverse-transformed back to data coordinates
    for plotting. Call this AFTER ``fig.tight_layout()`` and a
    ``fig.canvas.draw()`` so the axes' display transform is final.

    Center = family centroid (data coords). Radius = max distance from
    the centroid to any family member in pixel coords, plus ``pad_px``
    pixels of margin so every member sits inside with a small buffer.
    """
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    data_pts = np.column_stack([xs, ys])
    disp_pts = ax.transData.transform(data_pts)
    c_disp = disp_pts.mean(0)
    spread = float(np.linalg.norm(disp_pts - c_disp, axis=1).max()) if len(disp_pts) > 1 else 0.0
    r_disp = spread + pad_px

    theta = np.linspace(0, 2 * np.pi, 256)
    ring_disp = c_disp + r_disp * np.column_stack([np.cos(theta), np.sin(theta)])
    ring_data = ax.transData.inverted().transform(ring_disp)
    ax.fill(ring_data[:, 0], ring_data[:, 1], color=color, alpha=alpha,
            edgecolor=color, lw=lw, zorder=1)


def main():
    fig, ax = plt.subplots(figsize=(9.5, 7.0))

    xs = np.array([r[3] for r in COHORT])
    ys = np.array([r[4] for r in COHORT])

    # Per-model scatter: hue by paradigm, intensity by within-paradigm size.
    legend_handles = []
    for paradigm in PARADIGM_ORDER:
        cmap = PARADIGM_CMAP[paradigm]
        rows = [r for r in COHORT if r[1] == paradigm]
        params = np.array([SIZE_PARAMS_M[r[2]] for r in rows], dtype=float)
        if params.max() > params.min():
            norm = (params - params.min()) / (params.max() - params.min())
        else:
            norm = np.full_like(params, 0.5)
        intensities = 0.35 + 0.60 * norm
        for (_name, _par, _sz, s2, im), inten in zip(rows, intensities):
            ax.scatter([s2], [im], s=210, c=[cmap(inten)],
                       edgecolors="black", linewidths=0.7, zorder=3)
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", linestyle="",
                       markerfacecolor=cmap(0.90),
                       markeredgecolor="black", markersize=11,
                       label=paradigm)
        )

    # Per-family centroids (computed for the OLS fit) and the transparent
    # hull regions that visually encode "these encoders belong to one
    # paradigm family". Drawn AFTER the limits are set, just below the
    # per-model dots in z-order so the dots stay on top.
    mean_xs, mean_ys = [], []
    family_pts = {}
    for paradigm in PARADIGM_ORDER:
        rows = [r for r in COHORT if r[1] == paradigm]
        s2_arr = np.array([r[3] for r in rows])
        im_arr = np.array([r[4] for r in rows])
        family_pts[paradigm] = (s2_arr, im_arr)
        mean_xs.append(s2_arr.mean()); mean_ys.append(im_arr.mean())
    mean_xs = np.array(mean_xs); mean_ys = np.array(mean_ys)

    # Two OLS lines drawn across a slightly padded x-range.
    x_pad = 0.10 * (xs.max() - xs.min())
    x_lo, x_hi = xs.min() - x_pad, xs.max() + x_pad
    x_line = np.array([x_lo, x_hi])

    slope_e, intercept_e = np.polyfit(xs, ys, 1)
    ax.plot(x_line, slope_e * x_line + intercept_e,
            color="#202124", linestyle=(0, (1, 2)), linewidth=1.8, alpha=0.95,
            zorder=2, label="per-model fit")

    slope_p, intercept_p = np.polyfit(mean_xs, mean_ys, 1)
    ax.plot(x_line, slope_p * x_line + intercept_p,
            color="#5F6368", linestyle="--", linewidth=1.6, alpha=0.85,
            zorder=2, label="per-family fit")

    r_enc, p_enc = pearsonr(xs, ys)
    p_enc_1tail = p_enc / 2 if r_enc > 0 else 1 - p_enc / 2
    r_par, p_par = pearsonr(mean_xs, mean_ys)
    p_par_1tail = p_par / 2 if r_par > 0 else 1 - p_par / 2

    # Stats box: per-model first (headline view), per-family second.
    line_enc = stats_box_text(r_enc, p_enc_1tail)
    line_par = stats_box_text(r_par, p_par_1tail)
    ax.text(0.04, 0.97,
            f"Per model (n={len(COHORT)}, dotted):\n{line_enc}",
            transform=ax.transAxes, fontsize=12, va="top", ha="left",
            color="#202124",
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                      edgecolor="#444444", alpha=0.95))
    ax.text(0.04, 0.79,
            f"Per family (n={len(PARADIGM_ORDER)}, dashed):\n{line_par}",
            transform=ax.transAxes, fontsize=12, va="top", ha="left",
            color="#3C4043",
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                      edgecolor="#888888", alpha=0.92))

    # Paradigm legend (lower right).
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.95,
              fontsize=11, title="Pretraining paradigm",
              title_fontsize=11)

    ax.set_xlabel(r"Sensorimotor Alignment ($\mathrm{S}^2$)", fontsize=18)
    ax.set_ylabel(r"ImageNet-1K top-1 (\%)", fontsize=18)
    ax.tick_params(axis="both", labelsize=13)
    from matplotlib.ticker import FormatStrFormatter
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax.grid(True, alpha=0.25)

    # Generously padded axes so labels fit beside their dots.
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    x_rng = max(x_max - x_min, 1e-9)
    y_rng = max(y_max - y_min, 1e-9)
    ax.set_xlim(x_min - 0.12 * x_rng, x_max + 0.14 * x_rng)
    ax.set_ylim(y_min - 0.10 * y_rng, y_max + 0.13 * y_rng)

    # Per-family centroid markers: pentagram (*) so the shape is distinct
    # from the per-encoder circles, but rendered at roughly the same
    # visual size as one model dot. Paradigm-colored, black outline.
    # Each gets a short labeled tag next to it.
    PARADIGM_SHORT = {
        "Reconstruction":    "MAE",
        "Self-distillation": "DINO",
        "Contrastive":       "CLIP",
        "JEPA":              "JEPA",
    }
    # Manual label offsets in display points so they don't collide with
    # nearby per-encoder labels. (dx, dy) is in points relative to the
    # centroid; ha is the horizontal anchor.
    MEAN_LABEL_OFFSETS = {
        "Reconstruction":    (+14, +6,  "left"),
        "Self-distillation": (+14, -8,  "left"),
        "Contrastive":       (-14, -10, "right"),
        "JEPA":              (-14, +8,  "right"),
    }
    for paradigm, mx, my in zip(PARADIGM_ORDER, mean_xs, mean_ys):
        cmap = PARADIGM_CMAP[paradigm]
        ax.scatter([mx], [my], marker="*", s=260, c=[cmap(0.98)],
                   edgecolors="black", linewidths=1.2, zorder=5)
        dx, dy, ha = MEAN_LABEL_OFFSETS[paradigm]
        ax.annotate(f"{PARADIGM_SHORT[paradigm]} mean",
                    xy=(mx, my), xycoords="data",
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=10, fontweight="bold",
                    ha=ha, va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.18",
                              facecolor="white", edgecolor="#bbbbbb",
                              alpha=0.92))

    # Per-encoder text labels: pass display names directly, falls back
    # to the raw name (DISPLAY_NAMES doesn't include vision-encoder keys).
    labels = [r[0] for r in COHORT]
    add_labels(ax, xs, ys, labels, fontsize=9)

    fig.tight_layout()
    # Force a draw so ax.transData reflects the final axes box, then
    # paint the family circles in pixel space so they render as true
    # circles (not ellipses) regardless of axis aspect ratio.
    fig.canvas.draw()
    for paradigm in PARADIGM_ORDER:
        s2_arr, im_arr = family_pts[paradigm]
        cmap = PARADIGM_CMAP[paradigm]
        draw_family_region(ax, s2_arr, im_arr,
                           color=cmap(0.85), alpha=0.13,
                           pad_px=18, lw=1.0)

    out_pdf = os.path.join(FIG_DIR, "paradigm_imagenet_scatter.pdf")
    out_png = os.path.join(FIG_DIR, "paradigm_imagenet_scatter.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"n (per-encoder) = {len(COHORT)}")
    print(f"Per-encoder Pearson r = {r_enc:+.3f}, p (two-tailed) = {p_enc:.4f}")
    print(f"Paradigm-mean Pearson r = {r_par:+.3f}, p (two-tailed) = {p_par:.4f}, n = 4")
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
