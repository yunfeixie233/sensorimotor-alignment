"""Shared helpers for paper scatter figures.

Provides a single source of truth for:
  * DISPLAY_NAMES   : model_key -> short display label
  * add_labels      : draw per-point text labels with overlap avoidance
                      (adjustText) plus a dashed linear regression fit;
                      the fit gets no legend entry.

Generators import these so every panel in the paper uses identical labels
and the same fit-line styling.
"""

from __future__ import annotations

import numpy as np
from adjustText import adjust_text


# Short, paper-friendly display names. Keys are the model_key strings used
# across the cknna_*_complete.csv files under record/fig_*/data/.
DISPLAY_NAMES = {
    # ---- finetuned VLAs (n=12, BridgeDataV2/DROID -> SimplerEnv-WidowX) ----
    "Qwen-GR00T-Bridge":          "Qwen-GR00T",
    "Qwen-GR00T-Bridge-RT-1":     "Qwen-GR00T-RT1",
    "Qwen-OFT-Bridge-RT-1":       "Qwen-OFT-RT1",
    "Qwen3VL-GR00T-Bridge-RT-1":  "Qwen3VL-GR00T",
    "Qwen3VL-OFT-Bridge-RT-1":    "Qwen3VL-OFT",
    "cogact-small-bridge":        "CogACT-S",
    "cogact-base-bridge":         "CogACT-B",
    "cogact-large-bridge":        "CogACT-L",
    "groot-n15-bridge":           "GR00T-N1.5",
    "groot-n16-bridge":           "GR00T-N1.6",
    "pi0-lerobot-bridge":         "Pi0",
    "spatialvla-sft-bridge":      "SpatialVLA",
    "openvla-7b-bridge":          "OpenVLA",
    "openvla-7b-bridge-ft-200k":  "OpenVLA-FT",
    # ---- raw VLMs (n=8, DROID -> SimplerEnv after VLM4VLA finetune) ----
    "qwen25vl-3b-raw":            "Qwen2.5VL-3B",
    "qwen25vl-7b-raw":            "Qwen2.5VL-7B",
    "qwen3vl-2b-raw":             "Qwen3VL-2B",
    "qwen3vl-4b-raw":             "Qwen3VL-4B",
    "qwen3vl-8b-raw":             "Qwen3VL-8B",
    "paligemma1-raw":             "PaliGemma-1",
    "paligemma2-raw":             "PaliGemma-2",
    "kosmos2-raw":                "KosMos-2",
    # ---- World-Action Models (n=3, ALOHA -> RoboTwin) ----
    "lingbot-va-posttrain-robotwin": "LingBot-VA",
    "motus-robotwin2-aloha-native":  "Motus",
    "vidar_aloha_native":            "Vidar",
    # Same three WAMs but with the row-name conventions used by the DROID
    # master_finetuned table (different inference layout than ALOHA-native).
    "motus-robotwin2-lingbot":       "Motus",
    "vidar_3cam_modeB":              "Vidar",
}


def display(key: str) -> str:
    """Return the short paper label for a model key, falling back to the key."""
    return DISPLAY_NAMES.get(key, key)


def add_fit_line(ax, xs, ys, color="#666666", linewidth=1.0, alpha=0.75):
    """Draw a dashed linear-regression line over the panel's data.

    No legend entry: the line carries no label.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 2 or np.std(xs) == 0:
        return
    slope, intercept = np.polyfit(xs, ys, 1)
    x_line = np.linspace(xs.min(), xs.max(), 100)
    ax.plot(x_line, slope * x_line + intercept,
            linestyle="--", color=color,
            linewidth=linewidth, alpha=alpha,
            zorder=2, label="_nolegend_")


def add_fit_with_ci(ax, xs, ys, ci=0.95, color="#666666",
                    band_color=None, band_alpha=0.18,
                    linewidth=1.2, alpha=0.85, n_grid=120,
                    show_ci=False, x_extend=None):
    """Draw a dashed OLS regression line, optionally with a shaded CI band.

    Default is `show_ci=False`: just the dashed fit line, no shaded
    band. Per the 2026-05-06 review the paper now omits CI bands on
    every Pearson scatter -- the headline message is the slope sign +
    magnitude, and the band was distracting readers from that.

    Mirrors the seaborn.regplot default when `show_ci=True`: the band
    is the 95% confidence interval of the regression *line* (where the
    true line lies), not a prediction interval for individual points.
    The CI is computed from OLS residuals via the closed-form formula
        se_line(x) = sqrt(MSE * (1/n + (x - x_mean)^2 / Sxx))
    multiplied by t_{ci, df=n-2}. Bowtie shape: narrowest at x_mean,
    fans out at the data edges.

    For n<3 we cannot compute a CI (df<=0): we draw the fit line only.
    """
    from scipy.stats import t as student_t
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = len(xs)
    if n < 2 or np.std(xs) == 0:
        return
    slope, intercept = np.polyfit(xs, ys, 1)
    if x_extend is not None:
        x_lo, x_hi = x_extend
    else:
        x_lo, x_hi = xs.min(), xs.max()
    x_grid = np.linspace(x_lo, x_hi, n_grid)
    y_grid = intercept + slope * x_grid
    if show_ci and n >= 3:
        x_mean = xs.mean()
        residuals = ys - (intercept + slope * xs)
        mse = (residuals ** 2).sum() / (n - 2)
        sxx = ((xs - x_mean) ** 2).sum()
        if sxx > 0 and mse > 0:
            se_line = np.sqrt(mse * (1.0 / n + (x_grid - x_mean) ** 2 / sxx))
            t_crit = student_t.ppf(0.5 + ci / 2.0, df=n - 2)
            half = t_crit * se_line
            ax.fill_between(
                x_grid, y_grid - half, y_grid + half,
                color=(band_color if band_color is not None else color),
                alpha=band_alpha, linewidth=0,
                zorder=1, label="_nolegend_",
            )
    ax.plot(
        x_grid, y_grid,
        linestyle="--", color=color,
        linewidth=linewidth, alpha=alpha,
        zorder=2, label="_nolegend_",
    )


def format_p(p):
    """Render a p-value with 3 decimals, but '< 0.001' below threshold."""
    return "< 0.001" if p < 1e-3 else f"{p:.3f}"


def format_r(r, decimals=3, signed=True):
    """Render Pearson r with TRUNCATION (not rounding) to `decimals` places.

    Example: format_r(+0.999990) -> '+0.999' (not '+1.000'). Avoids the
    misleading 'r = 1.000' display when the true value is 0.99999 — a
    rounding artefact that masquerades as a perfect correlation. Negative
    numbers are truncated toward zero (e.g. -0.5005 -> '-0.500').
    Implemented via string-slice on a 10-decimal expansion to sidestep
    float-precision artefacts (e.g. 0.700 stored as 0.6999... still
    renders as '+0.700').

    Args
    ----
    r        : float                Pearson r in [-1, 1].
    decimals : int (default 3)      Number of decimal places to keep.
    signed   : bool (default True)  If True, prefix '+' on non-negative
                                    values (matches '{r:+.3f}' style).
                                    If False, only '-' on negatives
                                    (matches '{rv:.2f}' per-marker style).
    """
    if signed:
        s = f"{r:+.10f}"
    else:
        s = f"{r:.10f}"
    dot = s.index(".")
    return s[:dot + 1 + decimals]


SIG_THRESHOLD = 0.05  # standard significance cutoff (95% confidence)


def stats_box_text(r, p, threshold=SIG_THRESHOLD):
    """Standard stats-box text for Pearson scatter panels.

    Two-line block; deliberately drops `n` per paper-style decision.
    The full term "Pearson correlation" lives in the figure caption;
    here we just write "correlation $r$" to keep the in-figure box
    short. Convention follows the AI-Wellbeing paper (Fig 4 etc.) of
    spelling the symbol out only on first mention.
        "correlation r = +0.717"
        "p-value     = 0.004"

    When p > threshold (default 0.05) the p-value's right-hand side is
    rendered in mathtext bold and the threshold is appended to flag
    non-significance, e.g. "p-value = 0.342 > 0.05" with the
    "0.342 > 0.05" portion bolded. Significant panels render unchanged.

    r is truncated (not rounded) to 3 decimals — see `format_r`.
    """
    if p > threshold:
        # Bold the entire p-value line in one mathtext span (including the
        # "p-value =" prefix and the appended "> 0.05" suffix). \text{-}
        # is supported in default mathtext (>=3.4) and renders a real
        # hyphen rather than a math minus, so "p-value" reads correctly.
        p_line = f"$\\mathbf{{p\\text{{-}}value = {format_p(p)} > {threshold}}}$"
    else:
        p_line = f"$p$-value $= {format_p(p)}$"
    return f"correlation $r = {format_r(r)}$\n{p_line}"


def add_labels(ax, xs, ys, keys, fontsize=8,
               text_color="black",
               arrow_color="#888888",
               manual_offsets=None):
    """Annotate each (x, y) with its display name, then call adjustText to
    fan overlapping labels apart with thin grey leader lines.

    Parameters
    ----------
    keys : iterable of str
        Model keys -- looked up in DISPLAY_NAMES.
    fontsize : int
        Label size. 7-8 pt fits 12-model panels without crowding most of
        the time; bump to 9 for sparser panels.
    manual_offsets : dict, optional
        Map of model key -> (dx_pts, dy_pts) for manual label placement
        in points relative to the dot. Models in this dict have their
        label pinned at the offset and are excluded from adjustText. The
        ha/va alignment is inferred from the sign of dx/dy so that the
        label appears on the offset side of the dot.

    Notes
    -----
    The scatter PathCollection on the axes is gathered and passed to
    adjustText as a static-object list, so labels actively repel from
    every dot (not just from one another). Combined with the higher
    `force_static` below, this eliminates the previous label-on-dot
    overlap that was visible in Fig 3 (Pi0, CogACT-S/B, SpatialVLA).
    """
    manual_offsets = manual_offsets or {}
    auto_texts = []
    for x, y, k in zip(xs, ys, keys):
        label = display(k)
        if k in manual_offsets:
            dx, dy = manual_offsets[k]
            ha = "left" if dx >= 0 else "right"
            va = "bottom" if dy >= 0 else "top"
            # Draw a thin grey leader from the dot to the offset label
            # so the reader can trace the connection at a glance, then
            # place the label text at the same offset.
            ax.annotate(label, xy=(x, y), xytext=(dx, dy),
                        textcoords="offset points",
                        fontsize=fontsize, color=text_color,
                        ha=ha, va=va, zorder=4,
                        arrowprops=dict(arrowstyle="-",
                                        color=arrow_color,
                                        lw=0.5, alpha=0.7,
                                        shrinkA=0, shrinkB=4))
        else:
            # Anchor with bottom-left at dot centre. The previous default
            # has been validated against Figs 4-9; with the new larger
            # fontsize (>=10) the bottom-left baseline already sits visibly
            # above the dot upper edge.
            auto_texts.append(ax.text(x, y, label, fontsize=fontsize,
                                      color=text_color, ha="left",
                                      va="bottom", zorder=4))
    # Original tuning, with `ensure_inside_axes` made explicit so labels
    # never drift outside the panel. NB: do NOT pass `objects=` --
    # adjustText 1.3.0 crashes when its explode() codepath runs over
    # PathCollections.
    adjust_text(
        auto_texts, ax=ax,
        arrowprops=dict(arrowstyle="-",
                        color=arrow_color, lw=0.5, alpha=0.7),
        expand=(1.05, 1.15),
        force_text=(0.4, 0.5),
        force_static=(0.2, 0.3),
        force_pull=(0.5, 0.5),
        max_move=40,
        ensure_inside_axes=True,
    )
