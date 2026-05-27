
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
"""§3.2 figure: 2×3 grid of scatters answering 'what carries the prediction?'.

Two settings × three LLM-output positions = 6 panels, both rows at the same
representative horizon h=3.

  Settings (rows):
    (a) Finetuned VLA × CKNNA on DROID × SimplerEnv-WidowX SR, h=3, n=12
    (b) Raw VLM      × CKNNA on DROID × SimplerEnv SR,         h=3, n=8

  Positions (columns):
    LLM-out V      (mean over image tokens)
    LLM-out T      (mean over instruction tokens via build_task_mask)
    LLM-out V+T    (mean over image ∪ instruction tokens)

Two findings highlighted:
  1. V+T highest in both rows ⇒ vision+text fusion is the consistent best
     predictor.
  2. VLA-T < VLM-T and VLA-T < 0.5 ⇒ text-only predictive power collapses
     after manipulation finetuning.

Inputs (relative to this script's `code/` dir, resolved at runtime):
  Setting (a) CKNNA  : ../data/cknna_vla_droid_complete.csv
  Setting (a) SR     : ../data/widowx_sr.csv
  Setting (b) CKNNA  : ../data/cknna_vlm_droid_complete.csv
  Setting (b) SR     : ../data/benchmark_sr.csv

Outputs:
  figures/vt_what_carries_2x3.pdf  (vector, used in paper)
  figures/vt_what_carries_2x3_preview.png  (preview)

Run:
  cd figures && /opt/miniconda/bin/python regen_what_carries_2x3.py
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import pearsonr

from google_palette import GOOGLE_CMAP
from scatter_labels import add_labels, add_fit_with_ci, stats_box_text


FIG_DIR = str(_FIG_DIR)

VLA_DROID = str(_DATA / "cknna_vla_droid_complete.csv")
VLA_SR    = str(_DATA / "widowx_sr.csv")
RAWVLM    = str(_DATA / "cknna_vlm_droid_complete.csv")
RAWVLM_SR = str(_DATA / "benchmark_sr.csv")

H = 3  # representative horizon for both rows


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_vla_sr(path):
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


def load_raw_sr(path):
    return {r["model_key"]: r for r in csv.DictReader(open(path))}


def load_cknna(path):
    return list(csv.DictReader(open(path)))


def get_pairs(rows, sr_dict, variant, horizon, sr_col=None):
    xs, ys, ks = [], [], []
    for r in rows:
        if r["feats_A_variant"] != variant:
            continue
        if int(float(r["horizon"])) != int(horizon):
            continue
        m = r["model"]
        if m not in sr_dict:
            continue
        if sr_col is None:
            sr = sr_dict[m]
        else:
            v = sr_dict[m].get(sr_col, "")
            if v in ("", None):
                continue
            sr = float(v)
        xs.append(float(r["cknna_k10"]))
        ys.append(sr)
        ks.append(m)
    return np.asarray(xs), np.asarray(ys), ks


def _panel(ax, x, y, cmap, modality, setting_label, x_lim=None, y_lim=None,
           y_label=None, keys=None, fit_extend=None):
    if len(x) > 1 and x.max() > x.min():
        norm = (x - x.min()) / (x.max() - x.min())
    else:
        norm = np.full_like(x, 0.5)
    facecolors = cmap(0.30 + 0.65 * norm)
    # Slightly thicker fit line (1.8 vs default 1.2) per 2026-05-06
    # review -- the previous line read as a faint suggestion of
    # trend rather than as a clear summary of it.
    add_fit_with_ci(ax, x, y, color=cmap(0.85), linewidth=1.8,
                    x_extend=fit_extend)
    ax.scatter(x, y, s=170, c=facecolors, edgecolors="black",
               linewidths=0.7, zorder=3)
    r, p_two = pearsonr(x, y)
    p_one = p_two / 2 if r > 0 else 1 - p_two / 2
    stats = stats_box_text(r, p_one)
    # Per-figure font scaling: 2x3 grid renders at ~0.245x source
    # width (22in vs 0.98 textwidth). Sizes bumped slightly so axis
    # labels match Fig 3's rendered size at ~0.327x.
    # Stats / modality legend at fontsize=22 (vs xlabel=26): the top
    # corners of each panel sit close to high-SR dots (Qwen-GR00T,
    # KosMos-2). At full xlabel size both top boxes nibble into the
    # data points; 22pt clears them while still being clearly bigger
    # than ticks (21pt) and per-point labels (17pt).
    ax.text(0.04, 0.96, stats, transform=ax.transAxes, fontsize=22,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                      edgecolor="#888888", alpha=0.92))
    # Modality-only legend: just the bold modality label
    # ("Vision-Only" / "Text-Only" / "Vision + Text"). The previous
    # second line ("VLA · SimplerEnv\nh=3, n=12") was redundant with
    # the figure caption and main text and was dropped 2026-05-06.
    # Use fontweight="bold" rather than mathtext \bf{}, so the
    # hyphen in "Vision-Only"/"Text-Only" renders as a hyphen
    # rather than as a math-mode minus.
    ax.text(0.96, 0.96, modality,
            transform=ax.transAxes, fontsize=22, fontweight="bold",
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                      edgecolor="#cccccc", alpha=0.85))
    if x_lim is not None:
        ax.set_xlim(*x_lim)
    else:
        pad_x = 0.18 * max(x.max() - x.min(), 1e-9)
        ax.set_xlim(x.min() - pad_x, x.max() + pad_x)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    else:
        pad_y = 0.12 * max(y.max() - y.min(), 1e-9)
        ax.set_ylim(y.min() - pad_y, y.max() + pad_y)
    ax.set_xlabel(r"Sensorimotor Alignment ($\mathrm{S}^2$)", fontsize=26)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=26)
    ax.tick_params(axis="both", labelsize=21)
    # Standardise S^2 x-tick labels to 3 decimal places across all
    # Pearson scatters (Figs 3, 4-left, 5a, 6, 8).
    from matplotlib.ticker import FormatStrFormatter
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    # Match the y-tick count between rows. The default Locator picks
    # ~9 ticks for the narrow VLM range and ~7 for the wider VLA range,
    # which the user flagged as visually unbalanced. nbins=4 collapses
    # both rows to 5 round-number ticks.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
    ax.grid(True, alpha=0.25)
    if keys is not None:
        add_labels(ax, x, y, keys, fontsize=17)


def main():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    vla_sr = load_vla_sr(VLA_SR)
    raw_sr = load_raw_sr(RAWVLM_SR)
    vla_rows = load_cknna(VLA_DROID)
    raw_rows = load_cknna(RAWVLM)

    # Each row = (label, getter_for_(variant, horizon)→(x,y), cmap, sr_label).
    # The per-panel modality legend now shows ONLY the modality label
    # (no "VLA · SimplerEnv" + "h=3, n=12" detail) per 2026-05-06
    # review -- those facts live in the figure caption / main text.
    SETTINGS = [
        ("a", "",
         lambda v, h: get_pairs(vla_rows, vla_sr, v, h),
         GOOGLE_CMAP["red"], "SimplerEnv success rate (\\%)"),
        ("b", "",
         lambda v, h: get_pairs(raw_rows, raw_sr, v, h, sr_col="simpler_sr"),
         GOOGLE_CMAP["yellow"], "SimplerEnv success rate (\\%)"),
    ]
    # Each col defines: (variant, modality_label). Both VLA and VLM cohorts
    # now use the same canonical variant names. Hyphenated forms
    # ("Vision-Only", "Text-Only") match the body-text convention.
    COLS = [
        ("img",     "Vision-Only"),
        ("txt",     "Text-Only"),
        ("imgtext", "Vision + Text"),
    ]

    panel_data = {}
    row_xs = {"a": [], "b": []}
    row_ys = {"a": [], "b": []}
    for r_idx, (letter, setting_label, getter, cmap, sr_label) in enumerate(SETTINGS):
        for c_idx, (variant, modality) in enumerate(COLS):
            x, y, ks = getter(variant, H)
            panel_data[(r_idx, c_idx)] = (x, y, ks, cmap, modality, setting_label,
                                           letter, sr_label)
            row_xs[letter].append(x)
            row_ys[letter].append(y)

    def _range_pad(arrays, pad_lo=0.08, pad_hi=0.08):
        cat = np.concatenate([np.asarray(a) for a in arrays])
        lo, hi = float(cat.min()), float(cat.max())
        rng = max(hi - lo, 1e-9)
        return (lo - pad_lo * rng, hi + pad_hi * rng)

    # Shared X range across all 6 panels (per user request -- direct
    # comparability across rows). Y range stays per-row because VLA SR
    # spans 33-89 while VLM SR spans 47-67, so a shared y-range would
    # waste vertical area for one of them.
    all_xs = row_xs["a"] + row_xs["b"]
    global_x_lim = _range_pad(all_xs, 0.03, 0.03)
    row_x_lim = {"a": global_x_lim, "b": global_x_lim}
    row_y_lim = {r: _range_pad(row_ys[r], 0.08, 0.50) for r in row_ys}
    print(f"  shared X range: {global_x_lim}")
    for r in ("a", "b"):
        print(f"  row {r}  Y range: {row_y_lim[r]}")

    # Print summary per (row, col) with r and p
    print()
    print(f"{'row':<6}{'col':<14}{'variant':<18}{'r':<10}{'p_one':<10}{'n':<5}")
    print("-" * 60)
    for (r_idx, c_idx), (x, y, ks, cmap, modality, setting_label, rkey, sr_label) in panel_data.items():
        rval, p2 = pearsonr(x, y)
        p1 = p2 / 2 if rval > 0 else 1 - p2 / 2
        print(f"{rkey:<6}{modality:<14}{COLS[c_idx][0]:<18}"
              f"{rval:+.3f}    {p1:.4f}    {len(x)}")

    fig, axes = plt.subplots(2, 3, figsize=(22.0, 12.5))
    for (r_idx, c_idx), (x, y, ks, cmap, modality, setting_label, rkey, sr_label) in panel_data.items():
        ax = axes[r_idx, c_idx]
        y_label = sr_label if c_idx == 0 else None
        # Extend VLM-row (bottom) fit lines slightly beyond the data range
        # (~20% of the data span on each side) so the line reaches a bit past
        # the cluster, without stretching to the full panel xlim.
        if rkey == "b":
            xlo, xhi = float(x.min()), float(x.max())
            pad = 0.20 * max(xhi - xlo, 1e-9)
            fit_extend = (xlo - pad, xhi + pad)
        else:
            fit_extend = None
        _panel(ax, x, y, cmap, modality, setting_label,
               x_lim=row_x_lim[rkey], y_lim=row_y_lim[rkey], y_label=y_label,
               keys=ks, fit_extend=fit_extend)

    fig.tight_layout()
    out_pdf = os.path.join(FIG_DIR, "vt_what_carries_2x3.pdf")
    out_png = os.path.join(FIG_DIR, "vt_what_carries_2x3_preview.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n  combined PDF → {out_pdf}")
    print(f"  PNG preview  → {out_png}")


if __name__ == "__main__":
    main()
