
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
"""§3.1 figure: 1×3 scatter showing \\PA predicts success rate for VLA / VLM / WAM.

Panels:
    VLA  : finetuned VLAs (n=12), DROID reference, h=7  → SimplerEnv success rate.
    VLM  : raw VLMs       (n=8),  DROID reference, h=3  → SimplerEnv success rate.
    WAM  : World-Action Models (n=3), ALOHA reference, h=15 → RoboTwin success rate.

Each panel has its own x-axis label. Cohort palette uses Google brand colors:
VLA red, VLM yellow, WAM blue.

Outputs:
    figures/combined_3benchmarks_scatter.pdf  (vector, used in paper)
    figures/combined_3benchmarks_scatter.png  (preview)
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
from scatter_labels import add_labels, add_fit_with_ci, stats_box_text

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

VLA_DROID_CKNNA = str(_DATA / "cknna_vla_droid_complete.csv")
VLA_SR_CSV      = str(_DATA / "widowx_sr.csv")
RAWVLM_CKNNA    = str(_DATA / "cknna_vlm_droid_complete.csv")
RAWVLM_SR_CSV   = str(_DATA / "benchmark_sr.csv")
VA_ALOHA_MASTER = str(_DATA / "cknna_wam_aloha_complete.csv")

VA_KEEP = {"lingbot-va-posttrain-robotwin", "motus-robotwin2-aloha-native", "vidar_aloha_native"}

COHORT_CMAP = {"VLA": GOOGLE_CMAP["red"], "VLM": GOOGLE_CMAP["yellow"], "WAM": GOOGLE_CMAP["blue"]}

# Manual label-position overrides (offset from dot, in points).
# These bypass adjustText and pin the label to a fixed offset, used
# where automatic placement put a label on top of another dot or
# created an ambiguous label-to-dot mapping. Visual review:
# 2026-05-07 on combined_3benchmarks_scatter.pdf.
MANUAL_OFFSETS = {
    "VLA": {
        # CogACT-B and CogACT-S have nearly coincident dots
        # (PA=0.032, SR=51). Stack labels in opposite directions so
        # the reader can tell which is which: B above-right, S
        # below-right.
        "cogact-base-bridge":  ( 8,  10),
        "cogact-small-bridge": ( 8, -16),
        # Qwen3VL-GR00T sits left-of-and-above the Qwen-GR00T-RT1
        # dot, with its dot easy to confuse with the stats box edge.
        # Push the label well below-left of the dot and draw a
        # leader so the dot is unambiguously identified.
        "Qwen3VL-GR00T-Bridge-RT-1": (-25, -32),
    },
    "VLM": {
        # PaliGemma-2 default placement put its label on top of the
        # Qwen3VL-8B dot (the brightest, top-right point). Push the
        # label far enough left to clear that dot.
        "paligemma2-raw": (-20, 10),
    },
}


def _csv(path):
    with open(path) as f: return list(csv.DictReader(f))


def load_vla():
    # Drop the VLM4VLA-style finetuned rows (vlm4vla-qwen25vl3b-...): those
    # belong to the §4 frozen-vs-unfrozen ablation, not the headline n=12
    # finetuned-VLA cohort plotted here.
    sr = {r["model_key"]: float(r["success_rate"])
          for r in _csv(VLA_SR_CSV)
          if r.get("excluded", "").strip().lower() != "true"
          and not r["model_key"].startswith("vlm4vla")}
    return [(float(r["cknna_k10"]), sr[r["model"]], r["model"])
            for r in _csv(VLA_DROID_CKNNA)
            if r["feats_A_variant"] == "imgtext" and int(float(r["horizon"])) == 7
            and r["model"] in sr]


def load_vlm():
    sr = {r["model_key"]: r for r in _csv(RAWVLM_SR_CSV)}
    out = []
    for r in _csv(RAWVLM_CKNNA):
        if r["feats_A_variant"] != "imgtext": continue
        if int(float(r["horizon"])) != 3: continue
        if r["model"] not in sr: continue
        v = sr[r["model"]].get("simpler_sr", "")
        if v in ("", None): continue
        out.append((float(r["cknna_k10"]), float(v), r["model"]))
    return out


def load_va():
    out = []
    for r in _csv(VA_ALOHA_MASTER):
        if r["model"] not in VA_KEEP: continue
        if r["feature_source"] != "V-B7": continue
        if int(float(r["horizon"])) != 15: continue
        v = r.get("robotwin_sr_avg", "")
        if v in ("", None): continue
        out.append((float(r["cknna_k10"]), float(v), r["model"]))
    return out


def panel(ax, pairs, cohort, y_label, title, y_lim=None):
    cmap = COHORT_CMAP[cohort]
    xs = np.asarray([p[0] for p in pairs])
    ys = np.asarray([p[1] for p in pairs])
    keys = [p[2] for p in pairs]
    norm = (xs - xs.min()) / (xs.max() - xs.min()) if len(xs) > 1 and xs.max() > xs.min() else np.full_like(xs, 0.5)
    add_fit_with_ci(ax, xs, ys, color=cmap(0.85))
    ax.scatter(xs, ys, s=180, c=cmap(0.30 + 0.65 * norm),
               edgecolors="black", linewidths=0.7, zorder=3)
    r, p2 = pearsonr(xs, ys) if len(xs) >= 3 else (float("nan"), float("nan"))
    p1 = p2 / 2 if r > 0 else 1 - p2 / 2
    ax.text(0.04, 0.96, stats_box_text(r, p1), transform=ax.transAxes,
            fontsize=20, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#888888", alpha=0.92))
    ax.set_title(title, fontsize=22, fontweight="bold", pad=10)
    ax.set_xlabel(r"Sensorimotor Alignment ($\mathrm{S}^2$)", fontsize=20)
    ax.set_ylabel(y_label, fontsize=20)
    ax.tick_params(axis="both", labelsize=16)
    # Standardize S^2 x-tick labels to 3 decimal places across all
    # Pearson scatters in the paper (Figs 3, 4-left, 5a, 6, 8, 9).
    from matplotlib.ticker import FormatStrFormatter
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.grid(True, alpha=0.25)
    if len(xs) >= 2:
        # Generous padding so per-point labels sit beside their dots
        # without clipping at the panel edge. The new 11pt label size
        # + +7pt offset needs more room than the original 0.10/0.10.
        x_min, x_max = float(xs.min()), float(xs.max())
        x_rng = max(x_max - x_min, 1e-9)
        ax.set_xlim(x_min - 0.25 * x_rng, x_max + 0.30 * x_rng)
        if y_lim is not None:
            ax.set_ylim(*y_lim)
        else:
            y_min, y_max = float(ys.min()), float(ys.max())
            y_rng = max(y_max - y_min, 1e-9)
            ax.set_ylim(y_min - 0.15 * y_rng, y_max + 0.18 * y_rng)
    add_labels(ax, xs, ys, keys, fontsize=13,
               manual_offsets=MANUAL_OFFSETS.get(cohort))


def main():
    panels = [
        ("VLA", load_vla, "SimplerEnv success rate (\\%)", "VLAs"),
        ("VLM", load_vlm, "SimplerEnv success rate (\\%)", "VLMs"),
        ("WAM",  load_va,  "RoboTwin success rate (\\%)", "WAMs"),
    ]
    # Pre-compute the SimplerEnv y-range shared by VLA and VLM (both panels
    # plot SimplerEnv success rate on the y-axis, just for different cohorts).
    # The WAM panel keeps its own RoboTwin scale.
    pairs_per_cohort = {c: ld() for c, ld, _, _ in panels}
    simpler_ys = np.concatenate([
        np.asarray([p[1] for p in pairs_per_cohort["VLA"]]),
        np.asarray([p[1] for p in pairs_per_cohort["VLM"]]),
    ])
    simpler_y_pad = 0.10 * (simpler_ys.max() - simpler_ys.min() + 1e-9)
    simpler_y_lim = (simpler_ys.min() - simpler_y_pad,
                     simpler_ys.max() + simpler_y_pad)
    print(f"Shared SimplerEnv y-range: {simpler_y_lim[0]:.1f} .. {simpler_y_lim[1]:.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    for ax, (cohort, loader, y_label, title) in zip(axes, panels):
        pairs = pairs_per_cohort[cohort]
        if not pairs:
            print(f"WARNING: empty for {cohort}", file=sys.stderr); continue
        y_lim = simpler_y_lim if cohort in ("VLA", "VLM") else None
        panel(ax, pairs, cohort, y_label, title, y_lim=y_lim)

    fig.tight_layout()
    out_pdf = os.path.join(FIG_DIR, "combined_3benchmarks_scatter.pdf")
    out_png = os.path.join(FIG_DIR, "combined_3benchmarks_scatter.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
