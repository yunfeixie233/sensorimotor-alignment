
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
"""§3.1 raw-VLM figure: 1×2 scatter showing
    (a) \PA on the raw VLM predicts the resulting VLA's SimplerEnv success rate
    (b) VLM general capability (avg VQA, downweighted) does NOT.

Cohort: 8 raw VLMs that VLM4VLA finetuned into VLAs. The VLM general capability
recipe and per-model VQA averages come from
/lambda/nfs/vla/reproduction/vlm4vla_fig3/reproduce_fig3.py (BASELINE=49.0,
ALPHA=1.9 best fit to VLM4VLA paper r-values).

Outputs:
    figures/pa_vs_vqa_simpler.pdf   (vector, used in paper)
    figures/pa_vs_vqa_simpler.png   (preview)
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

from google_palette import GOOGLE_CMAP
from scatter_labels import add_labels, add_fit_with_ci, stats_box_text, format_r

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

RAWVLM_CKNNA = str(_DATA / "cknna_vlm_droid_complete.csv")
RAWVLM_SR_CSV = str(_DATA / "benchmark_sr.csv")


# Per-model VQA averages and benchmark counts (from the VLM4VLA reproduction).
VLM_BENCH = {
    # name in cknna CSV, raw-avg VQA score over N_M of 18 benchmarks, N_M
    "qwen25vl-3b-raw":  {"raw_avg": 61.93, "N":  13, "name": "Qwen2.5VL-3B"},
    "qwen25vl-7b-raw":  {"raw_avg": 66.74, "N":  13, "name": "Qwen2.5VL-7B"},
    "qwen3vl-2b-raw":   {"raw_avg": 60.52, "N":  16, "name": "Qwen3VL-2B"},
    "qwen3vl-4b-raw":   {"raw_avg": 68.40, "N":  16, "name": "Qwen3VL-4B"},
    "qwen3vl-8b-raw":   {"raw_avg": 70.42, "N":  16, "name": "Qwen3VL-8B"},
    "paligemma1-raw":   {"raw_avg": 72.28, "N":   4, "name": "Paligemma-1"},
    "paligemma2-raw":   {"raw_avg": 72.39, "N":   4, "name": "Paligemma-2"},
    "kosmos2-raw":      {"raw_avg": 52.21, "N":   1, "name": "KosMos-2"},
}

ALPHA, BASELINE, N_TARGET = 1.9, 49.0, 18


def vlm_capability(raw_avg, N):
    return BASELINE + (raw_avg - BASELINE) * (N / N_TARGET) ** ALPHA


def _csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_data():
    """For each model: SA at imgtext h=3 on DROID; VQA cap; SimplerEnv VLA SR."""
    sr_rows = {r["model_key"]: r for r in _csv(RAWVLM_SR_CSV)}
    out = []
    for r in _csv(RAWVLM_CKNNA):
        if r["feats_A_variant"] != "imgtext": continue
        if int(float(r["horizon"])) != 3: continue
        m = r["model"]
        if m not in VLM_BENCH or m not in sr_rows: continue
        v = sr_rows[m].get("simpler_sr", "")
        if v in ("", None): continue
        info = VLM_BENCH[m]
        out.append({
            "model": m,
            "name":  info["name"],
            "sa":    float(r["cknna_k10"]),
            "vqa_cap": vlm_capability(info["raw_avg"], info["N"]),
            "sr":    float(v),
        })
    return out


def panel(ax, x, y, cmap, x_label, stats_loc="left", keys=None):
    if len(x) > 1 and x.max() > x.min():
        norm = (x - x.min()) / (x.max() - x.min())
    else:
        norm = np.full_like(x, 0.5)
    facecolors = cmap(0.30 + 0.65 * norm)
    add_fit_with_ci(ax, x, y, color=cmap(0.85))
    ax.scatter(x, y, s=180, c=facecolors, edgecolors="black",
               linewidths=0.7, zorder=3)
    r, p2 = pearsonr(x, y) if len(x) >= 3 else (float("nan"), float("nan"))
    p1 = p2 / 2 if r > 0 else 1 - p2 / 2
    stats = stats_box_text(r, p1)
    # stats_loc: "left" -> top-left; "right" -> top-right;
    # "bottom-left" / "bottom-right" -> bottom corners (used when the
    # corresponding top corner has data labels nearby).
    pos_map = {
        "left":         (0.04, 0.96, "top",    "left"),
        "right":        (0.96, 0.96, "top",    "right"),
        "bottom-left":  (0.04, 0.04, "bottom", "left"),
        "bottom-right": (0.96, 0.04, "bottom", "right"),
    }
    sx, sy, sva, sha = pos_map.get(stats_loc, pos_map["left"])
    # Per-figure font scaling: this figure renders at ~0.29x source
    # width (13.5in vs 0.7 textwidth). Sizes bumped slightly so axis
    # labels match Fig 3's rendered size at ~0.327x.
    # Stats box at fontsize=18 (vs xlabel=23): the right-panel VLM
    # data is scattered across all four corners (KosMos-2 NW,
    # Qwen3VL-8B NE, Qwen2.5VL-7B SW, Qwen3VL-2B SE), so a 23pt box
    # would overlap whichever corner we anchor it in. 18pt is the
    # largest size that clears every cohort point.
    ax.text(sx, sy, stats, transform=ax.transAxes, fontsize=18,
            va=sva, ha=sha,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#888888", alpha=0.92))
    ax.set_xlabel(x_label, fontsize=23)
    ax.tick_params(axis="both", labelsize=18)
    # Standardise S^2 x-tick labels to 3 decimal places (matches Figs
    # 3, 5a, 6, 8, 9). Only the left panel (S^2) gets this; the right
    # panel uses VLM general capability on a 0-100 scale.
    if r"\mathrm{S}^2" in x_label:
        from matplotlib.ticker import FormatStrFormatter
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.grid(True, alpha=0.25)
    if len(x) >= 2:
        x_pad = 0.20 * (x.max() - x.min() + 1e-9)
        y_pad = 0.12 * (y.max() - y.min() + 1e-9)
        ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
        ax.set_ylim(y.min() - y_pad, y.max() + y_pad)
    if keys is not None:
        add_labels(ax, x, y, keys, fontsize=15)


def main():
    rows = load_data()
    if not rows:
        print("WARNING: no data", file=sys.stderr); sys.exit(1)
    sa  = np.array([r["sa"]      for r in rows])
    cap = np.array([r["vqa_cap"] for r in rows])
    sr  = np.array([r["sr"]      for r in rows])
    print(f"{'Model':<22}{'SA':>8}{'VQA cap':>10}{'SR (%)':>9}")
    for r in rows:
        print(f"{r['name']:<22}{r['sa']:>8.4f}{r['vqa_cap']:>10.2f}{r['sr']:>9.1f}")
    r_sa, p_sa = pearsonr(sa, sr)
    r_cap, p_cap = pearsonr(cap, sr)
    p_sa_one = p_sa / 2 if r_sa > 0 else 1 - p_sa / 2
    p_cap_one = p_cap / 2 if r_cap > 0 else 1 - p_cap / 2
    print(f"\nLeft  panel: SiMA vs SimplerEnv SR     r={format_r(r_sa)}  p_one={p_sa_one:.4f}")
    print(f"Right panel: VQA cap vs SimplerEnv SR r={format_r(r_cap)}  p_one={p_cap_one:.4f}")

    keys = [r["model"] for r in rows]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4), sharey=True)
    panel(axL, sa,  sr, GOOGLE_CMAP["blue"],
          x_label=r"Sensorimotor Alignment ($\mathrm{S}^2$)",
          stats_loc="left", keys=keys)
    panel(axR, cap, sr, GOOGLE_CMAP["grey"],
          x_label="VLM general capability (avg VQA)",
          stats_loc="bottom-left", keys=keys)
    axL.set_ylabel("SimplerEnv success rate (\\%)", fontsize=23)

    fig.tight_layout()
    out_pdf = os.path.join(FIG_DIR, "pa_vs_vqa_simpler.pdf")
    out_png = os.path.join(FIG_DIR, "pa_vs_vqa_simpler.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
