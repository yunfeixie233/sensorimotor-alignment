
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
"""§3.3(a) figure: VLA r-vs-h line chart at LLM-out V+T (`imgtext`).

Two lines on one panel:
    BridgeDataV2 reference (in-embodiment, 6 horizons {1,3,7,15,25,40}), green
        — dashed line, hollow green circles.
    DROID    reference (cross-embodiment, 10 horizons), blue
        — solid line, filled blue squares.

Both correlate against SimplerEnv-WidowX SR (n=12 finetuned VLAs by default;
openvla rows excluded via the SR file's `excluded` flag).

Disambiguation recipe (chosen variant G):
  - ±4% multiplicative jitter on log-x so co-located markers don't merge at
    the h=15..40 crossover.
  - Distinct line styles: BridgeDataV2 dashed + hollow circles, DROID solid +
    filled squares.
  - Per-marker r-value labels (14 pt bold) placed outside the gap between
    the two lines: the higher line at h gets its label above the marker,
    the lower line gets its label below.

Use `--filter-to-intersection` (recommended) so both lines use the model
set common to both Bridge and DROID CKNNA CSVs (n=12). Without it, Bridge
runs on n=12 (no VLM4VLA features extracted on Bridge) while DROID runs on
n=14 — apples-to-oranges.

Significance threshold for n=12 one-tailed p<0.05: |r| ≥ 0.4973.

Outputs:
    figures/horizon_vla_llmout_vt.pdf
    figures/horizon_vla_llmout_vt_preview.png
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from google_palette import GOOGLE_GREEN, GOOGLE_BLUE
from scatter_labels import format_r


FIG_DIR = str(_FIG_DIR)
VLA_BRIDGE = str(_DATA / "cknna_vla_bridgev2_complete.csv")
VLA_DROID  = str(_DATA / "cknna_vla_droid_complete.csv")
VLA_SR     = str(_DATA / "widowx_sr.csv")

HORIZONS_BRIDGE = [1, 3, 7, 15, 25, 40]
HORIZONS_DROID  = [1, 3, 7, 15, 25, 40, 75, 110, 150, 220]

VARIANT = "imgtext"  # LLM-out V+T (legacy; finetuned VLAs not re-extracted with
                     # the round-6 V+T fix — disclosed in caption)
SIG_THRESH_ONE_TAILED = 0.4973  # n=12, df=10, t=qt(0.95,10)=1.812


def load_sr(path):
    # Drop the VLM4VLA-style finetuned rows (vlm4vla-qwen25vl3b-...): those
    # belong to the §4 frozen-vs-unfrozen ablation, not the headline n=12
    # finetuned-VLA cohort plotted here. They live in the DROID CKNNA CSV
    # (since the 2026-05-02 cohort extension) but not in the Bridge CSV,
    # which would otherwise produce a Bridge n=12 vs DROID n=14
    # apples-to-oranges r-vs-h plot.
    sr = {}
    for r in csv.DictReader(open(path)):
        if r.get("excluded", "").strip().lower() == "true":
            continue
        if r["model_key"].startswith("vlm4vla"):
            continue
        sr[r["model_key"]] = float(r["success_rate"])
    return sr


def csv_models(rows_csv):
    return {r["model"] for r in csv.DictReader(open(rows_csv))}


def trace(rows_csv, variant, horizons, sr, model_filter=None):
    out = []
    rows = list(csv.DictReader(open(rows_csv)))
    for h in horizons:
        xs, ys = [], []
        for r in rows:
            if r["feats_A_variant"] != variant: continue
            if int(float(r["horizon"])) != h: continue
            if r["model"] not in sr: continue
            if model_filter is not None and r["model"] not in model_filter: continue
            xs.append(float(r["cknna_k10"]))
            ys.append(sr[r["model"]])
        if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
            continue
        rval, _ = pearsonr(xs, ys)
        out.append((h, rval, len(xs)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter-to-intersection", action="store_true",
                        help="Restrict both lines to models present in BOTH "
                             "Bridge and DROID CKNNA CSVs (and in SR). Use "
                             "this when one CSV has been re-extracted with a "
                             "wider cohort and the other has not, so the two "
                             "lines stay apples-to-apples.")
    args = parser.parse_args()

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    sr = load_sr(VLA_SR)
    print(f"SR rows (excluded dropped): {len(sr)}")

    # Per-CSV cohort sizes after intersecting with SR
    bridge_in_sr = {m for m in csv_models(VLA_BRIDGE) if m in sr}
    droid_in_sr  = {m for m in csv_models(VLA_DROID)  if m in sr}
    print(f"Bridge CSV ∩ SR: n={len(bridge_in_sr)}")
    print(f"DROID  CSV ∩ SR: n={len(droid_in_sr)}")

    model_filter = None
    if args.filter_to_intersection:
        model_filter = bridge_in_sr & droid_in_sr
        dropped = (bridge_in_sr | droid_in_sr) - model_filter
        print(f"Filter-to-intersection ON: keeping n={len(model_filter)} "
              f"models present in both CSVs.")
        if dropped:
            print(f"  Dropped (in only one CSV): {sorted(dropped)}")

    bridge = trace(VLA_BRIDGE, VARIANT, HORIZONS_BRIDGE, sr, model_filter)
    droid  = trace(VLA_DROID,  VARIANT, HORIZONS_DROID,  sr, model_filter)
    print(f"BridgeDataV2 r-vs-h: {[(h, round(r, 3), n) for h, r, n in bridge]}")
    print(f"DROID    r-vs-h: {[(h, round(r, 3), n) for h, r, n in droid]}")

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 5.5))

    # Reference lines (all observed r values are positive, so we don't draw
    # the -|r| threshold and we crop the y-axis at 0).
    ax.axhline(SIG_THRESH_ONE_TAILED, color="#aaaaaa", linewidth=0.6,
               linestyle=":", alpha=0.7, zorder=1)
    ax.axhline(0.50, color="#00838F", linewidth=0.9, linestyle="--",
               alpha=0.7, zorder=1)

    # ±4% multiplicative jitter on log-x so co-located markers don't fully
    # overlap when the two lines cross at h=15..40.
    h_b_arr = np.array([p[0] for p in bridge], dtype=float)
    r_b = np.array([p[1] for p in bridge])
    h_d_arr = np.array([p[0] for p in droid], dtype=float)
    r_d = np.array([p[1] for p in droid])
    jb = h_b_arr * (1.0 - 0.04)
    jd = h_d_arr * (1.0 + 0.04)
    h_b = jb  # used by axes block below
    h_d = jd

    # BridgeDataV2: dashed line + hollow green circles
    ax.plot(jb, r_b, linestyle="--", marker="o", color=GOOGLE_GREEN,
            linewidth=2.2, markersize=10, markerfacecolor="white",
            markeredgecolor=GOOGLE_GREEN, markeredgewidth=2.2,
            zorder=4, label="BridgeDataV2 (WidowX)")
    # DROID: solid line + filled blue squares
    ax.plot(jd, r_d, linestyle="-", marker="s", color=GOOGLE_BLUE,
            linewidth=2.2, markersize=9, markerfacecolor=GOOGLE_BLUE,
            markeredgecolor="black", markeredgewidth=0.5,
            zorder=4, label="DROID (Franka Panda)")

    # Per-marker r-value labels: only at a SPARSE subset of horizons
    # so labels don't pile up on each other on the log axis. We pick
    # 1, 7, 15, 40, 220 (covers the early ramp, the peak, mid-range,
    # and the long tail). Bridge always above, DROID always below to
    # guarantee vertical separation when both lines are at ~0.69.
    # Per-figure font scaling: this figure renders at ~0.20x source
    # width (9in vs 0.33 textwidth) -- the smallest scale in the
    # paper. Sizes bumped so axis labels match Fig 3's rendered size
    # at ~0.327x.
    LABEL_HORIZONS = {1, 7, 15, 40, 220}
    for h_orig, hx, rv in zip(h_b_arr.astype(int), jb, r_b):
        if int(h_orig) not in LABEL_HORIZONS:
            continue
        ax.annotate(format_r(rv, decimals=2, signed=False), (hx, rv),
                    textcoords="offset points",
                    xytext=(0, 22), ha="center",
                    fontsize=24, color=GOOGLE_GREEN, fontweight="bold",
                    zorder=5)
    for h_orig, hx, rv in zip(h_d_arr.astype(int), jd, r_d):
        if int(h_orig) not in LABEL_HORIZONS:
            continue
        ax.annotate(format_r(rv, decimals=2, signed=False), (hx, rv),
                    textcoords="offset points",
                    xytext=(0, -25), ha="center",
                    fontsize=24, color=GOOGLE_BLUE, fontweight="bold",
                    zorder=5)

    # Axes
    ax.set_xscale("log")
    all_h = sorted(set(HORIZONS_BRIDGE + HORIZONS_DROID))
    ax.set_xticks(all_h)
    # Sparsify x-tick LABELS too (same key horizons): the four right-
    # side ticks 75/110/150/220 cram together on the log axis, and
    # rotating them just made them tilt over each other. Show every
    # tick MARK, but only label 1/7/15/40/220 so the axis reads
    # cleanly. The other tick marks remain visible as guides.
    tick_labels = [str(h) if h in LABEL_HORIZONS else "" for h in all_h]
    ax.set_xticklabels(tick_labels, fontsize=25)
    ax.set_xlim(min(all_h) * 0.7, max(all_h) * 1.3)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Horizon $h$", fontsize=31)
    ax.set_ylabel(r"Pearson $r$", fontsize=31)
    ax.tick_params(axis="both", labelsize=25)
    # Horizontal gridlines only; vertical dashed lines drawn explicitly at the
    # tick positions {1, 3, 7, 15, ...} to avoid the dense log-minor grid.
    ax.grid(True, axis="y", alpha=0.25, which="major")
    ax.grid(False, axis="x")
    for h in all_h:
        ax.axvline(h, color="#cccccc", linewidth=0.5,
                   linestyle="--", alpha=0.55, zorder=0)
    ax.legend(loc="lower right", fontsize=20, framealpha=0.92)

    fig.tight_layout()
    out_pdf = os.path.join(FIG_DIR, "horizon_vla_llmout_vt.pdf")
    out_png = os.path.join(FIG_DIR, "horizon_vla_llmout_vt_preview.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n  PDF → {out_pdf}")
    print(f"  PNG → {out_png}")


if __name__ == "__main__":
    main()
