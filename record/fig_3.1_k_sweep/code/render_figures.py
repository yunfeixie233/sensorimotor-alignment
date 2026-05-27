#!/usr/bin/env python3
"""Render appendix figures for the DROID k-sweep.

Inputs:
  results_k{2,5,10,15,20}.csv  -- per-(model, horizon) CKNNA at each k
  summary_r_p.csv              -- Pearson r and one-tailed p per (k, h)

Outputs (vector PDF):
  fig_cknna_r_vs_k_droid.pdf       -- one line per horizon, x-axis = k
  fig_cknna_vs_sr_droid_k{K}.pdf   -- 6-row scatter per k (h = 1..40)
"""
from __future__ import annotations

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# Resolve paths relative to this script's location so the release works from any clone path.
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_DATA = _HERE.parent / "data"
_FIG_DIR = _HERE.parent
sys.path.insert(0, str(_HERE.parent.parent / "shared"))
from google_palette import GOOGLE_CMAP  # noqa: E402
from scatter_labels import add_labels, add_fit_line  # noqa: E402


HERE = str(_DATA)
OUT_DIR = str(_FIG_DIR)

KS = [2, 5, 10, 15, 20]
HORIZONS = [1, 3, 7, 15, 25, 40]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_summary():
    rows = list(csv.DictReader(open(os.path.join(HERE, "summary_r_p.csv"))))
    table = {(int(r["k"]), int(r["h"])): {
        "r": float(r["r"]),
        "p": float(r["p_one"]),
        "n": int(r["n"]),
    } for r in rows}
    return table


def load_per_k(k):
    rows = list(csv.DictReader(open(os.path.join(HERE, f"results_k{k}.csv"))))
    return [(r["model"], int(r["horizon"]),
             float(r["cknna"]), float(r["success_rate"])) for r in rows]


# ---------------------------------------------------------------------------
# Figure 1: r vs k, one line per horizon
# ---------------------------------------------------------------------------
def render_r_vs_k(table, focus_h=7):
    """Single-line plot of r vs k at the headline horizon."""
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    rs = [table[(k, focus_h)]["r"] for k in KS]
    color = GOOGLE_CMAP["red"](0.85)
    ax.plot(KS, rs, "-o", color=color, linewidth=2.0, markersize=8,
            markerfacecolor=color, markeredgecolor="black",
            markeredgewidth=0.6, zorder=3)
    ax.axvline(10, color="#c0392b", linestyle="--", linewidth=1.2,
               alpha=0.85, zorder=1)
    ax.set_xticks(KS)
    ax.set_xlabel(r"Neighbourhood size $k$", fontsize=12)
    ax.set_ylabel(r"Pearson $r$ ($SiMA$ vs SimplerEnv success rate)",
                  fontsize=11)
    ax.set_ylim(0.45, 0.80)
    ax.set_xlim(KS[0] - 1.2, KS[-1] + 1.2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_cknna_r_vs_k_droid.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}  (h={focus_h} only)")


# ---------------------------------------------------------------------------
# Figure 2..6: per-k 6-row scatter, one panel per horizon
# ---------------------------------------------------------------------------
def render_scatter_per_k(k, rows):
    by_h = {h: [] for h in HORIZONS}
    for m, h, c, sr in rows:
        if h in by_h:
            by_h[h].append((m, c, sr))

    n_rows = len(HORIZONS)
    fig, axes = plt.subplots(n_rows, 1, figsize=(7.0, 2.6 * n_rows))
    if n_rows == 1:
        axes = [axes]
    for ax, h in zip(axes, HORIZONS):
        recs = by_h[h]
        xs = np.array([r[1] for r in recs])
        ys = np.array([r[2] for r in recs])
        keys = [r[0] for r in recs]
        cmap = GOOGLE_CMAP["red"]
        norm = (xs - xs.min()) / max(xs.max() - xs.min(), 1e-9)
        ax.scatter(xs, ys, s=80, c=cmap(0.30 + 0.65 * norm),
                   edgecolors="black", linewidths=0.5, zorder=3)
        add_fit_line(ax, xs, ys)
        r, p2 = pearsonr(xs, ys)
        p1 = p2 / 2 if r > 0 else 1 - p2 / 2
        stats = f"$r={r:+.3f}$\n$p={p1:.3f}$\n$n={len(xs)}$"
        ax.text(0.04, 0.96, stats, transform=ax.transAxes, fontsize=10,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                          edgecolor="#888888", alpha=0.92))
        ax.text(0.96, 0.04, f"$h{{=}}{h}$",
                transform=ax.transAxes, fontsize=11, fontweight="bold",
                va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#cccccc", alpha=0.85))
        ax.set_xlabel("Sensorimotor Alignment (SiMA)", fontsize=10)
        ax.set_ylabel("SimplerEnv success rate (\\%)", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(True, alpha=0.25)
        if len(xs) >= 2:
            x_pad = 0.10 * max(xs.max() - xs.min(), 1e-9)
            y_pad = 0.10 * max(ys.max() - ys.min(), 1e-9)
            ax.set_xlim(xs.min() - x_pad, xs.max() + x_pad)
            ax.set_ylim(ys.min() - y_pad, ys.max() + y_pad)
        add_labels(ax, xs, ys, keys, fontsize=7)
    fig.suptitle(rf"$k{{=}}{k}$: \PA vs SimplerEnv success rate (LLM-out V+T)",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"fig_cknna_vs_sr_droid_k{k}.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    table = load_summary()
    print("Rendering r vs k summary (h=7 only)...")
    render_r_vs_k(table, focus_h=7)


if __name__ == "__main__":
    main()
