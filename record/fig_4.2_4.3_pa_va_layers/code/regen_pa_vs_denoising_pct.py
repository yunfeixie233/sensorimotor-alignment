
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
"""§4.2 figure: WAM \\PA at the video-stream final layer normalization vs.
percent of native denoising schedule, three World-Action Models, one line each.

Replaces tables/final-norm-vs-denoising-step.tex.

Style mirrors figures/horizon_vla_llmout_vt.pdf (Figure 7):
    - distinct per-series line/marker styles (Motus solid + filled red
      circles, LingBot-VA dashed + hollow blue squares, Vidar dotted +
      filled green triangles),
    - per-marker value labels in colored text matching each line,
    - explicit dashed vertical guides at every tick.

Source data (hard-coded from the table):
  /lambda/nfs/vla/cknna_project/analysis/out/cknna_denoise_3model_50x10.csv
50 RoboTwin tasks x 10 episodes each. Columns are 0/25/50/75/100% of native
schedule, where 0% is pure Gaussian noise and 100% is the fully-denoised
latent. Each model's native step indices differ (Motus 10-step linear Euler,
LingBot-VA 25+pad DDIM, Vidar 20-step UniPC), so we plot in % of schedule.

Outputs:
  figures/pa_vs_denoising_pct.pdf  (vector, used in paper)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from google_palette import GOOGLE_RED, GOOGLE_BLUE, GOOGLE_GREEN


FIG_DIR = str(_FIG_DIR)

# x-axis = % of native denoising schedule
PCT = np.array([0, 25, 50, 75, 100], dtype=float)

# y-axis = S^2 at V-Norm (final layer normalization of the vision stream).
SIMA = {
    "Motus":      np.array([0.425, 0.421, 0.421, 0.418, 0.402]),
    "LingBot-VA": np.array([0.425, 0.429, 0.431, 0.439, 0.451]),
    "Vidar":      np.array([0.487, 0.491, 0.491, 0.481, 0.476]),
}

COLORS = {
    "Motus":      GOOGLE_RED,
    "LingBot-VA": GOOGLE_BLUE,
    "Vidar":      GOOGLE_GREEN,
}
STYLES = {
    "Motus":      dict(linestyle="-",  marker="o", filled=True),
    "LingBot-VA": dict(linestyle="--", marker="s", filled=False),
    "Vidar":      dict(linestyle=":",  marker="^", filled=True),
}


def main():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"
    ]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 5.5))

    # Small additive jitter on x so co-located markers stay distinguishable.
    JITTERS = {"Motus": -1.5, "LingBot-VA": 0.0, "Vidar": +1.5}

    for name, ys in SIMA.items():
        c = COLORS[name]
        st = STYLES[name]
        xs = PCT + JITTERS[name]
        ax.plot(
            xs, ys,
            linestyle=st["linestyle"], color=c,
            linewidth=2.2,
            marker=st["marker"], markersize=11,
            markerfacecolor=(c if st["filled"] else "white"),
            markeredgecolor=c, markeredgewidth=2.0,
            zorder=4, label=name,
        )

    # Per-marker value labels in matching colour, fanned across three
    # corners of the marker so the three series never stack vertically:
    #   highest -> directly above
    #   middle  -> to the right
    #   lowest  -> directly below
    # Same fan-out as Fig 11(b) -- the three lines run almost parallel
    # across the schedule, so without lateral offset the labels would
    # collide at every checkpoint.
    series_order = ["Motus", "LingBot-VA", "Vidar"]
    series_xs = {n: PCT + JITTERS[n] for n in series_order}
    POS = {0: dict(dx=0,  dy=18,  ha="center", va="bottom"),
           1: dict(dx=14, dy=0,   ha="left",   va="center"),
           2: dict(dx=0,  dy=-18, ha="center", va="top")}
    # Rightmost-column flips: at 100% the rank-1 label's +14pt right
    # offset would clip the panel, and the rank-2 label's -18pt down
    # offset crosses into the bottom-right legend box. Flip rank-1
    # to the LEFT of the marker, and rank-2 ABOVE the marker (it has
    # ~0.05 of headroom up to the next series in this dataset).
    POS_FLIPPED_LEFT = dict(dx=-14, dy=0,  ha="right",  va="center")
    POS_FLIPPED_UP   = dict(dx=0,   dy=18, ha="center", va="bottom")
    rightmost_x = PCT.max()
    for ci, _ in enumerate(PCT):
        triples = sorted(series_order, key=lambda n: -SIMA[n][ci])
        for rank, name in enumerate(triples):
            y_val = SIMA[name][ci]
            x_val = series_xs[name][ci]
            p = POS[rank]
            if PCT[ci] == rightmost_x:
                if rank == 1:
                    p = POS_FLIPPED_LEFT
                elif rank == 2:
                    p = POS_FLIPPED_UP
            ax.annotate(
                f"{y_val:.3f}", (x_val, y_val),
                textcoords="offset points",
                xytext=(p["dx"], p["dy"]), ha=p["ha"], va=p["va"],
                fontsize=15, color=COLORS[name], fontweight="bold",
                zorder=5,
            )

    # Per-figure font scaling: each Fig-11 sub-panel renders at ~0.30x
    # source width (9in vs 0.49 textwidth). Sizes bumped slightly so
    # axis labels match Fig 3's rendered size at ~0.327x.
    ax.set_xticks(PCT)
    ax.set_xticklabels([f"{int(p)}%" for p in PCT], fontsize=17)
    ax.set_xlim(-8, 108)
    ax.set_ylim(0.30, 0.60)

    ax.grid(True, axis="y", alpha=0.25, which="major")
    ax.grid(False, axis="x")
    for p in PCT:
        ax.axvline(p, color="#cccccc", linewidth=0.5,
                   linestyle="--", alpha=0.55, zorder=0)

    ax.set_xlabel("Denoising schedule (% of native steps)", fontsize=22)
    ax.set_ylabel(r"Sensorimotor Alignment ($\mathrm{S}^2$)", fontsize=22)
    ax.tick_params(axis="both", labelsize=17)
    ax.legend(loc="lower right", fontsize=17, framealpha=0.92)


    fig.tight_layout()
    out_pdf = os.path.join(FIG_DIR, "pa_vs_denoising_pct.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  PDF -> {out_pdf}")
    print()
    for name in series_order:
        ys = SIMA[name]
        drift = float(ys.max() - ys.min())
        print(f"  {name:<11}  min={ys.min():.3f}  max={ys.max():.3f}  drift={drift:.3f}")


if __name__ == "__main__":
    main()
