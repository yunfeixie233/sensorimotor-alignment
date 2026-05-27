
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
"""§4.4 figure: WAM \\PA across video-backbone block depths, three World-Action
Models, one line each.

Replaces the in-body version of tables/layerwise-vs-denoising-step.tex (the
full layer x denoising-step table moves to the appendix as
app:layerwise-vs-denoising-step).

For each (model, block) we average \\PA across the 5 denoising checkpoints
(0, 25, 50, 75, 100% of native schedule) so that one number per block is
plotted on the y-axis.

Style mirrors figures/horizon_vla_llmout_vt.pdf (Figure 7):
    - distinct per-series line/marker styles
      (Motus solid + filled red circles, LingBot-VA dashed + hollow blue
      squares, Vidar dotted + filled green triangles),
    - per-marker value labels in colored text matching each line,
    - explicit dashed vertical guides at every tick,
    - peak-block callout via dashed vertical line.

Source data (hard-coded from the table; raw CSV at
  /lambda/nfs/vla/cknna_project/analysis/out/cknna_denoise_3model_50x10.csv).

Outputs:
  figures/pa_layerwise.pdf  (vector, used in paper)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from google_palette import GOOGLE_RED, GOOGLE_BLUE, GOOGLE_GREEN


FIG_DIR = str(_FIG_DIR)

# x-axis = video-backbone block index. The 30-block WAN-2.2-5B / Vidar
# backbones share these five hook positions for cross-model comparison.
BLOCKS = np.array([0, 7, 14, 21, 29])

# y-axis = S^2 at each (model, block), averaged across the 5 denoising
# checkpoints in tables/layerwise-vs-denoising-step.tex.
SIMA = {
    "Motus": np.array([
        np.mean([0.044, 0.203, 0.307, 0.321, 0.324]),
        np.mean([0.342, 0.388, 0.399, 0.390, 0.378]),
        np.mean([0.479, 0.492, 0.482, 0.479, 0.471]),
        np.mean([0.454, 0.448, 0.447, 0.436, 0.424]),
        np.mean([0.420, 0.424, 0.419, 0.421, 0.405]),
    ]),
    "LingBot-VA": np.array([
        np.mean([0.142, 0.160, 0.238, 0.326, 0.384]),
        np.mean([0.404, 0.399, 0.431, 0.449, 0.455]),
        np.mean([0.498, 0.504, 0.514, 0.514, 0.513]),
        np.mean([0.422, 0.435, 0.428, 0.445, 0.449]),
        np.mean([0.423, 0.440, 0.441, 0.447, 0.448]),
    ]),
    "Vidar": np.array([
        np.mean([0.483, 0.477, 0.459, 0.429, 0.422]),
        np.mean([0.467, 0.484, 0.485, 0.479, 0.469]),
        np.mean([0.504, 0.555, 0.531, 0.518, 0.473]),
        np.mean([0.508, 0.510, 0.500, 0.488, 0.475]),
        np.mean([0.487, 0.490, 0.489, 0.481, 0.474]),
    ]),
}

COLORS = {
    "Motus":      GOOGLE_RED,
    "LingBot-VA": GOOGLE_BLUE,
    "Vidar":      GOOGLE_GREEN,
}
# Distinct (linestyle, marker, fill) per series so the three lines remain
# legible when they cross. Matches the Figure 7 disambiguation recipe.
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

    # ±2% additive jitter on x so co-located markers at block 14 don't fully
    # merge when the three lines peak together.
    JITTERS = {"Motus": -0.35, "LingBot-VA": 0.0, "Vidar": +0.35}

    for name, ys in SIMA.items():
        c = COLORS[name]
        st = STYLES[name]
        xs = BLOCKS + JITTERS[name]
        ax.plot(
            xs, ys,
            linestyle=st["linestyle"], color=c,
            linewidth=2.2,
            marker=st["marker"], markersize=11,
            markerfacecolor=(c if st["filled"] else "white"),
            markeredgecolor=c, markeredgewidth=2.0,
            zorder=4, label=name,
        )

    # Per-marker value labels in matching colour. At each block we sort
    # the three series high->low and assign them DIFFERENT corners
    # around the cluster so labels don't stack on top of each other:
    #   highest -> directly above the marker
    #   middle  -> to the right of the marker
    #   lowest  -> directly below the marker
    # Without this fan-out, the three peaks at Block 14 (0.48/0.51/0.52)
    # collided into a single illegible blob.
    series_order = ["Motus", "LingBot-VA", "Vidar"]
    series_xs = {n: BLOCKS + JITTERS[n] for n in series_order}
    POS = {0: dict(dx=0,  dy=20,  ha="center", va="bottom"),
           1: dict(dx=18, dy=0,   ha="left",   va="center"),
           2: dict(dx=0,  dy=-20, ha="center", va="top")}
    # Flip rank-2 (lowest) labels ABOVE the marker when the marker
    # sits within ~12% of the panel bottom -- otherwise the 20pt-below
    # offset puts the label on top of the x-axis (block 0 Motus at
    # 0.240 with y_min 0.20 needs 0.05 of headroom for the 16pt label).
    # Flip rank-1 (middle) labels to the LEFT of the marker when the
    # marker is the rightmost x position, otherwise the +18pt right
    # offset clips off the panel (block 29 LingBot-VA 0.440 case).
    Y_MIN = 0.20
    Y_MAX = 0.62
    FLIP_THRESHOLD = Y_MIN + 0.12 * (Y_MAX - Y_MIN)  # ~0.250
    POS_FLIPPED_DOWN = dict(dx=0,   dy=20, ha="center", va="bottom")
    POS_FLIPPED_LEFT = dict(dx=-18, dy=0,  ha="right",  va="center")
    rightmost_block = BLOCKS.max()
    for bi, b in enumerate(BLOCKS):
        triples = sorted(series_order, key=lambda n: -SIMA[n][bi])
        for rank, name in enumerate(triples):
            y_val = SIMA[name][bi]
            x_val = series_xs[name][bi]
            p = POS[rank]
            if rank == 2 and y_val < FLIP_THRESHOLD:
                p = POS_FLIPPED_DOWN
            if rank == 1 and b == rightmost_block:
                p = POS_FLIPPED_LEFT
            ax.annotate(
                f"{y_val:.3f}", (x_val, y_val),
                textcoords="offset points",
                xytext=(p["dx"], p["dy"]), ha=p["ha"], va=p["va"],
                fontsize=16, color=COLORS[name], fontweight="bold",
                zorder=5,
            )

    # Reference line + callout at the peak block (14).
    peak_x = 14
    ax.axvline(peak_x, color="#888888", linewidth=0.8,
               linestyle="--", alpha=0.6, zorder=1)
    ax.text(
        peak_x, 0.595, r"peak $\mathrm{S}^2$",
        color="#444444", fontsize=16, fontstyle="italic",
        ha="center", va="center",
    )


    # Per-figure font scaling: each Fig-11 sub-panel renders at ~0.30x
    # source width (9in vs 0.49 textwidth). Sizes bumped slightly so
    # axis labels match Fig 3's rendered size at ~0.327x.
    ax.set_xticks(BLOCKS)
    ax.set_xticklabels([str(b) for b in BLOCKS], fontsize=17)
    ax.set_xlim(-2, 31)
    ax.set_ylim(0.20, 0.62)

    # Horizontal gridlines only; explicit dashed vertical guides at the tick
    # positions to mirror the Figure 7 layout.
    ax.grid(True, axis="y", alpha=0.25, which="major")
    ax.grid(False, axis="x")
    for b in BLOCKS:
        ax.axvline(b, color="#cccccc", linewidth=0.5,
                   linestyle="--", alpha=0.55, zorder=0)

    ax.set_xlabel("Block index", fontsize=22)
    ax.set_ylabel(r"Sensorimotor Alignment ($\mathrm{S}^2$)", fontsize=22)
    ax.tick_params(axis="both", labelsize=17)
    ax.legend(loc="lower right", fontsize=17, framealpha=0.92)

    fig.tight_layout()
    out_pdf = os.path.join(FIG_DIR, "pa_layerwise.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  PDF -> {out_pdf}")
    print()
    for name in series_order:
        ys = SIMA[name]
        peak_idx = int(ys.argmax())
        print(f"  {name:<11}  peak at Block {BLOCKS[peak_idx]} (S^2={ys[peak_idx]:.3f})")


if __name__ == "__main__":
    main()
