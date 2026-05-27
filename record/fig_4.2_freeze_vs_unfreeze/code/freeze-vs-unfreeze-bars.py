
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
"""§4.2 figure: two-panel horizontal bar chart comparing the unfrozen vs.
frozen Qwen2.5-VL-3B VLAs trained under the VLM4VLA recipe on BridgeDataV2.

Top panel: SiMA (S^2) at LLM-out vision-text feature, h=7, DROID N=6000.
Bottom panel: SimplerBridge success rate (4 tasks x 24 episodes, exec_step=2).

Style mimics Cambrian-1 (arXiv 2504.13181) Fig. 2: horizontal bars in a
light blue/teal palette, bar values printed at the end, no decorative
gridlines or 3D effects. Two bars per panel: "Unfrozen ViT" (longer)
and "Frozen ViT" (shorter).

Sources (numeric values below are extracted from these CSVs and
hardcoded inline in the DATA dict for this 2-bar figure):
- SiMA: record/fig_4.2_freeze_vs_unfreeze/data/cknna_vla_droid_complete.csv
        rows: vlm4vla-qwen25vl3b-bridge-step10k     (Unfrozen)
              vlm4vla-qwen25vl3b-bridge-freezevis   (Frozen)
        filter: feats_A_variant=imgtext, horizon=7
- SR  : record/fig_4.2_freeze_vs_unfreeze/data/widowx_sr.csv

Outputs: figures/freeze-vs-unfreeze-bars.pdf  (vector, used in paper)
         figures/freeze-vs-unfreeze-bars.png  (preview)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "mathtext.fontset": "cm",
    "mathtext.default": "regular",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FIG_DIR = str(_FIG_DIR)

# ---- Numbers (final, post fixed-preprocessing pipeline) ----
DATA = {
    "Unfrozen ViT fine-tuning": {"sima": 0.0245, "sr": 50.00},
    "Frozen ViT fine-tuning":   {"sima": 0.0144, "sr": 30.21},
}
ROW_ORDER = ["Unfrozen ViT fine-tuning", "Frozen ViT fine-tuning"]

# ---- Colors: deeper teal for unfrozen, muted gray-teal for frozen ----
COLOR_UNFROZEN = "#5B9BD5"   # light blue
COLOR_FROZEN   = "#A6C8E0"   # paler blue
COLOR_BAR_EDGE = "#1F3A5F"   # dark navy

ROW_COLORS = {
    "Unfrozen ViT fine-tuning": COLOR_UNFROZEN,
    "Frozen ViT fine-tuning":   COLOR_FROZEN,
}


def panel(ax, key, value_format, header, x_max, value_offset_frac=0.012):
    """Draw a one-axis horizontal bar panel in a Cambrian-1 Fig 2 style:
    metric name as a small top-of-panel header, no visible x-axis ticks,
    bar values printed just past each bar tip."""
    rows = ROW_ORDER
    values = [DATA[r][key] for r in rows]
    y = list(range(len(rows)))[::-1]  # top-down

    for yi, name, v in zip(y, rows, values):
        ax.barh(
            yi, v,
            color=ROW_COLORS[name],
            edgecolor=COLOR_BAR_EDGE,
            linewidth=0.7,
            height=0.55,
            zorder=2,
        )
        ax.text(
            v + value_offset_frac * x_max, yi,
            value_format.format(v),
            va="center", ha="left",
            fontsize=11, color="black",
            zorder=3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(rows, fontsize=11)
    ax.set_xlim(0, x_max)
    ax.set_xticks([])
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_axisbelow(True)

    # Cambrian-style top header: small metric label at top-left of plot area.
    ax.text(
        0.0, 1.06, header,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=11, fontweight="bold", color="#1F3A5F",
    )


def main():
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(5.4, 2.6), sharey=False)

    # Top panel: S^2
    panel(
        ax_top, key="sima",
        value_format="{:.4f}",
        header=r"Sensorimotor Alignment ($\mathrm{S}^2$)",
        x_max=0.030,
    )

    # Bottom panel: Success rate (%)
    panel(
        ax_bot, key="sr",
        value_format="{:.2f}",
        header="SimplerEnv success rate (%)",
        x_max=60.0,
    )

    fig.tight_layout(h_pad=1.6)
    out_pdf = os.path.join(FIG_DIR, "freeze-vs-unfreeze-bars.pdf")
    out_png = os.path.join(FIG_DIR, "freeze-vs-unfreeze-bars.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
