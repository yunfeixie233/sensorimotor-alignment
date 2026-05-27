"""
Frozen-VLM Hypothesis Confirmed: each VLA model vs its raw VLM backbone.

Layout (2x3 grid):
  Row 0: Qwen2.5-VL-3B vs GR00T-Bridge, GR00T-Bridge-RT-1, OFT-Bridge-RT-1
  Row 1: Qwen3-VL-4B vs Qwen3-GR00T, Qwen3-OFT  |  Prismatic vs OpenVLA-7B

Two independent VLM families (Qwen, Prismatic) provide controlled before/after pairs.
Single y-axis per subfigure so absolute CKNNA values are directly comparable.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR    = SCRIPT_DIR
CSV_PATH = os.path.join(
    EXP_DIR, "cknna_data_exp2", "finetune", "dtw_sym2_cuda_nowin",
    "results_groups_dtw_sym2_cuda_nowin.csv",
)
META_CSV = os.path.join(SCRIPT_DIR, "..", "..", "cknna_data", "performance",
                        "model_metadata.csv")
OUT_DIR = os.path.join(EXP_DIR, "Frozen-VLM")


def _load_display_names(csv_path):
    names = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["model_key"] not in names:
                names[row["model_key"]] = row["figure_label"]
    return names


_DISPLAY_NAMES = _load_display_names(META_CSV)


def display_name(key):
    return _DISPLAY_NAMES.get(key, key)


# (raw_model_key, ft_model_key)
PAIRS = [
    ("qwen25vl-3b-raw", "Qwen-GR00T-Bridge"),
    ("qwen25vl-3b-raw", "Qwen-GR00T-Bridge-RT-1"),
    ("qwen25vl-3b-raw", "Qwen-OFT-Bridge-RT-1"),
    ("qwen3vl-4b-raw",  "Qwen3VL-GR00T-Bridge-RT-1"),
    ("qwen3vl-4b-raw",  "Qwen3VL-OFT-Bridge-RT-1"),
    ("prismatic-raw",   "openvla-7b-bridge"),
]

COLOR_RAW = "#2196F3"
COLOR_FT  = "#E53935"


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def get_curve(rows, model, variant="imgtext"):
    data = {}
    for r in rows:
        if r["model"] == model and r["feats_A_variant"] == variant:
            data[int(r["horizon"])] = float(r["cknna_k10"])
    hs = sorted(data)
    return hs, [data[h] for h in hs]


def plot_pair(ax, rows, raw_model, ft_model):
    hs_raw, vals_raw = get_curve(rows, raw_model)
    hs_ft, vals_ft   = get_curve(rows, ft_model)

    ax.plot(hs_raw, vals_raw, "o-", color=COLOR_RAW, linewidth=2.2,
            markersize=5, label=display_name(raw_model), zorder=3)
    ax.plot(hs_ft, vals_ft, "s-", color=COLOR_FT, linewidth=2.2,
            markersize=5, label=display_name(ft_model), zorder=3)

    ax.set_ylabel("CKNNA (imgtext, k=10)", fontsize=9)
    ax.set_xlabel("Horizon h", fontsize=9)

    all_vals = vals_raw + vals_ft
    y_lo = min(all_vals) - 0.02
    y_hi = max(all_vals) + 0.02
    y_lo = max(0, round(y_lo * 20) / 20)
    y_hi = round(y_hi * 20) / 20
    ax.set_ylim(y_lo, y_hi)

    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.2)

    peak_idx_ft = int(np.argmax(vals_ft))
    peak_h_ft = hs_ft[peak_idx_ft]
    peak_v_ft = vals_ft[peak_idx_ft]
    ax.axvline(peak_h_ft, color=COLOR_FT, linestyle="--", alpha=0.4, linewidth=1.5)
    ax.annotate(
        f"peak h={peak_h_ft}",
        xy=(peak_h_ft, peak_v_ft),
        xytext=(peak_h_ft + 4, peak_v_ft - 0.02),
        fontsize=10, fontweight="bold", color=COLOR_FT, alpha=0.85,
        arrowprops=dict(arrowstyle="-", color=COLOR_FT, alpha=0.4, lw=1),
    )


def plot_absolute_summary(ax, rows):
    """All curves on one absolute y-axis.
    Blue group = raw/pretrained VLMs (raw Qwen, raw Prismatic).
    Red group  = VLA fine-tuned from the same base.
    """
    FROZEN_KEYS = [
        ("qwen25vl-3b-raw",   display_name("qwen25vl-3b-raw")),
        ("qwen3vl-4b-raw",    display_name("qwen3vl-4b-raw")),
        ("prismatic-raw",     display_name("prismatic-raw")),
    ]
    all_vals = []
    frozen_data = {}
    for model, label in FROZEN_KEYS:
        hs, vals = get_curve(rows, model)
        ax.plot(hs, vals, "o-", linewidth=1.8, markersize=4,
                alpha=0.8, color=COLOR_RAW, label=label)
        all_vals.extend(vals)
        for h, v in zip(hs, vals):
            frozen_data.setdefault(h, []).append(v)

    ft_keys = [(ft, display_name(ft)) for _, ft in PAIRS]
    ft_data = {}
    for model, label in ft_keys:
        hs_ft, vals_ft = get_curve(rows, model)
        ax.plot(hs_ft, vals_ft, "s-", linewidth=1.3, markersize=3,
                alpha=0.5, color=COLOR_FT, label=label)
        all_vals.extend(vals_ft)
        for h, v in zip(hs_ft, vals_ft):
            ft_data.setdefault(h, []).append(v)

    hs_common = sorted(set(frozen_data) & set(ft_data))
    mean_frozen = [np.mean(frozen_data[h]) for h in hs_common]
    mean_ft     = [np.mean(ft_data[h]) for h in hs_common]
    n_frozen = len(FROZEN_KEYS)
    n_ft = len(ft_keys)
    ax.plot(hs_common, mean_frozen, "o-", color="#0D47A1", linewidth=3.5,
            markersize=7, label=f"Mean raw VLM (n={n_frozen})", zorder=5)
    ax.plot(hs_common, mean_ft, "s-", color="#B71C1C", linewidth=3.5,
            markersize=7, label=f"Mean VLA fine-tuned (n={n_ft})", zorder=5)

    y_lo = max(0.0, min(all_vals) - 0.02)
    y_hi = max(all_vals) + 0.025
    ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel("Horizon h", fontsize=10)
    ax.set_ylabel("CKNNA (imgtext, k=10)", fontsize=10)
    ax.set_title(
        "All models\nshared absolute y-axis",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=7.5, loc="lower right", ncol=1)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=9)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_csv(CSV_PATH)

    # ------------------------------------------------------------------ #
    # Figure 1: 2x3 grid of pair comparisons (no summary panel here)
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 3, figsize=(18, 10),
                             gridspec_kw={"hspace": 0.42, "wspace": 0.32})
    pair_axes = [axes[r][c] for r, c in
                 [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]]

    for ax, (raw_model, ft_model) in zip(pair_axes, PAIRS):
        ax.set_title(f"{display_name(ft_model)}  vs  {display_name(raw_model)}",
                     fontsize=10, fontweight="bold")
        plot_pair(ax, rows, raw_model, ft_model)

    fig.suptitle(
        "Frozen-VLM Hypothesis: Raw Pretrained VLM vs VLA Fine-tuned\n"
        "Each panel: same base architecture, only E2E fine-tuning differs.  Two VLM families: Qwen + Prismatic.",
        fontsize=13, fontweight="bold", y=1.01,
    )
    path = os.path.join(OUT_DIR, "frozen_vlm_hypothesis_confirmed.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # ------------------------------------------------------------------ #
    # Figure 2: standalone absolute summary
    # ------------------------------------------------------------------ #
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    plot_absolute_summary(ax2, rows)
    ax2.set_title(
        "Frozen-VLM Hypothesis: All Models on Shared Absolute Y-axis\n"
        "Blue = frozen/raw backbone  |  Red = VLA fine-tuned",
        fontsize=12, fontweight="bold",
    )
    fig2.tight_layout()
    p2 = os.path.join(OUT_DIR, "frozen_vlm_summary.png")
    fig2.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {p2}")



if __name__ == "__main__":
    main()
