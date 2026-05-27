"""
plot_composite_proprio_seq.py

4 rows (horizons) x N columns (Flatten, DTW v1, DTW v2, DTW v3, DTW v4, DTW v5),
using only the img+txt feats_A variant.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.environ.get("RECORD_DIR", "/home/ubuntu/verl/starVLA/cknna/record")

CSV_PATHS = [
    os.path.join(RECORD_DIR, "finetune", "flatten",        "cknna_50k_fix_oricknna_results.csv"),
    os.path.join(RECORD_DIR, "finetune", "dtw_v1",         "cknna_50k_dtw_fix_oricknna_results.csv"),
    os.path.join(RECORD_DIR, "finetune", "dtw_v2",         "cknna_50k_dtw_fix_v2_oricknna_results.csv"),
    os.path.join(RECORD_DIR, "finetune", "dtw_v3",         "cknna_50k_dtw_fix_v3_oricknna_results.csv"),
    os.path.join(RECORD_DIR, "finetune", "dtw_v4",         "cknna_50k_dtw_v4_oricknna_results.csv"),
    os.path.join(RECORD_DIR, "finetune", "dtw_v5",         "cknna_50k_dtw_v5_oricknna_results.csv"),
    os.path.join(RECORD_DIR, "finetune", "dtw_v6",         "cknna_50k_dtw_v6_oricknna_results.csv"),
    os.path.join(RECORD_DIR, "finetune", "dtw_v7",         "cknna_50k_dtw_v7_oricknna_results.csv"),
    os.path.join(RECORD_DIR, "finetune", "dtw_soft_cuda",  "cknna_50k_dtw_soft_cuda_oricknna_results.csv"),
]
COL_TITLES = [
    "Flatten (no_pad)",
    "DTW v1 (raw 7D, no window)",
    "DTW v2 (sin/cos + z-score + SC window)",
    "DTW v3 (sin/cos + z-score, no window)",
    "DTW v4 (SO(3) uniform w, SC window)",
    "DTW v5 (SO(3) equalized w, SC window)",
    "DTW v6 (SO(3) uniform w, no window)",
    "DTW v7 (SO(3) equalized w, no window)",
    "SoftDTW CUDA (SO(3), sym1, SC window)",
]

OUT_PATH = os.path.join(RECORD_DIR, "fig_composite_proprio_seq_imgtxt.png")

MODEL_META = {
    "Qwen-GR00T-Bridge":         {"sr": 71.4, "group": "StarVLA-Qwen2.5", "label": "Qwen2.5-GR00T-Br"},
    "Qwen-GR00T-Bridge-RT-1":    {"sr": 63.6, "group": "StarVLA-Qwen2.5", "label": "Qwen2.5-GR00T"},
    "Qwen-OFT-Bridge-RT-1":      {"sr": 44.2, "group": "StarVLA-Qwen2.5", "label": "Qwen2.5-OFT"},
    "Qwen3VL-GR00T-Bridge-RT-1": {"sr": 65.3, "group": "StarVLA-Qwen3",   "label": "Qwen3-GR00T"},
    "Qwen3VL-OFT-Bridge-RT-1":   {"sr": 42.7, "group": "StarVLA-Qwen3",   "label": "Qwen3-OFT"},
    "cogact-small-bridge":        {"sr": 51.0, "group": "CogACT",          "label": "CogACT-S"},
    "cogact-base-bridge":         {"sr": 51.3, "group": "CogACT",          "label": "CogACT-B"},
    "cogact-large-bridge":        {"sr": 58.3, "group": "CogACT",          "label": "CogACT-L"},
    "groot-n15-bridge":           {"sr": 36.5, "group": "GR00T",           "label": "GR00T-N1.5"},
    "groot-n16-bridge":           {"sr": 57.1, "group": "GR00T",           "label": "GR00T-N1.6"},
    "openvla-7b-bridge-ft-200k":  {"sr": 10.4, "group": "OpenVLA",         "label": "OpenVLA-ft"},
    "pi0-lerobot-bridge":         {"sr": 47.9, "group": "Pi0",             "label": "Pi0"},
    "spatialvla-sft-bridge":      {"sr": 42.7, "group": "SpatialVLA",      "label": "SpatialVLA"},
    "rt1x-bridge":                {"sr":  0.0, "group": "RT-1-X",          "label": "RT-1-X"},
    "octo-base-bridge":           {"sr": 20.3, "group": "Octo",            "label": "Octo"},
}

EXCLUDE = {"openvla-7b-bridge"}

GROUP_COLORS = {
    "StarVLA-Qwen2.5": "#1f77b4",
    "StarVLA-Qwen3":   "#aec7e8",
    "CogACT":          "#ff7f0e",
    "GR00T":           "#2ca02c",
    "OpenVLA":         "#d62728",
    "Pi0":             "#9467bd",
    "SpatialVLA":      "#8c564b",
    "RT-1-X":          "#7f7f7f",
    "Octo":            "#bcbd22",
}

GROUP_MARKERS = {
    "StarVLA-Qwen2.5": "o",
    "StarVLA-Qwen3":   "s",
    "CogACT":          "^",
    "GR00T":           "D",
    "OpenVLA":         "v",
    "Pi0":             "P",
    "SpatialVLA":      "X",
    "RT-1-X":          "*",
    "Octo":            "h",
}

SEQ_HORIZONS = [1, 3, 7, 15]
K = 10


def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def get_val(rows, model, fb_type, horizon):
    col = f"cknna_k{K}"
    for r in rows:
        if (r["model"] == model and
                r["feats_A_variant"] == "imgtext" and
                r["feats_B_type"] == fb_type and
                int(r["horizon"]) == horizon and
                col in r):
            return float(r[col])
    return None


def draw_panel(ax, rows, horizon, title=""):
    xs, ys = [], []
    for model_key, meta in MODEL_META.items():
        if model_key in EXCLUDE:
            continue
        val = get_val(rows, model_key, "proprio_seq", horizon)
        if val is None:
            continue
        c = GROUP_COLORS[meta["group"]]
        m = GROUP_MARKERS[meta["group"]]
        ax.scatter(val, meta["sr"], c=c, marker=m, s=80,
                   edgecolors="k", linewidths=0.4, zorder=3)
        ax.annotate(meta["label"], (val, meta["sr"]),
                    textcoords="offset points", xytext=(4, 2),
                    fontsize=5.5, color=c)
        xs.append(val)
        ys.append(meta["sr"])

    if len(xs) >= 3:
        rho, pval = spearmanr(xs, ys)
        p_str = f"{pval:.3f}" if pval >= 0.001 else f"{pval:.2e}"
        ax.text(0.04, 0.97,
                f"rho={rho:.3f}  p={p_str}\nn={len(xs)}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#aaaaaa", alpha=0.85))

    ax.set_xlabel(f"CKNNA k={K}", fontsize=8)
    ax.set_ylabel("WidowX SR (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title, fontsize=8, pad=3)


def main():
    present = [(p, t) for p, t in zip(CSV_PATHS, COL_TITLES) if os.path.exists(p)]
    present_paths  = [p for p, _ in present]
    present_titles = [t for _, t in present]
    skipped = [p for p in CSV_PATHS if not os.path.exists(p)]
    for p in skipped:
        print(f"WARNING: CSV not found, skipping column: {p}")

    all_csv_rows = [load_csv(p) for p in present_paths]

    nrows = len(SEQ_HORIZONS)
    ncols = len(present_paths)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.8 * nrows))
    if ncols == 1:
        axes = [[ax] for ax in axes]

    fig.suptitle("(b) proprio seq (img+txt) -- Flatten vs DTW v1..v7",
                 fontsize=12, y=1.005)

    for ri, h in enumerate(SEQ_HORIZONS):
        for ci, csv_rows in enumerate(all_csv_rows):
            ax = axes[ri][ci]
            title_parts = []
            if ri == 0:
                title_parts.append(present_titles[ci])
            title_parts.append(f"t+1 to t+{h}")
            draw_panel(ax, csv_rows, h, title="  |  ".join(title_parts))

    shown_groups = set()
    for meta in MODEL_META.values():
        shown_groups.add(meta["group"])
    handles = []
    for g in GROUP_COLORS:
        if g in shown_groups:
            handles.append(
                mlines.Line2D([], [], color=GROUP_COLORS[g],
                              marker=GROUP_MARKERS[g], linestyle="None",
                              markersize=7, markeredgecolor="k",
                              markeredgewidth=0.4, label=g)
            )
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=7, bbox_to_anchor=(0.5, -0.01), framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
