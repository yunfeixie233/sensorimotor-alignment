"""
plot_50k_base.py

Produces scatter figures (x=CKNNA alignment, y=WidowX SR) for BASE model evaluation.

Base models only (no fine-tuned checkpoints):
  CogACT-{S,B,L}, GR00T-N1.5, GR00T-N1.6, OpenVLA-7B, Pi0, SpatialVLA, RT-1-X, Octo

Two versions:
  _all       -- all models
  _onlyvla   -- excluding RT-1-X and Octo

Usage:
    cd /home/ubuntu/verl/starVLA
    python cknna/record/scripts/plot_50k_base.py
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.environ.get("RECORD_DIR", "/home/ubuntu/verl/starVLA/cknna/record")
RESULTS_CSV = os.path.join(RECORD_DIR, "base", "flatten",
                           "cknna_50k_base_oricknna_results.csv")
OUT_DIR = os.path.join(RECORD_DIR, "base", "flatten")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_META = {
    "cogact-small-bridge":   {"sr": 51.0, "group": "CogACT",      "label": "CogACT-S"},
    "cogact-base-bridge":    {"sr": 51.3, "group": "CogACT",      "label": "CogACT-B"},
    "cogact-large-bridge":   {"sr": 58.3, "group": "CogACT",      "label": "CogACT-L"},
    "groot-n15-base":        {"sr": 36.5, "group": "GR00T",       "label": "GR00T-N1.5"},
    "groot-n16-base":        {"sr": 57.1, "group": "GR00T",       "label": "GR00T-N1.6"},
    "openvla-7b-bridge":     {"sr":  1.0, "group": "OpenVLA",     "label": "OpenVLA-7B"},
    "pi0-base":              {"sr": 47.9, "group": "Pi0",         "label": "Pi0"},
    "spatialvla-pt-base":    {"sr": 42.7, "group": "SpatialVLA",  "label": "SpatialVLA"},
    "rt1x-bridge":           {"sr":  0.0, "group": "RT-1-X",      "label": "RT-1-X"},
    "octo-base-bridge":      {"sr": 20.3, "group": "Octo",        "label": "Octo"},
}

EXCLUDE_ONLYVLA = {"rt1x-bridge", "octo-base-bridge"}

GROUP_COLORS = {
    "CogACT":      "#ff7f0e",
    "GR00T":       "#2ca02c",
    "OpenVLA":     "#d62728",
    "Pi0":         "#9467bd",
    "SpatialVLA":  "#8c564b",
    "RT-1-X":      "#7f7f7f",
    "Octo":        "#bcbd22",
}

GROUP_MARKERS = {
    "CogACT":      "^",
    "GR00T":       "D",
    "OpenVLA":     "v",
    "Pi0":         "P",
    "SpatialVLA":  "X",
    "RT-1-X":      "*",
    "Octo":        "h",
}

FA_VARIANTS   = ["imgtext", "img", "txt"]
FA_LABELS     = {"imgtext": "img+txt", "img": "image only", "txt": "text only"}
SEQ_HORIZONS  = [1, 3, 7, 15]
K             = 10


def load_csv():
    rows = []
    with open(RESULTS_CSV) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def get_val(rows, model, fa_var, fb_type, horizon):
    col = f"cknna_k{K}"
    for r in rows:
        if (r["model"] == model and
                r["feats_A_variant"] == fa_var and
                r["feats_B_type"] == fb_type and
                int(r["horizon"]) == horizon and
                col in r):
            return float(r[col])
    return None


def draw_panel(ax, rows, fa_var, fb_type, horizon,
               exclude=None, row_label="", col_label=""):
    exclude = exclude or set()
    xs, ys = [], []
    for model_key, meta in MODEL_META.items():
        if model_key in exclude:
            continue
        val = get_val(rows, model_key, fa_var, fb_type, horizon)
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
        ax.text(0.04, 0.97, f"rho={rho:.3f}  p={p_str}\nn={len(xs)}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#aaaaaa", alpha=0.85))

    ax.set_xlabel(f"CKNNA k={K}", fontsize=8)
    ax.set_ylabel("WidowX SR (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25)

    parts = []
    if col_label:
        parts.append(col_label)
    if row_label:
        parts.append(row_label)
    if parts:
        ax.set_title("  |  ".join(parts), fontsize=8, pad=3)


def add_legend(fig, exclude=None):
    exclude = exclude or set()
    shown_groups = {meta["group"] for key, meta in MODEL_META.items()
                    if key not in exclude}
    patches = [mpatches.Patch(color=GROUP_COLORS[g], label=g)
               for g in GROUP_COLORS if g in shown_groups]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=7, bbox_to_anchor=(0.5, 0.0), framealpha=0.9)


def make_fig_1row(rows, fb_type, title, stem, exclude):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(title, fontsize=11, y=1.01)
    for ax, fa in zip(axes, FA_VARIANTS):
        draw_panel(ax, rows, fa, fb_type, 0,
                   exclude=exclude, col_label=FA_LABELS[fa])
    add_legend(fig, exclude)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, stem)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def make_fig_5rows(rows, fb_type, title, stem, exclude):
    fig, axes = plt.subplots(len(SEQ_HORIZONS), 3,
                             figsize=(13, 4 * len(SEQ_HORIZONS)))
    fig.suptitle(title, fontsize=12, y=1.005)
    for ri, h in enumerate(SEQ_HORIZONS):
        for ci, fa in enumerate(FA_VARIANTS):
            draw_panel(axes[ri][ci], rows, fa, fb_type, h,
                       exclude=exclude,
                       row_label=f"t+1 to t+{h}",
                       col_label=FA_LABELS[fa] if ri == 0 else "")
    add_legend(fig, exclude)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, stem)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    print(f"Loading {RESULTS_CSV} ...")
    rows = load_csv()
    print(f"  {len(rows)} rows\n")

    VERSIONS = [
        ("all",     set()),
        ("onlyvla", EXCLUDE_ONLYVLA),
    ]

    for suffix, exclude in VERSIONS:
        print(f"--- version: {suffix} ---")

        print("  (a) proprio at t")
        make_fig_1row(rows, "proprio_t",
                      "(a) proprio at t  [BASE models]",
                      f"fig_a_proprio_t_{suffix}.png", exclude)

        print("  (b) proprio from t to t+a")
        make_fig_5rows(rows, "proprio_seq",
                       "(b) proprio from t to t+a  [BASE models]",
                       f"fig_b_proprio_seq_{suffix}.png", exclude)

        print("  (c) model action repr at t")
        make_fig_1row(rows, "feats_action_t",
                      "(c) model action representation at t  [BASE models]",
                      f"fig_c_feats_action_t_{suffix}.png", exclude)

        print("  (d) real action at t")
        make_fig_1row(rows, "real_action_t",
                      "(d) real action at t  [BASE models]",
                      f"fig_d_real_action_t_{suffix}.png", exclude)

        print("  (e) real action from t to t+a")
        make_fig_5rows(rows, "real_action_seq",
                       "(e) real action from t to t+a  [BASE models]",
                       f"fig_e_real_action_seq_{suffix}.png", exclude)

        print()

    print(f"Done. Figures in {OUT_DIR}/")


if __name__ == "__main__":
    main()
