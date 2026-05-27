"""
Plot Exp 3: Raw Qwen vs StarVLA (finetuned) CKNNA-vs-horizon comparison.

Figure A: Qwen2.5-VL family (left: raw, right: StarVLA finetuned from Qwen2.5)
Figure B: Qwen3-VL family (left: raw, right: StarVLA finetuned from Qwen3)
Figure C: All raw/frozen models vs all end-to-end models (mean + individual)
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.environ.get("RECORD_DIR", "/home/ubuntu/verl/starVLA/cknna/record")
CSV_PATH = os.path.join(
    RECORD_DIR,
    "cknna_data_exp2",
    "finetune",
    "dtw_sym2_cuda_nowin",
    "results_groups_dtw_sym2_cuda_nowin.csv",
)
FIG_DIR = os.path.join(RECORD_DIR, "exp3_figures")

QWEN25_RAW = ["qwen25vl-3b-raw"]
QWEN25_FT = [
    "Qwen-GR00T-Bridge",
    "Qwen-GR00T-Bridge-RT-1",
    "Qwen-OFT-Bridge-RT-1",
]
QWEN3_RAW = ["qwen3vl-4b-raw"]
QWEN3_FT = [
    "Qwen3VL-GR00T-Bridge-RT-1",
    "Qwen3VL-OFT-Bridge-RT-1",
]

FROZEN_MODELS = ["qwen25vl-3b-raw", "qwen3vl-4b-raw", "pi0-lerobot-bridge", "groot-n15-bridge"]
E2E_MODELS = [
    "Qwen-GR00T-Bridge", "Qwen-GR00T-Bridge-RT-1", "Qwen-OFT-Bridge-RT-1",
    "Qwen3VL-GR00T-Bridge-RT-1", "Qwen3VL-OFT-Bridge-RT-1",
    "cogact-base-bridge", "cogact-large-bridge", "cogact-small-bridge",
    "groot-n16-bridge", "openvla-7b-bridge",
    "spatialvla-sft-bridge", "rt1x-bridge", "octo-base-bridge",
]

LABEL_MAP = {
    "qwen25vl-3b-raw": "Qwen2.5-VL-3B (raw)",
    "qwen3vl-4b-raw": "Qwen3-VL-4B (raw)",
    "pi0-lerobot-bridge": "Pi0 (frozen SigLIP)",
    "groot-n15-bridge": "GR00T-N1.5 (frozen Eagle)",
    "Qwen-GR00T-Bridge": "Qwen-GR00T-Br (FT)",
    "Qwen-GR00T-Bridge-RT-1": "Qwen-GR00T (FT)",
    "Qwen-OFT-Bridge-RT-1": "Qwen-OFT (FT)",
    "Qwen3VL-GR00T-Bridge-RT-1": "Qwen3-GR00T (FT)",
    "Qwen3VL-OFT-Bridge-RT-1": "Qwen3-OFT (FT)",
}


def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def get_cknna_by_horizon(rows, model, variant="imgtext"):
    data = {}
    for r in rows:
        if r["model"] == model and r["feats_A_variant"] == variant:
            h = int(r["horizon"])
            data[h] = float(r["cknna_k10"])
    return data


def plot_family_comparison(rows, raw_models, ft_models, family_name, figpath):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    ax_raw = axes[0]
    ax_ft = axes[1]

    for m in raw_models:
        data = get_cknna_by_horizon(rows, m)
        hs = sorted(data.keys())
        vals = [data[h] for h in hs]
        label = LABEL_MAP.get(m, m)
        ax_raw.plot(hs, vals, "o-", linewidth=2.5, markersize=6, label=label)

    ax_raw.set_title(f"{family_name}: Raw (pretrained, no robot FT)", fontsize=12)
    ax_raw.set_xlabel("Horizon h")
    ax_raw.set_ylabel("CKNNA (imgtext, k=10)")
    ax_raw.legend(fontsize=9)
    ax_raw.grid(True, alpha=0.3)

    for m in ft_models:
        data = get_cknna_by_horizon(rows, m)
        hs = sorted(data.keys())
        vals = [data[h] for h in hs]
        label = LABEL_MAP.get(m, m)
        ax_ft.plot(hs, vals, "s-", linewidth=1.5, markersize=5, label=label)

    ax_ft.set_title(f"{family_name}: End-to-end finetuned (StarVLA)", fontsize=12)
    ax_ft.set_xlabel("Horizon h")
    ax_ft.set_ylabel("CKNNA (imgtext, k=10)")
    ax_ft.legend(fontsize=9)
    ax_ft.grid(True, alpha=0.3)

    fig.suptitle(
        f"Exp 3: {family_name} -- Raw Pretrained vs End-to-End Finetuned",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(figpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {figpath}")


def plot_frozen_vs_e2e(rows, figpath):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax_frozen = axes[0]
    ax_e2e = axes[1]

    frozen_curves = []
    for m in FROZEN_MODELS:
        data = get_cknna_by_horizon(rows, m)
        if not data:
            continue
        hs = sorted(data.keys())
        vals = [data[h] for h in hs]
        label = LABEL_MAP.get(m, m)
        ax_frozen.plot(hs, vals, "o-", linewidth=1.5, markersize=4, alpha=0.7, label=label)
        frozen_curves.append(vals)

    if frozen_curves:
        mean_frozen = np.mean(frozen_curves, axis=0)
        ax_frozen.plot(hs, mean_frozen, "k-", linewidth=3, markersize=0, label="Mean (frozen/raw)")

    ax_frozen.set_title("Frozen / Raw VLM Backbones", fontsize=12)
    ax_frozen.set_xlabel("Horizon h")
    ax_frozen.set_ylabel("CKNNA (imgtext, k=10)")
    ax_frozen.legend(fontsize=8, loc="lower right")
    ax_frozen.grid(True, alpha=0.3)

    e2e_curves = []
    for m in E2E_MODELS:
        data = get_cknna_by_horizon(rows, m)
        if not data:
            continue
        hs = sorted(data.keys())
        vals = [data[h] for h in hs]
        label = LABEL_MAP.get(m, m)
        ax_e2e.plot(hs, vals, "s-", linewidth=1, markersize=3, alpha=0.5, label=label)
        e2e_curves.append(vals)

    if e2e_curves:
        mean_e2e = np.mean(e2e_curves, axis=0)
        ax_e2e.plot(hs, mean_e2e, "k-", linewidth=3, markersize=0, label="Mean (end-to-end)")

    ax_e2e.set_title("End-to-End Finetuned VLM Backbones", fontsize=12)
    ax_e2e.set_xlabel("Horizon h")
    ax_e2e.set_ylabel("CKNNA (imgtext, k=10)")
    ax_e2e.legend(fontsize=6, loc="lower right", ncol=2)
    ax_e2e.grid(True, alpha=0.3)

    fig.suptitle(
        "Exp 3: Frozen/Raw vs End-to-End -- CKNNA Horizon Trends",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(figpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {figpath}")


def plot_normalized_comparison(rows, figpath):
    """Normalize each model's CKNNA by its h=1 value to compare trend shapes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, models, title, marker in [
        (axes[0], FROZEN_MODELS, "Frozen / Raw VLM", "o-"),
        (axes[1], E2E_MODELS, "End-to-End Finetuned VLM", "s-"),
    ]:
        curves = []
        for m in models:
            data = get_cknna_by_horizon(rows, m)
            if not data:
                continue
            hs = sorted(data.keys())
            vals = [data[h] for h in hs]
            base = vals[0]
            if base > 0.01:
                normed = [v / base for v in vals]
                label = LABEL_MAP.get(m, m)
                ax.plot(hs, normed, marker, linewidth=1.2, markersize=4, alpha=0.7, label=label)
                curves.append(normed)

        if curves:
            mean_curve = np.mean(curves, axis=0)
            ax.plot(hs, mean_curve, "k-", linewidth=3, label="Mean")

        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Horizon h")
        ax.set_ylabel("CKNNA / CKNNA(h=1)")
        ax.legend(fontsize=7, loc="upper left", ncol=1)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Exp 3: Normalized CKNNA Trends (relative to h=1)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(figpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {figpath}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    rows = load_csv(CSV_PATH)

    plot_family_comparison(
        rows, QWEN25_RAW, QWEN25_FT, "Qwen2.5-VL",
        os.path.join(FIG_DIR, "fig_exp3_qwen25_family.png"),
    )
    plot_family_comparison(
        rows, QWEN3_RAW, QWEN3_FT, "Qwen3-VL",
        os.path.join(FIG_DIR, "fig_exp3_qwen3_family.png"),
    )
    plot_frozen_vs_e2e(
        rows, os.path.join(FIG_DIR, "fig_exp3_frozen_vs_e2e.png"),
    )
    plot_normalized_comparison(
        rows, os.path.join(FIG_DIR, "fig_exp3_normalized.png"),
    )
    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
