"""
Plot CKNNA results for the DROID dataset (Franka Panda, Group L only).

Generates:
  1. fig_cknna_vs_horizon_groups.png  — CKNNA vs horizon (line per model, 3 variant cols)
  2. fig_cknna_vs_sr_L.png            — CKNNA vs WidowX SR (scatter, rows=horizons, cols=variants)
  3. fig_droid_vs_bridge.png          — cross-dataset scatter (DROID vs Bridge CKNNA)

Mimics the style of exp_horizon_L/plot_exp_horizon_L.py.
"""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import spearmanr, kendalltau

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKNNA_ROOT = os.environ.get("DATA_STORE", "/home/ubuntu/verl/starVLA/cknna")

DTW_VERSION = "sym2_cuda_nowin"
K = 10

DROID_CSV = os.path.join(SCRIPT_DIR, "finetune",
                         f"dtw_{DTW_VERSION}",
                         f"results_groups_dtw_{DTW_VERSION}.csv")

BRIDGE_CSV = os.path.join(CKNNA_ROOT, "record", "exp_horizon",
                          "cknna_data_exp1v2_full_L", "finetune",
                          f"dtw_{DTW_VERSION}",
                          f"results_groups_dtw_{DTW_VERSION}.csv")

META_CSV = os.path.join(CKNNA_ROOT, "cknna_data", "performance",
                        "model_metadata.csv")

DROID_HORIZONS = [1, 3, 7, 15, 25, 40, 75, 110, 150, 220]
BRIDGE_HORIZONS = [1, 3, 7, 15, 25, 40]

FIG_DIR = SCRIPT_DIR

FA_VARIANTS = ["imgtext", "img", "txt"]
FA_LABELS = {"imgtext": "img+txt", "img": "image only", "txt": "text only"}


# ── colors (same palette as Bridge plots) ──────────────────────────────────
FAMILY_COLOR = {
    "Qwen-GR00T-Bridge":          "#1f6eb5",
    "Qwen-GR00T-Bridge-RT-1":     "#4a90d9",
    "Qwen3VL-GR00T-Bridge-RT-1":  "#7bb8f0",
    "Qwen-OFT-Bridge-RT-1":       "#e07c00",
    "Qwen3VL-OFT-Bridge-RT-1":    "#f5a840",
    "spatialvla-sft-bridge":       "#6b3a9e",
    "pi0-lerobot-bridge":          "#c9436a",
    "openvla-7b-bridge":           "#2e7d46",
    "openvla-7b-bridge-ft-200k":   "#5ab070",
    "rt1x-bridge":                 "#555555",
    "octo-base-bridge":            "#999999",
    "cogact-small-bridge":         "#8b0000",
    "cogact-base-bridge":          "#c62828",
    "cogact-large-bridge":         "#ef5350",
    "groot-n15-bridge":            "#5d4037",
    "groot-n16-bridge":            "#a1887f",
    "qwen25vl-3b-raw":             "#00838f",
    "qwen3vl-4b-raw":              "#4dd0e1",
    "prismatic-raw":               "#78909c",
}

COLOR_CARD = {
    "StarVLA-Qwen2.5": {"color": "#1f77b4", "marker": "o"},
    "StarVLA-Qwen3":   {"color": "#aec7e8", "marker": "s"},
    "CogACT":          {"color": "#ff7f0e", "marker": "^"},
    "GR00T":           {"color": "#2ca02c", "marker": "D"},
    "OpenVLA":         {"color": "#d62728", "marker": "v"},
    "Pi0":             {"color": "#9467bd", "marker": "P"},
    "SpatialVLA":      {"color": "#8c564b", "marker": "X"},
    "RT-1-X":          {"color": "#7f7f7f", "marker": "*"},
    "Octo":            {"color": "#bcbd22", "marker": "h"},
    "RawVLM":          {"color": "#17becf", "marker": "d"},
}


# ── helpers ────────────────────────────────────────────────────────────────
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


def load_model_meta():
    meta = {}
    with open(META_CSV, newline="") as f:
        for row in csv.DictReader(f):
            key = row["model_key"]
            is_ft = row["finetune_on_bridge"] == "yes"
            # CSV has duplicate keys (ft=yes then ft=no for same model).
            # Prefer the ft=yes entry: skip no-entry if yes already stored.
            if key in meta and meta[key]["finetune_on_bridge"] and not is_ft:
                continue
            meta[key] = {
                "sr":    float(row["success_rate"]),
                "group": row["group"],
                "label": row["figure_label"],
                "finetune_on_bridge": is_ft,
            }
    return meta


def load_cknna_csv(csv_path):
    """Return dict[model][variant][horizon] = cknna_k10."""
    data = defaultdict(lambda: defaultdict(dict))
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            data[r["model"]][r["feats_A_variant"]][int(r["horizon"])] = float(r["cknna_k10"])
    return data


# ── Fig 1: CKNNA vs horizon (line plot) ───────────────────────────────────
def fig_cknna_vs_horizon(data, horizons):
    all_cknna = []
    models = sorted(data.keys())
    for fa in FA_VARIANTS:
        for m in models:
            if fa not in data[m]:
                continue
            for h in horizons:
                v = data[m][fa].get(h)
                if v is not None:
                    all_cknna.append(v)

    if all_cknna:
        y_min, y_max = min(all_cknna), max(all_cknna)
        y_pad = (y_max - y_min) * 0.08
        y_lo, y_hi = y_min - y_pad, y_max + y_pad
    else:
        y_lo, y_hi = 0, 0.3

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    for ax, fa in zip(axes, FA_VARIANTS):
        xs_all, ys_all = [], []
        for m in models:
            if fa not in data[m]:
                continue
            ys = [data[m][fa].get(h, np.nan) for h in horizons]
            color = FAMILY_COLOR.get(m, "#aaaaaa")
            ax.plot(horizons, ys, marker="o", markersize=4, linewidth=1.3,
                    color=color, alpha=0.7, label=display_name(m))
            for h, y in zip(horizons, ys):
                if not np.isnan(y):
                    xs_all.append(h)
                    ys_all.append(y)

        # Mean line
        means = []
        for h in horizons:
            vals = [data[m][fa][h]
                    for m in models
                    if fa in data[m] and h in data[m][fa]]
            means.append(np.mean(vals) if vals else np.nan)
        ax.plot(horizons, means, marker="D", markersize=7, linewidth=2.8,
                color="black", zorder=10, label="Mean", linestyle="--")

        if len(xs_all) >= 3:
            rho, p = spearmanr(xs_all, ys_all)
            p_str = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        else:
            rho, p_str, sig = 0, "N/A", "N/A"

        ax.set_title(f"{FA_LABELS[fa]}  (H_max={horizons[-1]})\n"
                     f"Spearman rho={rho:.3f}  p={p_str}  {sig}",
                     fontsize=10.5)
        ax.set_xlabel("Horizon h", fontsize=11)
        ax.set_ylabel("CKNNA  (k=10)", fontsize=11)
        ax.set_xticks(horizons)
        ax.set_xticklabels([str(h) for h in horizons], fontsize=7, rotation=45)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.98), loc="upper left",
               fontsize=7, framealpha=0.9, ncol=1)
    fig.suptitle(
        "proprio_seq CKNNA [DTW] vs horizon\n"
        f"(DROID / Franka Panda, N=7762, {len(models)} models)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_cknna_vs_horizon_groups.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Fig 2: CKNNA vs WidowX SR (scatter) ──────────────────────────────────
def fig_cknna_vs_sr(data, model_meta, horizons):
    # Filter to models with Bridge SR and finetune_on_bridge
    ft_meta = {k: v for k, v in model_meta.items()
               if v["finetune_on_bridge"] and k in data}

    all_cknna, all_sr = [], []
    for fa in FA_VARIANTS:
        for h in horizons:
            for mk, meta in ft_meta.items():
                if fa in data[mk] and h in data[mk][fa]:
                    all_cknna.append(data[mk][fa][h])
                    all_sr.append(meta["sr"])

    if not all_cknna:
        print("No overlapping data for CKNNA vs SR plot; skipping.")
        return

    cknna_pad = (max(all_cknna) - min(all_cknna)) * 0.08
    sr_pad = (max(all_sr) - min(all_sr)) * 0.08
    x_lo, x_hi = min(all_cknna) - cknna_pad, max(all_cknna) + cknna_pad
    y_lo, y_hi = min(all_sr) - sr_pad, max(all_sr) + sr_pad

    n_h = len(horizons)
    fig, axes = plt.subplots(n_h, 3, figsize=(13, 3.5 * n_h))
    if n_h == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f"proprio_seq CKNNA [DTW] vs WidowX SR  "
                 f"(DROID features, H_max={horizons[-1]}, N=7762)",
                 fontsize=12, y=1.005)

    for ri, h in enumerate(horizons):
        for ci, fa in enumerate(FA_VARIANTS):
            ax = axes[ri, ci]
            xs, ys = [], []
            for mk, meta in ft_meta.items():
                if fa not in data[mk] or h not in data[mk][fa]:
                    continue
                val = data[mk][fa][h]
                c = COLOR_CARD.get(meta["group"], {"color": "#aaa"})["color"]
                m = COLOR_CARD.get(meta["group"], {"marker": "o"})["marker"]
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

            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.set_xlabel(f"CKNNA k={K}", fontsize=8)
            ax.set_ylabel("WidowX SR (%)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25)

            parts = []
            if ri == 0:
                parts.append(FA_LABELS[fa])
            parts.append(f"t+1 to t+{h}")
            ax.set_title("  |  ".join(parts), fontsize=8, pad=3)

    shown_groups = {meta["group"] for meta in ft_meta.values()}
    patches = [mpatches.Patch(color=COLOR_CARD[g]["color"], label=g)
               for g in COLOR_CARD if g in shown_groups]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=7, bbox_to_anchor=(0.5, 0.0), framealpha=0.9)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_cknna_vs_sr_L.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Fig 3: DROID vs Bridge cross-dataset scatter ─────────────────────────
def fig_droid_vs_bridge(droid_data, bridge_data, model_meta):
    droid_max_h = max(DROID_HORIZONS)
    bridge_max_h = max(BRIDGE_HORIZONS)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, fa in zip(axes, FA_VARIANTS):
        xs, ys, labels_plot = [], [], []
        for m in sorted(set(droid_data.keys()) & set(bridge_data.keys())):
            if fa not in droid_data[m] or fa not in bridge_data[m]:
                continue
            dv = droid_data[m][fa].get(droid_max_h)
            bv = bridge_data[m][fa].get(bridge_max_h)
            if dv is None or bv is None:
                continue
            xs.append(bv)
            ys.append(dv)
            labels_plot.append(m)

            c = FAMILY_COLOR.get(m, "#aaaaaa")
            meta = model_meta.get(m, {})
            grp = meta.get("group", "")
            mk = COLOR_CARD.get(grp, {"marker": "o"})["marker"]
            ax.scatter(bv, dv, c=c, marker=mk, s=100,
                       edgecolors="k", linewidths=0.5, zorder=3)
            ax.annotate(display_name(m), (bv, dv),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=6.5, color=c)

        if len(xs) >= 3:
            rho, p_rho = spearmanr(xs, ys)
            tau, p_tau = kendalltau(xs, ys)
            p_str = f"{p_rho:.2e}" if p_rho < 0.001 else f"{p_rho:.3f}"
            ax.text(0.04, 0.97,
                    f"Spearman rho={rho:.3f}  p={p_str}\n"
                    f"Kendall  tau={tau:.3f}\nn={len(xs)}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#aaaaaa", alpha=0.9))

            # Fit line
            z = np.polyfit(xs, ys, 1)
            xr = np.linspace(min(xs) * 0.95, max(xs) * 1.05, 50)
            ax.plot(xr, np.polyval(z, xr), "--", color="#888888",
                    linewidth=1.2, alpha=0.6, zorder=1)

        # Identity reference
        lo = min(min(xs), min(ys)) * 0.9
        hi = max(max(xs), max(ys)) * 1.1
        ax.plot([lo, hi], [lo, hi], ":", color="#cccccc", linewidth=0.8, zorder=0)

        ax.set_xlabel(f"Bridge CKNNA (h={bridge_max_h}, k={K})", fontsize=10)
        ax.set_ylabel(f"DROID CKNNA (h={droid_max_h}, k={K})", fontsize=10)
        ax.set_title(FA_LABELS[fa], fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=9)

    fig.suptitle(
        "Cross-Dataset CKNNA: DROID (Franka) vs Bridge (WidowX)\n"
        f"DTW {DTW_VERSION}, Group L, N=7762",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_droid_vs_bridge.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Statistics ─────────────────────────────────────────────────────────────
def print_statistics(droid_data, bridge_data, model_meta):
    print("\n" + "=" * 70)
    print("DROID CKNNA Statistics")
    print("=" * 70)

    # CKNNA vs SR correlations (Bridge SR, DROID CKNNA)
    ft_meta = {k: v for k, v in model_meta.items()
               if v["finetune_on_bridge"] and k in droid_data}

    print("\n--- CKNNA (DROID) vs SR (Bridge) Spearman ---")
    for fa in FA_VARIANTS:
        print(f"\n  {FA_LABELS[fa]}:")
        for h in DROID_HORIZONS:
            xs, ys = [], []
            for mk, meta in ft_meta.items():
                if fa in droid_data[mk] and h in droid_data[mk][fa]:
                    xs.append(droid_data[mk][fa][h])
                    ys.append(meta["sr"])
            if len(xs) >= 3:
                rho, p = spearmanr(xs, ys)
                p_str = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
                print(f"    h={h:3d}: rho={rho:.3f}  p={p_str}  {sig}  n={len(xs)}")

    # Cross-dataset correlations
    print("\n--- Cross-Dataset Spearman (DROID vs Bridge CKNNA) ---")
    droid_max_h = max(DROID_HORIZONS)
    bridge_max_h = max(BRIDGE_HORIZONS)
    for fa in FA_VARIANTS:
        common = sorted(k for k in set(droid_data) & set(bridge_data)
                        if fa in droid_data[k] and fa in bridge_data[k]
                        and droid_max_h in droid_data[k][fa]
                        and bridge_max_h in bridge_data[k][fa])
        if len(common) < 3:
            continue
        dv = [droid_data[m][fa][droid_max_h] for m in common]
        bv = [bridge_data[m][fa][bridge_max_h] for m in common]
        rho, p = spearmanr(dv, bv)
        tau, pt = kendalltau(dv, bv)
        print(f"  {FA_LABELS[fa]:>10s}: rho={rho:.4f} (p={p:.2e})  "
              f"tau={tau:.4f} (p={pt:.2e})  n={len(common)}")

    # Per-model table
    print(f"\n--- Per-model CKNNA at max horizon (imgtext) ---")
    print(f"{'Model':<35s} {'DROID':>8s} {'Bridge':>8s} {'SR':>7s}")
    rows = []
    for m in sorted(droid_data.keys()):
        dv = droid_data[m].get("imgtext", {}).get(droid_max_h)
        bv = bridge_data.get(m, {}).get("imgtext", {}).get(bridge_max_h)
        sr = model_meta.get(m, {}).get("sr", 0)
        if dv is not None:
            rows.append((m, dv, bv, sr))
    rows.sort(key=lambda x: -x[1])
    for m, dv, bv, sr in rows:
        bstr = f"{bv:8.5f}" if bv is not None else "     N/A"
        print(f"{display_name(m):<35s} {dv:8.5f} {bstr} {sr:7.1f}")

    # CKNNA vs horizon Spearman
    print("\n--- CKNNA vs horizon overall Spearman (all models × all horizons) ---")
    models = sorted(droid_data.keys())
    for fa in ["imgtext"]:
        xs_all, ys_all = [], []
        for m in models:
            if fa not in droid_data[m]:
                continue
            for h in DROID_HORIZONS:
                if h in droid_data[m][fa]:
                    xs_all.append(h)
                    ys_all.append(droid_data[m][fa][h])
        if len(xs_all) >= 3:
            rho, p = spearmanr(xs_all, ys_all)
            p_str = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
            print(f"  {FA_LABELS[fa]:>10s}: rho={rho:.3f}  p={p_str}  n={len(xs_all)}")

    # Mean CKNNA per horizon
    print(f"\n--- Mean CKNNA (imgtext) per horizon ---")
    for h in DROID_HORIZONS:
        vals = [droid_data[m]["imgtext"][h] for m in models
                if "imgtext" in droid_data[m] and h in droid_data[m]["imgtext"]]
        print(f"  h={h:3d}: mean={np.mean(vals):.5f}  std={np.std(vals):.5f}  n={len(vals)}")


# ── main ───────────────────────────────────────────────────────────────────
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    model_meta = load_model_meta()
    droid_data = load_cknna_csv(DROID_CSV)
    bridge_data = load_cknna_csv(BRIDGE_CSV)

    print(f"DROID models: {len(droid_data)}")
    print(f"Bridge models: {len(bridge_data)}")
    print(f"DROID horizons: {DROID_HORIZONS}")

    fig_cknna_vs_horizon(droid_data, DROID_HORIZONS)
    fig_cknna_vs_sr(droid_data, model_meta, DROID_HORIZONS)
    fig_droid_vs_bridge(droid_data, bridge_data, model_meta)
    print_statistics(droid_data, bridge_data, model_meta)
    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
