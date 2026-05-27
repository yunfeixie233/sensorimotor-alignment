"""
Compare Pearson r vs Spearman rho for CKNNA-SR correlation.

Reads existing CKNNA CSVs + SR data, computes both correlations,
prints comparison tables, and generates scatter plots for significant pairs.

Usage:
    python analysis/compare_pearson_spearman.py
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINETUNED_CSV = os.path.join(BASE, "record/0325_vlm4vla_raw/droid/results_cknna_7feat_droid_platonic.csv")
RAWVLM_CSV = os.path.join(BASE, "record/0325_vlm4vla_raw/droid/results_cknna_7feat_droid_rawvlm_asymplat.csv")
OUT_DIR = os.path.join(BASE, "record/0325_vlm4vla_raw/droid_pearsonr")

# ── SR data ───────────────────────────────────────────────────────────────────
# Finetuned VLAs: WidowX SR (n=13, excludes openvla-ft-200k)
FINETUNED_SR = {
    "Qwen-GR00T-Bridge": 71.4,
    "Qwen-GR00T-Bridge-RT-1": 63.6,
    "Qwen-OFT-Bridge-RT-1": 44.2,
    "Qwen3VL-GR00T-Bridge-RT-1": 65.3,
    "Qwen3VL-OFT-Bridge-RT-1": 42.7,
    "cogact-small-bridge": 51.0,
    "cogact-base-bridge": 51.3,
    "cogact-large-bridge": 58.3,
    "groot-n15-bridge": 36.5,
    "groot-n16-bridge": 57.1,
    "openvla-7b-bridge": 1.0,
    "pi0-lerobot-bridge": 47.9,
    "spatialvla-sft-bridge": 42.7,
}

# Raw VLMs: VLM4VLA benchmark SR (n=8)
VLM_SR = {
    "qwen25vl-3b-raw":  {"calvin": 3.856, "simpler": 48.0, "libero": 43.0},
    "qwen25vl-7b-raw":  {"calvin": 4.057, "simpler": 46.8, "libero": 45.0},
    "qwen3vl-2b-raw":   {"calvin": 4.142, "simpler": 49.0, "libero": 55.8},
    "qwen3vl-4b-raw":   {"calvin": 3.943, "simpler": 56.3, "libero": 44.4},
    "qwen3vl-8b-raw":   {"calvin": 4.035, "simpler": 58.3, "libero": 46.2},
    "paligemma1-raw":    {"calvin": 3.506, "simpler": 55.3, "libero": 44.2},
    "paligemma2-raw":    {"calvin": 3.406, "simpler": 57.3, "libero": 46.2},
    "kosmos2-raw":       {"calvin": 3.096, "simpler": 60.4, "libero": 55.0},
}

FEATURES = ["vit_raw", "pre_llm", "pre_llm_txt", "pre_llm_vt", "img", "txt", "imgtext"]
FEAT_LABELS = {
    "vit_raw": "P1-V", "pre_llm": "P2-V", "pre_llm_txt": "P2-T",
    "pre_llm_vt": "P2-VT", "img": "P3-V", "txt": "P3-T", "imgtext": "P3-VT",
}
HORIZONS = [1, 3, 7, 15, 25, 40, 75, 110, 150, 220]
BENCHMARKS_RAW = [("simpler", "SimplerEnv"), ("libero", "Libero"), ("calvin", "Calvin")]


def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["horizon"] = int(r["horizon"])
            r["cknna_k10"] = float(r["cknna_k10"])
            rows.append(r)
    return rows


def get_pairs_finetuned(rows, feat, horizon):
    """Return (cknna_list, sr_list) for finetuned VLAs."""
    cknna_vals, sr_vals = [], []
    for r in rows:
        if r["feats_A_variant"] != feat or r["horizon"] != horizon:
            continue
        m = r["model"]
        if m not in FINETUNED_SR:
            continue
        cknna_vals.append(r["cknna_k10"])
        sr_vals.append(FINETUNED_SR[m])
    return cknna_vals, sr_vals


def get_pairs_rawvlm(rows, feat, horizon, benchmark_key):
    """Return (cknna_list, sr_list) for raw VLMs."""
    cknna_vals, sr_vals = [], []
    for r in rows:
        if r["feats_A_variant"] != feat or r["horizon"] != horizon:
            continue
        m = r["model"]
        if m not in VLM_SR:
            continue
        cknna_vals.append(r["cknna_k10"])
        sr_vals.append(VLM_SR[m][benchmark_key])
    return cknna_vals, sr_vals


def get_model_labels_finetuned(rows, feat, horizon):
    """Return model labels for finetuned pairs."""
    labels = []
    for r in rows:
        if r["feats_A_variant"] != feat or r["horizon"] != horizon:
            continue
        m = r["model"]
        if m not in FINETUNED_SR:
            continue
        labels.append(m)
    return labels


def get_model_labels_rawvlm(rows, feat, horizon, benchmark_key):
    """Return model labels for raw VLM pairs."""
    labels = []
    for r in rows:
        if r["feats_A_variant"] != feat or r["horizon"] != horizon:
            continue
        m = r["model"]
        if m not in VLM_SR:
            continue
        labels.append(m)
    return labels


def corr_both(x, y):
    """Return (spearman_rho, spearman_p_onetail, pearson_r, pearson_p_onetail)."""
    x, y = np.array(x), np.array(y)
    if len(x) < 3:
        return None, None, None, None

    rho_s, p_s = spearmanr(x, y)
    p_s_one = p_s / 2 if rho_s > 0 else 1 - p_s / 2

    r_p, p_p = pearsonr(x, y)
    p_p_one = p_p / 2 if r_p > 0 else 1 - p_p / 2

    return rho_s, p_s_one, r_p, p_p_one


def fmt_cell(val, pval, sig_thresh=0.05):
    """Format a correlation cell: +0.72 (0.003) with ** if sig."""
    if val is None:
        return "---"
    star = "**" if pval < sig_thresh else ""
    return f"{star}{val:+.3f} ({pval:.3f}){star}"


def print_comparison_table(title, results, horizons):
    """Print a side-by-side Spearman vs Pearson table."""
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")

    header = f"{'Feature':<8}"
    for h in horizons:
        header += f" | {'h='+str(h):^28}"
    print(header)

    sub = f"{'':8}"
    for _ in horizons:
        sub += f" | {'Spearman':^13} {'Pearson':^13}"
    print(sub)
    print("-" * len(header))

    for feat in FEATURES:
        line = f"{FEAT_LABELS[feat]:<8}"
        for h in horizons:
            key = (feat, h)
            if key in results:
                rho_s, p_s, r_p, p_p = results[key]
                s_str = f"{rho_s:+.2f}" if rho_s is not None else "---"
                p_str = f"{r_p:+.2f}" if r_p is not None else "---"
                s_sig = "*" if p_s is not None and p_s < 0.05 else " "
                p_sig = "*" if p_p is not None and p_p < 0.05 else " "
                line += f" | {s_str}{s_sig}  {p_str}{p_sig}       "
            else:
                line += f" | {'---':^28}"
        print(line)


def make_scatter(x, y, labels, feat, horizon, benchmark, rho_s, p_s, r_p, p_p, out_path):
    """Generate a scatter plot with both regression lines and correlation stats."""
    x, y = np.array(x), np.array(y)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    ax.scatter(x, y, s=80, c="#2196F3", edgecolors="k", linewidths=0.5, zorder=3)

    for i, lbl in enumerate(labels):
        short = lbl.replace("-raw", "").replace("-bridge", "").replace("-lerobot", "")
        ax.annotate(short, (x[i], y[i]), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, color="#555555")

    # Linear fit (Pearson line)
    if len(x) >= 2:
        z = np.polyfit(x, y, 1)
        xline = np.linspace(x.min(), x.max(), 100)
        ax.plot(xline, np.polyval(z, xline), "--", color="#F44336", alpha=0.7,
                linewidth=1.5, label="Linear fit")

    # Stats text box
    stats = (f"Spearman rho = {rho_s:+.3f}  (p = {p_s:.4f})\n"
             f"Pearson r    = {r_p:+.3f}  (p = {p_p:.4f})\n"
             f"n = {len(x)}")
    ax.text(0.03, 0.97, stats, transform=ax.transAxes, fontsize=9, va="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#aaaaaa", alpha=0.9))

    ax.set_xlabel(f"CKNNA ({FEAT_LABELS[feat]})", fontsize=11)
    ax.set_ylabel(f"{benchmark} SR (%)", fontsize=11)
    ax.set_title(f"{FEAT_LABELS[feat]} vs {benchmark} SR  (h={horizon})", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    scatter_dir = os.path.join(OUT_DIR, "scatter_plots")
    os.makedirs(scatter_dir, exist_ok=True)

    # Load CSVs
    ft_rows = load_csv(FINETUNED_CSV)
    raw_rows = load_csv(RAWVLM_CSV)

    # ── A. Finetuned VLAs vs WidowX SR ────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  A. FINETUNED VLAs vs WidowX SR (n=13)")
    print("=" * 80)

    ft_results = {}
    ft_sig_spearman = 0
    ft_sig_pearson = 0
    ft_sig_both = 0
    ft_total = 0

    # Collect all results for CSV output
    all_rows_out = []

    for feat in FEATURES:
        for h in HORIZONS:
            cknna, sr = get_pairs_finetuned(ft_rows, feat, h)
            rho_s, p_s, r_p, p_p = corr_both(cknna, sr)
            ft_results[(feat, h)] = (rho_s, p_s, r_p, p_p)
            ft_total += 1

            if rho_s is not None:
                s_sig = p_s < 0.05
                p_sig = p_p < 0.05
                ft_sig_spearman += s_sig
                ft_sig_pearson += p_sig
                ft_sig_both += (s_sig and p_sig)

                all_rows_out.append({
                    "set": "finetuned", "benchmark": "WidowX",
                    "feature": FEAT_LABELS[feat], "horizon": h, "n": len(cknna),
                    "spearman_rho": f"{rho_s:.4f}", "spearman_p": f"{p_s:.4f}",
                    "pearson_r": f"{r_p:.4f}", "pearson_p": f"{p_p:.4f}",
                    "spearman_sig": s_sig, "pearson_sig": p_sig,
                })

    # Print compact table
    print(f"\n{'Feature':<8}", end="")
    for h in HORIZONS:
        print(f" | h={h:>3} S / P ", end="")
    print()
    print("-" * 130)

    for feat in FEATURES:
        print(f"{FEAT_LABELS[feat]:<8}", end="")
        for h in HORIZONS:
            rho_s, p_s, r_p, p_p = ft_results[(feat, h)]
            if rho_s is None:
                print(f" | {'---':^13}", end="")
                continue
            s_mark = "*" if p_s < 0.05 else " "
            p_mark = "*" if p_p < 0.05 else " "
            print(f" | {rho_s:+.2f}{s_mark}/{r_p:+.2f}{p_mark}", end="")
        print()

    print(f"\nSignificance counts (out of {ft_total}):")
    print(f"  Spearman p<0.05: {ft_sig_spearman}")
    print(f"  Pearson  p<0.05: {ft_sig_pearson}")
    print(f"  Both     p<0.05: {ft_sig_both}")
    print(f"  Spearman only:   {ft_sig_spearman - ft_sig_both}")
    print(f"  Pearson only:    {ft_sig_pearson - ft_sig_both}")

    # ── B. Raw VLMs vs VLM4VLA benchmarks ─────────────────────────────────────
    raw_sig_spearman = 0
    raw_sig_pearson = 0
    raw_sig_both = 0
    raw_total = 0

    for bk_key, bk_name in BENCHMARKS_RAW:
        print(f"\n{'='*80}")
        print(f"  B. RAW VLMs vs {bk_name} SR (n=8)")
        print(f"{'='*80}")

        print(f"\n{'Feature':<8}", end="")
        for h in HORIZONS:
            print(f" | h={h:>3} S / P ", end="")
        print()
        print("-" * 130)

        for feat in FEATURES:
            print(f"{FEAT_LABELS[feat]:<8}", end="")
            for h in HORIZONS:
                cknna, sr = get_pairs_rawvlm(raw_rows, feat, h, bk_key)
                rho_s, p_s, r_p, p_p = corr_both(cknna, sr)
                raw_total += 1

                if rho_s is not None:
                    s_sig = p_s < 0.05
                    p_sig = p_p < 0.05
                    raw_sig_spearman += s_sig
                    raw_sig_pearson += p_sig
                    raw_sig_both += (s_sig and p_sig)

                    s_mark = "*" if s_sig else " "
                    p_mark = "*" if p_sig else " "
                    print(f" | {rho_s:+.2f}{s_mark}/{r_p:+.2f}{p_mark}", end="")

                    all_rows_out.append({
                        "set": "rawvlm", "benchmark": bk_name,
                        "feature": FEAT_LABELS[feat], "horizon": h, "n": len(cknna),
                        "spearman_rho": f"{rho_s:.4f}", "spearman_p": f"{p_s:.4f}",
                        "pearson_r": f"{r_p:.4f}", "pearson_p": f"{p_p:.4f}",
                        "spearman_sig": s_sig, "pearson_sig": p_sig,
                    })
                else:
                    print(f" | {'---':^13}", end="")
            print()

    total = ft_total + raw_total
    total_s = ft_sig_spearman + raw_sig_spearman
    total_p = ft_sig_pearson + raw_sig_pearson
    total_both = ft_sig_both + raw_sig_both

    print(f"\n{'='*80}")
    print(f"  GRAND TOTAL (out of {total})")
    print(f"{'='*80}")
    print(f"  Spearman p<0.05: {total_s}")
    print(f"  Pearson  p<0.05: {total_p}")
    print(f"  Both     p<0.05: {total_both}")
    print(f"  Spearman only:   {total_s - total_both}")
    print(f"  Pearson only:    {total_p - total_both}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "pearson_vs_spearman.csv")
    fields = ["set", "benchmark", "feature", "horizon", "n",
              "spearman_rho", "spearman_p", "pearson_r", "pearson_p",
              "spearman_sig", "pearson_sig"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows_out)
    print(f"\nSaved: {csv_path} ({len(all_rows_out)} rows)")

    # ── Scatter plots for significant pairs ───────────────────────────────────
    print(f"\nGenerating scatter plots in {scatter_dir}/")
    n_plots = 0

    # Finetuned: plot any (feat, h) where either Spearman or Pearson is sig
    for feat in FEATURES:
        for h in HORIZONS:
            rho_s, p_s, r_p, p_p = ft_results[(feat, h)]
            if rho_s is None:
                continue
            if p_s >= 0.05 and p_p >= 0.05:
                continue
            cknna, sr = get_pairs_finetuned(ft_rows, feat, h)
            labels = get_model_labels_finetuned(ft_rows, feat, h)
            out = os.path.join(scatter_dir,
                               f"ft_WidowX_{FEAT_LABELS[feat]}_h{h}.png")
            make_scatter(cknna, sr, labels, feat, h, "WidowX",
                         rho_s, p_s, r_p, p_p, out)
            n_plots += 1

    # Raw VLMs: plot any (feat, h, benchmark) where either is sig
    for bk_key, bk_name in BENCHMARKS_RAW:
        for feat in FEATURES:
            for h in HORIZONS:
                cknna, sr = get_pairs_rawvlm(raw_rows, feat, h, bk_key)
                rho_s, p_s, r_p, p_p = corr_both(cknna, sr)
                if rho_s is None:
                    continue
                if p_s >= 0.05 and p_p >= 0.05:
                    continue
                labels = get_model_labels_rawvlm(raw_rows, feat, h, bk_key)
                out = os.path.join(scatter_dir,
                                   f"raw_{bk_name}_{FEAT_LABELS[feat]}_h{h}.png")
                make_scatter(cknna, sr, labels, feat, h, bk_name,
                             rho_s, p_s, r_p, p_p, out)
                n_plots += 1

    print(f"Generated {n_plots} scatter plots")

    # ── Disagreement analysis ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  DISAGREEMENT CASES (one sig, other not)")
    print(f"{'='*80}")

    for row in all_rows_out:
        s_sig = row["spearman_sig"]
        p_sig = row["pearson_sig"]
        if s_sig != p_sig:
            which = "Spearman-only" if s_sig else "Pearson-only"
            print(f"  {which}: {row['set']}/{row['benchmark']} "
                  f"{row['feature']} h={row['horizon']}  "
                  f"rho={row['spearman_rho']} p={row['spearman_p']}  "
                  f"r={row['pearson_r']} p={row['pearson_p']}")

    print("\nDone!")


if __name__ == "__main__":
    main()
