# fig_3.1_k_sweep

**Paper section**: §3.1 (k-sensitivity, app:k-sweep)

**Paper figure (.tex)**: `inline \includegraphics{fig_cknna_r_vs_k_droid.pdf}`

**Summary**: Pearson r vs mutual-kNN neighborhood size k ∈ {2, 5, 10, 15, 20} at h=7. r ∈ [+0.644, +0.717] across all k. Headline uses k=10.

## Files

**Data** (under `data/`):
- `data/figure_subset.csv`
- `data/results_k10.csv`
- `data/results_k15.csv`
- `data/results_k2.csv`
- `data/results_k20.csv`
- `data/results_k5.csv`
- `data/summary_r_p.csv`

**Figure-subset rows** (the exact (model, x, y) points plotted in the paper figure):
- `data/figure_subset.csv`

**Filter / extraction**: summary_r_p.csv filtered to horizon=7 (one row per k)

**Code** (under `code/`):
- `code/render_figures.py`
- `code/sweep_k_droid.py`

## Reproduce the figure

The figure can be re-rendered from the committed CSVs alone (no GPU needed):

```bash
cd record/fig_3.1_k_sweep
conda run -n starVLA python code/render_figures.py
```

To recompute the underlying S² values from scratch (heavy GPU step, ≈1 h on one A100; requires `$DATA_STORE/cknna_data_droid`, the `_7feat` extracted cache, and the DTW caches under `$DATA_STORE/record/dtw_cache/droid` plus a `droid_topk20` cache built via `compute/compute_dtw.py --topk 20`):

```bash
conda run -n starVLA python code/sweep_k_droid.py
```

Output PDFs/PNGs land in this folder.
