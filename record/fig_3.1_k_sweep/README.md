# fig_3.1_k_sweep

[Result index](../README.md) · [Main README](../../README.md)

**Current-paper location:** Robustness appendix; neighbor count

**Release coverage:** Released k-sweep analysis; summary uses h=7. The raw GPU sweep is separate from figure rendering.

The directory name is historical. See the [coverage table](../README.md#coverage-and-current-paper-differences) before using these plots as current-paper reproductions.

## Files

**Data** (under `data/`):
- `data/figure_subset.csv`
- `data/results_k10.csv`
- `data/results_k15.csv`
- `data/results_k2.csv`
- `data/results_k20.csv`
- `data/results_k5.csv`
- `data/summary_r_p.csv`

**Figure-subset rows** (the selected (model, x, y) points in this release):
- `data/figure_subset.csv`

**Filter / extraction**: summary_r_p.csv filtered to horizon=7 (one row per k)

**Code** (under `code/`):
- `code/render_figures.py`
- `code/sweep_k_droid.py`

## Reproduce the figure

The released figure can be rendered from the bundled inputs (no GPU needed):

```bash
cd record/fig_3.1_k_sweep
conda run -n starVLA python code/render_figures.py
```

To recompute the underlying S² values from scratch (heavy GPU step, ≈1 h on one A100; requires `$DATA_STORE/cknna_data_droid`, the `_7feat` extracted cache, and the DTW caches under `$DATA_STORE/record/dtw_cache/droid` plus a `droid_topk20` cache built via `compute/compute_dtw.py --topk 20`):

```bash
conda run -n starVLA python code/sweep_k_droid.py
```

Output PDFs/PNGs land in this folder.
