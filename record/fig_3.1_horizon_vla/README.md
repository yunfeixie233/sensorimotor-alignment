# fig_3.1_horizon_vla

**Paper section**: §3.1 (horizon stability)

**Paper figure (.tex)**: `horizon-vla-llmout-vt.tex`

**Summary**: Pearson r vs DTW trajectory horizon h ∈ [1, 220] on DROID and h ∈ [1, 40] on BridgeV2. Demonstrates that the S²–SR correlation is stable across all measured horizons.

## Files

**Data** (under `data/`):
- `data/cknna_vla_bridgev2_complete.csv`
- `data/cknna_vla_droid_complete.csv`
- `data/figure_subset.csv`
- `data/widowx_sr.csv`

**Figure-subset rows** (the exact (model, x, y) points plotted in the paper figure):
- `data/figure_subset.csv`

**Filter / extraction**: feats_A_variant=imgtext, per-horizon Pearson r over 12 VLAs

**Code** (under `code/`):
- `code/regen_horizon_vla_llmout_vt.py`

## Reproduce the figure

```bash
cd record/fig_3.1_horizon_vla
conda run -n starVLA python code/regen_horizon_vla_llmout_vt.py
```

Output PDFs/PNGs land in this folder.
