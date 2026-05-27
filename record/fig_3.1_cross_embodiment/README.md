# fig_3.1_cross_embodiment

**Paper section**: §3.1 (cross-embodiment robustness)

**Paper figure (.tex)**: `cross-embodiment-vla.tex`

**Summary**: S²–SR correlation holds across two reference datasets (DROID and BridgeV2). DROID r=+0.717 at h=7; BridgeV2 r=+0.699 at h=15. Confirms the predictive signal is dataset-agnostic.

## Files

**Data** (under `data/`):
- `data/cknna_vla_bridgev2_complete.csv`
- `data/cknna_vla_droid_complete.csv`
- `data/figure_subset.csv`
- `data/widowx_sr.csv`

**Figure-subset rows** (the exact (model, x, y) points plotted in the paper figure):
- `data/figure_subset.csv`

**Filter / extraction**: feats_A_variant=imgtext on both DROID and BridgeV2, per-horizon Pearson r over 12 VLAs

**Code** (under `code/`):
- `code/regen_cross_embodiment_vla.py`

## Reproduce the figure

```bash
cd record/fig_3.1_cross_embodiment
conda run -n starVLA python code/regen_cross_embodiment_vla.py
```

Output PDFs/PNGs land in this folder.
