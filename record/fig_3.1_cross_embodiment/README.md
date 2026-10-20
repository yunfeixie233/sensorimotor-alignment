# fig_3.1_cross_embodiment

[Result index](../README.md) · [Main README](../../README.md)

**Current-paper location:** §3.3; reference datasets

**Release coverage:** Released plots give DROID r=+0.717 and BridgeDataV2 r=+0.699.

The directory name is historical. See the [coverage table](../README.md#coverage-and-current-paper-differences) before using these plots as current-paper reproductions.

## Files

**Data** (under `data/`):
- `data/cknna_vla_bridgev2_complete.csv`
- `data/cknna_vla_droid_complete.csv`
- `data/figure_subset.csv`
- `data/widowx_sr.csv`

**Figure-subset rows** (the selected (model, x, y) points in this release):
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
