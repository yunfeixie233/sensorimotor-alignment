# fig_3.1_horizon_vla

[Result index](../README.md) · [Main README](../../README.md)

**Current-paper location:** §3.3; trajectory horizons

**Release coverage:** DROID r ranges from +0.610 to +0.717. BridgeDataV2 exceeds +0.5 at h=3 through 40; h=1 is lower.

The directory name is historical. See the [coverage table](../README.md#coverage-and-current-paper-differences) before using these plots as current-paper reproductions.

## Files

**Data** (under `data/`):
- `data/cknna_vla_bridgev2_complete.csv`
- `data/cknna_vla_droid_complete.csv`
- `data/figure_subset.csv`
- `data/widowx_sr.csv`

**Figure-subset rows** (the selected (model, x, y) points in this release):
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
