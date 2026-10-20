# fig_4.2_4.3_pa_va_layers

[Result index](../README.md) · [Main README](../../README.md)

**Current-paper location:** §4.3–4.4; denoising and layer depth

**Release coverage:** Five positions along each native schedule, not matched noise levels; Block 14 has the highest schedule-averaged S². Renderers embed their values.

The directory name is historical. See the [coverage table](../README.md#coverage-and-current-paper-differences) before using these plots as current-paper reproductions.

## Files

**Data** (under `data/`):
- `data/cknna_denoise_3model_complete.csv`
- `data/figure_subset.csv`

**Figure-subset rows** (the selected (model, x, y) points in this release):
- `data/figure_subset.csv`

**Filter / extraction**: Hardcoded SIMA dicts in scripts; complete CSV has cknna at 8 features × 5 denoise checkpoints × 3 WAMs (120 rows)

**Code** (under `code/`):
- `code/regen_pa_layerwise.py`
- `code/regen_pa_vs_denoising_pct.py`

## Reproduce the figure

```bash
cd record/fig_4.2_4.3_pa_va_layers
conda run -n starVLA python code/regen_pa_layerwise.py
conda run -n starVLA python code/regen_pa_vs_denoising_pct.py
```

Output PDFs/PNGs land in this folder.
