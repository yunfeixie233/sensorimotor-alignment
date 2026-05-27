# fig_4.2_4.3_pa_va_layers

**Paper section**: §4.2 (sec:pa-reveals-denoising) + §4.3 (sec:pa-reveals-midlayer)

**Paper figure (.tex)**: `pa-va-layers.tex (Figure 6, two panels: denoise + per-layer)`

**Summary**: Left panel (§4.2): S² is already high at 0% of the native denoising schedule and varies by ≤6% across 0/25/50/75/100% for all 3 WAMs. Right panel (§4.3): S² peaks at video-transformer Block 14 (mid-depth) in all 3 WAMs.

## Files

**Data** (under `data/`):
- `data/cknna_denoise_3model_complete.csv`
- `data/figure_subset.csv`

**Figure-subset rows** (the exact (model, x, y) points plotted in the paper figure):
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
