# fig_3.3_va_sr_aloha_vs_droid

**Paper section**: §3.3 (WAM cross-embodiment robustness)

**Paper figure (.tex)**: `va-sr-aloha-vs-droid.tex`

**Summary**: S²–SR holds on ALOHA (r=+0.999) but not on DROID (r=+0.246) as reference dataset for the 3 WAMs. Indicates that the reference dataset must be embodiment-matched for WAMs.

## Files

**Data** (under `data/`):
- `data/cknna_wam_aloha_complete.csv`
- `data/cknna_wam_droid_complete.csv`
- `data/figure_subset.csv`

**Figure-subset rows** (the exact (model, x, y) points plotted in the paper figure):
- `data/figure_subset.csv`

**Filter / extraction**: feature_source=V-B7, h=15 on both ALOHA and DROID masters

**Code** (under `code/`):
- `code/regen_va_sr_aloha_vs_droid.py`

## Reproduce the figure

```bash
cd record/fig_3.3_va_sr_aloha_vs_droid
conda run -n starVLA python code/regen_va_sr_aloha_vs_droid.py
```

Output PDFs/PNGs land in this folder.
