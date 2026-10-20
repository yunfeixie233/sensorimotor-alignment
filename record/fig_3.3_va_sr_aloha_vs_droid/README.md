# fig_3.3_va_sr_aloha_vs_droid

[Result index](../README.md) · [Main README](../../README.md)

**Current-paper location:** §3.3 and robustness appendix; embodiment mismatch

**Release coverage:** Three WAMs: matched ALOHA r=+0.999; single-arm DROID r=+0.246.

The directory name is historical. See the [coverage table](../README.md#coverage-and-current-paper-differences) before using these plots as current-paper reproductions.

## Files

**Data** (under `data/`):
- `data/cknna_wam_aloha_complete.csv`
- `data/cknna_wam_droid_complete.csv`
- `data/figure_subset.csv`

**Figure-subset rows** (the selected (model, x, y) points in this release):
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
