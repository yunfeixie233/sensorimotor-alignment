# fig_4.2_freeze_vs_unfreeze

[Result index](../README.md) · [Main README](../../README.md)

**Current-paper location:** §4.2; frozen vision encoder

**Release coverage:** S² decreases by 41% and success rate by 40%. The renderer embeds its values.

The directory name is historical. See the [coverage table](../README.md#coverage-and-current-paper-differences) before using these plots as current-paper reproductions.

## Files

**Data** (under `data/`):
- `data/cknna_vla_droid_complete.csv`
- `data/figure_subset.csv`
- `data/widowx_sr.csv`

**Figure-subset rows** (the selected (model, x, y) points in this release):
- `data/figure_subset.csv`

**Filter / extraction**: Two specific Qwen2.5-VL-3B VLAs at imgtext h=7; values hardcoded in script

**Code** (under `code/`):
- `code/freeze-vs-unfreeze-bars.py`

## Reproduce the figure

```bash
cd record/fig_4.2_freeze_vs_unfreeze
conda run -n starVLA python code/freeze-vs-unfreeze-bars.py
```

Output PDFs/PNGs land in this folder.
