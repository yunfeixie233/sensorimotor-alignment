# fig_4.1_vt_what_carries_2x3

[Result index](../README.md) · [Main README](../../README.md)

**Current-paper location:** §4.1; modality

**Release coverage:** Same qualitative finding. Released VLA text r=+0.238 and raw-VLM vision/text r=+0.426/+0.502 differ by 0.001 from the manuscript.

The directory name is historical. See the [coverage table](../README.md#coverage-and-current-paper-differences) before using these plots as current-paper reproductions.

## Files

**Data** (under `data/`):
- `data/benchmark_sr.csv`
- `data/cknna_vla_droid_complete.csv`
- `data/cknna_vlm_droid_complete.csv`
- `data/figure_subset.csv`
- `data/widowx_sr.csv`

**Figure-subset rows** (the selected (model, x, y) points in this release):
- `data/figure_subset.csv`

**Filter / extraction**: VLA: variants {img, txt, imgtext} at h=7 · VLM: variants {img, txt, imgtext} at h=3

**Code** (under `code/`):
- `code/regen_what_carries_2x3.py`

## Reproduce the figure

```bash
cd record/fig_4.1_vt_what_carries_2x3
conda run -n starVLA python code/regen_what_carries_2x3.py
```

Output PDFs/PNGs land in this folder.
