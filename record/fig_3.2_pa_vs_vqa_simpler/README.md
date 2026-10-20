# fig_3.2_pa_vs_vqa_simpler

[Result index](../README.md) · [Main README](../../README.md)

**Current-paper location:** §3.2; VQA comparison

**Release coverage:** S² r=+0.681; VQA r=−0.154 for the 8 raw backbones.

The directory name is historical. See the [coverage table](../README.md#coverage-and-current-paper-differences) before using these plots as current-paper reproductions.

## Files

**Data** (under `data/`):
- `data/benchmark_sr.csv`
- `data/cknna_vlm_droid_complete.csv`
- `data/figure_subset.csv`

**Figure-subset rows** (the selected (model, x, y) points in this release):
- `data/figure_subset.csv`

**Filter / extraction**: feats_A_variant=imgtext, h=3 (8 rows)

**Code** (under `code/`):
- `code/regen_pa_vs_vqa_simpler.py`

## Reproduce the figure

```bash
cd record/fig_3.2_pa_vs_vqa_simpler
conda run -n starVLA python code/regen_pa_vs_vqa_simpler.py
```

Output PDFs/PNGs land in this folder.
