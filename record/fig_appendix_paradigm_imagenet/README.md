# fig_appendix_paradigm_imagenet

[Result index](../README.md) · [Main README](../../README.md)

**Current-paper location:** Historical experiment, outside the current paper

**Release coverage:** Standalone-encoder/ImageNet comparison. Distinct from the current 14-backbone policy-learning appendix.

The directory name is historical. See the [coverage table](../README.md#coverage-and-current-paper-differences) before using these plots as current-paper reproductions.

## Files

**Data** (under `data/`):
- `data/cknna_jepa_realclip.csv`
- `data/cknna_jepa_tiled4890.csv`
- `data/cknna_standalone_vit_complete.csv`
- `data/cknna_standalone_vit_jepa_complete.csv`
- `data/figure_subset.csv`

**Figure-subset rows** (the selected (model, x, y) points in this release):
- `data/figure_subset.csv`

**Filter / extraction**: COHORT list hardcoded in script (22 encoder × 4-paradigm cells); complete CSVs hold full per-(model, feature, horizon) S²

**Code** (under `code/`):
- `code/paradigm_imagenet_scatter.py`

## Reproduce the figure

```bash
cd record/fig_appendix_paradigm_imagenet
conda run -n starVLA python code/paradigm_imagenet_scatter.py
```

Output PDFs/PNGs land in this folder.
