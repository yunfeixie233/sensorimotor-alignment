# fig_appendix_paradigm_imagenet

**Paper section**: §Appendix (app:paradigm-data-size)

**Paper figure (.tex)**: `paradigm-imagenet-scatter.tex (Figure 9)`

**Summary**: S² vs ImageNet top-1 across 22 standalone vision encoders (4 paradigms: Reconstruction-SSL / Self-distillation / Contrastive / JEPA). Per-encoder Pearson r=+0.576 (n=22, p=0.005). Paradigm-mean r=+0.907 (n=4 families).

## Files

**Data** (under `data/`):
- `data/cknna_jepa_realclip.csv`
- `data/cknna_jepa_tiled4890.csv`
- `data/cknna_standalone_vit_complete.csv`
- `data/cknna_standalone_vit_jepa_complete.csv`
- `data/figure_subset.csv`

**Figure-subset rows** (the exact (model, x, y) points plotted in the paper figure):
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
