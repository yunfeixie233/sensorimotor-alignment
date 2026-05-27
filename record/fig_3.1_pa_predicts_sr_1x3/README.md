# fig_3.1_pa_predicts_sr_1x3

**Paper section**: §3.1 (sec:finding-pa-predicts-across-families)

**Paper figure (.tex)**: `pa-predicts-sr-1x3.tex (Figure 1, 3-panel headline)`

**Summary**: Sensorimotor alignment (S²) predicts closed-loop success across 3 model families: 12 finetuned VLAs (r=+0.717), 8 raw VLMs (r=+0.681), 3 WAMs (r=+1.000). Single scatter with three panels.

## Files

**Data** (under `data/`):
- `data/benchmark_sr.csv`
- `data/cknna_vla_droid_complete.csv`
- `data/cknna_vlm_droid_complete.csv`
- `data/cknna_wam_aloha_complete.csv`
- `data/figure_subset.csv`
- `data/widowx_sr.csv`

**Figure-subset rows** (the exact (model, x, y) points plotted in the paper figure):
- `data/figure_subset.csv`

**Filter / extraction**: VLA: feats_A_variant=imgtext, h=7 (12 rows) · VLM: feats_A_variant=imgtext, h=3 (8 rows) · WAM: feature_source=V-B7, h=15 (3 rows)

**Code** (under `code/`):
- `code/combined_3benchmarks_scatter.py`

## Reproduce the figure

```bash
cd record/fig_3.1_pa_predicts_sr_1x3
conda run -n starVLA python code/combined_3benchmarks_scatter.py
```

Output PDFs/PNGs land in this folder.
