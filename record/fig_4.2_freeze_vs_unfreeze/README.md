# fig_4.2_freeze_vs_unfreeze

**Paper section**: §4.2 (sec:pa-reveals-frozen)

**Paper figure (.tex)**: `freeze-vs-unfreeze-bars.tex (Figure 5)`

**Summary**: Two-panel bar chart: frozen vision encoder reduces S² by 41% (0.0245 → 0.0144) and downstream SimplerEnv SR by 40% (50.0 → 30.2). Demonstrates that VLA fine-tuning improves SR by reshaping the vision encoder toward stronger sensorimotor alignment.

## Files

**Data** (under `data/`):
- `data/cknna_vla_droid_complete.csv`
- `data/figure_subset.csv`
- `data/widowx_sr.csv`

**Figure-subset rows** (the exact (model, x, y) points plotted in the paper figure):
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
