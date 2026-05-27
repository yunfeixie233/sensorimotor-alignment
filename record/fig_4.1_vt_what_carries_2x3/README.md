# fig_4.1_vt_what_carries_2x3

**Paper section**: §4.1 (sec:finding-what-carries) — modality dominance

**Paper figure (.tex)**: `vt-what-carries-2x3.tex (Figure 4, 2×3 panel matrix)`

**Summary**: Decomposes the LLM-out feature into vision-only / text-only / vision+text. Top row: 12 finetuned VLAs; bottom row: 8 raw VLMs. Vision+text is the strongest predictor in both cohorts.

## Files

**Data** (under `data/`):
- `data/benchmark_sr.csv`
- `data/cknna_vla_droid_complete.csv`
- `data/cknna_vlm_droid_complete.csv`
- `data/figure_subset.csv`
- `data/widowx_sr.csv`

**Figure-subset rows** (the exact (model, x, y) points plotted in the paper figure):
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
