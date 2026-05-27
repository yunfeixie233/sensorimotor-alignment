# fig_3.2_pa_vs_vqa_simpler

**Paper section**: §3.2 (sec:finding-vlm-predict-vla)

**Paper figure (.tex)**: `pa-vs-vqa-simpler.tex (Figure 2)`

**Summary**: Two-panel scatter showing S² on a raw VLM predicts the resulting VLA's SimplerEnv success rate (r=+0.681, p=0.031), while the VLM's general VQA capability does NOT (r=−0.154, p=0.642). 8 raw VLMs (KosMos-2 + PaliGemma 1/2 + Qwen2.5VL 3B/7B + Qwen3VL 2B/4B/8B).

## Files

**Data** (under `data/`):
- `data/benchmark_sr.csv`
- `data/cknna_vlm_droid_complete.csv`
- `data/figure_subset.csv`

**Figure-subset rows** (the exact (model, x, y) points plotted in the paper figure):
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
