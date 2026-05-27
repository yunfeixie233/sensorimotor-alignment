# fig_3.3_pa_vs_video_quality

**Paper section**: §3.3 (sec:finding-va-sr)

**Paper figure (.tex)**: `pa-vs-video-quality.tex (Figure 3, 1×4 composite)`

**Summary**: S² predicts WAM success rate on RoboTwin (r=+0.994), while per-frame video-quality metrics (SSIM r=+0.476, PSNR r=+0.381, FID r=+0.747) do not — none reach p<0.05 at n=3. 3 WAMs: Motus, LingBot-VA, Vidar.

## Files

**Data** (under `data/`):
- `data/cknna_wam_aloha_complete.csv`
- `data/figure_subset.csv`

**Figure-subset rows** (the exact (model, x, y) points plotted in the paper figure):
- `data/figure_subset.csv`

**Filter / extraction**: feature_source=V-B7, horizon=15 in master_aloha (3 rows); SSIM/PSNR/FID/SR inline in script

**Code** (under `code/`):
- `code/regen_pa_vs_video_quality.py`

## Reproduce the figure

```bash
cd record/fig_3.3_pa_vs_video_quality
conda run -n starVLA python code/regen_pa_vs_video_quality.py
```

Output PDFs/PNGs land in this folder.
