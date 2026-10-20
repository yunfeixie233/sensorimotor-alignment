# fig_3.3_pa_vs_video_quality

[Result index](../README.md) · [Main README](../../README.md)

**Current-paper location:** §3.2; video-quality comparison

**Release coverage:** Older inputs: released S²/SSIM/PSNR/FID r=+0.994/+0.476/+0.381/+0.747; current paper reports +0.999/+0.558/+0.468/+0.680.

The directory name is historical. See the [coverage table](../README.md#coverage-and-current-paper-differences) before using these plots as current-paper reproductions.

## Files

**Data** (under `data/`):
- `data/cknna_wam_aloha_complete.csv`
- `data/figure_subset.csv`

**Figure-subset rows** (the selected (model, x, y) points in this release):
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
