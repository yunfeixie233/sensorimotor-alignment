# `record/` — Per-figure data and figure-regeneration scripts

Each subfolder corresponds to a single figure in the paper. The directory layout is uniform:

```
record/
├── fig_<section>_<slug>/
│   ├── README.md           ← paper section, figure label, summary, exact filter
│   ├── data/
│   │   ├── cknna_*_complete.csv   ← all models × all features × all horizons
│   │   ├── <SR or metric>.csv     ← success-rate / benchmark inputs
│   │   └── figure_subset.csv      ← exactly the (model, x, y) rows plotted in the paper
│   ├── code/
│   │   └── <regen_script>.py      ← reads data/ → produces figure.pdf/png
│   └── <figure>.{pdf,png}        ← rendered after running the script
├── shared/
│   ├── google_palette.py
│   └── scatter_labels.py
└── README.md (this file)
```

## Figure index

| Folder | Paper section | Paper figure | Headline number |
|---|---|---|---|
| `fig_3.1_pa_predicts_sr_1x3/` | §3.1 | `pa-predicts-sr-1x3.tex` | VLA r=+0.717, VLM r=+0.681, WAM r=+1.000 |
| `fig_3.1_cross_embodiment/` | §3.1 | `cross-embodiment-vla.tex` | DROID +0.717, BridgeV2 +0.699 |
| `fig_3.1_horizon_vla/` | §3.1 | `horizon-vla-llmout-vt.tex` | r ∈ [+0.61, +0.72] across all 10 horizons |
| `fig_3.1_k_sweep/` | §3.1 + appendix | `fig_cknna_r_vs_k_droid.pdf` | r ∈ [+0.644, +0.717] for k ∈ {2,5,10,15,20} |
| `fig_3.2_pa_vs_vqa_simpler/` | §3.2 | `pa-vs-vqa-simpler.tex` | S² r=+0.681; VQA r=−0.154 |
| `fig_3.3_pa_vs_video_quality/` | §3.3 | `pa-vs-video-quality.tex` | S² r=+0.994; SSIM/PSNR/FID p>0.05 |
| `fig_3.3_va_sr_aloha_vs_droid/` | §3.3 | `va-sr-aloha-vs-droid.tex` | ALOHA +0.999; DROID +0.246 |
| `fig_4.1_vt_what_carries_2x3/` | §4.1 | `vt-what-carries-2x3.tex` | V+T strongest in both cohorts |
| `fig_4.2_freeze_vs_unfreeze/` | §4.2 | `freeze-vs-unfreeze-bars.tex` | S² −41 %, SR −40 % |
| `fig_4.2_4.3_pa_va_layers/` | §4.2 + §4.3 | `pa-va-layers.tex` | Denoise drift ≤ 6 %; Block 14 peaks |
| `fig_appendix_paradigm_imagenet/` | Appendix | `paradigm-imagenet-scatter.tex` | per-encoder r=+0.576; paradigm-mean r=+0.907 |

## What is "complete" vs "subset"?

- **Complete CSV** (e.g. `cknna_vla_droid_complete.csv`): all models × all 7 feature positions × all 10 horizons (typically ~400 rows). Useful for re-deriving any slice.
- **`figure_subset.csv`**: exactly the rows plotted in the paper figure (after the figure's filter and join with SR data). Useful for at-a-glance verification — open in Excel and compare against the paper.

## Reproduce all figures from the committed CSVs

You do not need to re-run feature extraction, DTW, or S² scoring — every CSV needed is already in this tree. From the repo root:

```bash
for d in record/fig_*/; do
    for s in "$d"/code/*.py; do
        # Skip sweep_k_droid.py — it is a heavy GPU re-compute of the §3.1
        # k-sweep S² scoring, not a render. The render script in the same
        # folder (render_figures.py) already consumes its committed CSV.
        [ "$(basename "$s")" = "sweep_k_droid.py" ] && continue
        conda run -n starVLA python "$s"
    done
done
```

Each script writes its PDF/PNG into its own folder.

## Compute scripts

The Stage-3 motor-kernel (DTW) and Stage-4 S²-scoring (CKNNA) compute code lives at top-level `compute/` (not under `record/`). `record/` holds only data and figure-regeneration code.

