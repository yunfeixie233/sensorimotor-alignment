# Released results and figure regeneration

[README](../README.md) · [Setup](../docs/setup.md) · [Experiment guides](../README.md#reproduction-guides)

This directory contains result tables and plot scripts from an earlier code release. Folder names retain their historical section numbers so existing paths keep working. Use the table below to find their role in the current manuscript.

## Coverage and current-paper differences

| Folder | Current-paper location | Coverage |
|---|---|---|
| [fig_3.1_pa_predicts_sr_1x3](fig_3.1_pa_predicts_sr_1x3/README.md) | §3.1; earlier three-panel version | VLA and raw-VLM panels match the reported correlations. The WAM panel contains only 3 models; the current paper uses 6 WAMs and adds a real-robot panel. |
| [fig_3.1_cross_embodiment](fig_3.1_cross_embodiment/README.md) | §3.3; reference datasets | Released plots give DROID r=+0.717 and BridgeDataV2 r=+0.699. |
| [fig_3.1_horizon_vla](fig_3.1_horizon_vla/README.md) | §3.3; trajectory horizons | DROID r ranges from +0.610 to +0.717. BridgeDataV2 exceeds +0.5 at h=3 through 40; h=1 is lower. |
| [fig_3.1_k_sweep](fig_3.1_k_sweep/README.md) | Robustness appendix; neighbor count | Released k-sweep analysis; summary uses h=7. The raw GPU sweep is separate from figure rendering. |
| [fig_3.2_pa_vs_vqa_simpler](fig_3.2_pa_vs_vqa_simpler/README.md) | §3.2; VQA comparison | S² r=+0.681; VQA r=−0.154 for the 8 raw backbones. |
| [fig_3.3_pa_vs_video_quality](fig_3.3_pa_vs_video_quality/README.md) | §3.2; video-quality comparison | Older inputs: released S²/SSIM/PSNR/FID r=+0.994/+0.476/+0.381/+0.747; current paper reports +0.999/+0.558/+0.468/+0.680. |
| [fig_3.3_va_sr_aloha_vs_droid](fig_3.3_va_sr_aloha_vs_droid/README.md) | §3.3 and robustness appendix; embodiment mismatch | Three WAMs: matched ALOHA r=+0.999; single-arm DROID r=+0.246. |
| [fig_4.1_vt_what_carries_2x3](fig_4.1_vt_what_carries_2x3/README.md) | §4.1; modality | Same qualitative finding. Released VLA text r=+0.238 and raw-VLM vision/text r=+0.426/+0.502 differ by 0.001 from the manuscript. |
| [fig_4.2_freeze_vs_unfreeze](fig_4.2_freeze_vs_unfreeze/README.md) | §4.2; frozen vision encoder | S² decreases by 41% and success rate by 40%. The renderer embeds its values. |
| [fig_4.2_4.3_pa_va_layers](fig_4.2_4.3_pa_va_layers/README.md) | §4.3–4.4; denoising and layer depth | Five positions along each native schedule, not matched noise levels; Block 14 has the highest schedule-averaged S². Renderers embed their values. |
| [fig_appendix_paradigm_imagenet](fig_appendix_paradigm_imagenet/README.md) | Historical experiment, outside the current paper | Standalone-encoder/ImageNet comparison. Distinct from the current 14-backbone policy-learning appendix. |

The full six-WAM result, five-policy real-robot result, and 14-backbone policy-learning result do not yet have matching reproduction bundles here. The combined scatter therefore does not recreate the current four-panel paper figure.

The published paper's real-robot correlation uses an exploratory horizon of `h=110`; it should not be compared with an `h=7` run as though both used the same setting.

## Read the tables

Each figure folder lists its inputs and selection rules. A `cknna_*_complete.csv` contains the released scores for that experiment; `figure_subset.csv` records a selected plotting subset. Neither filename guarantees coverage of every experiment in the current paper. Some scripts also embed success rates or other result values.

For example:

- [VLA S² scores](fig_3.1_pa_predicts_sr_1x3/data/cknna_vla_droid_complete.csv) and [model IDs / success rates](fig_3.1_pa_predicts_sr_1x3/data/widowx_sr.csv).
- [Raw-VLM S² scores](fig_3.2_pa_vs_vqa_simpler/data/cknna_vlm_droid_complete.csv).
- [Released WAM S² scores](fig_3.3_pa_vs_video_quality/data/cknna_wam_aloha_complete.csv).

## Render on CPU

Use the environment from [Getting started](../README.md#getting-started). From the repository root, choose a script:

```bash
# §3.1; earlier three-panel version
python record/fig_3.1_pa_predicts_sr_1x3/code/combined_3benchmarks_scatter.py

# §3.3; reference datasets
python record/fig_3.1_cross_embodiment/code/regen_cross_embodiment_vla.py

# §3.3; trajectory horizons
python record/fig_3.1_horizon_vla/code/regen_horizon_vla_llmout_vt.py

# Robustness appendix; neighbor count
python record/fig_3.1_k_sweep/code/render_figures.py

# §3.2; VQA comparison
python record/fig_3.2_pa_vs_vqa_simpler/code/regen_pa_vs_vqa_simpler.py

# §3.2; video-quality comparison
python record/fig_3.3_pa_vs_video_quality/code/regen_pa_vs_video_quality.py

# §3.3 and robustness appendix; embodiment mismatch
python record/fig_3.3_va_sr_aloha_vs_droid/code/regen_va_sr_aloha_vs_droid.py

# §4.1; modality
python record/fig_4.1_vt_what_carries_2x3/code/regen_what_carries_2x3.py

# §4.2; frozen vision encoder
python record/fig_4.2_freeze_vs_unfreeze/code/freeze-vs-unfreeze-bars.py

# §4.3–4.4; denoising and layer depth
python record/fig_4.2_4.3_pa_va_layers/code/regen_pa_vs_denoising_pct.py

python record/fig_4.2_4.3_pa_va_layers/code/regen_pa_layerwise.py

# Historical experiment, outside the current paper
python record/fig_appendix_paradigm_imagenet/code/paradigm_imagenet_scatter.py

```

Each script writes PDFs and, where implemented, PNG previews into its figure folder. These commands render the released inputs; they do not run model extraction, DTW, or S² recomputation. `sweep_k_droid.py` is a separate GPU computation and is intentionally absent from this list.

## Use newly computed scores

The experiment guides write new scores to `$DATA_STORE/results/` to preserve the released tables. Plot scripts read their own `data/` files or embedded arrays; they do not pick up a new score file automatically.

Compare the model IDs, feature labels, horizons, episode counts, and scores first. Then update inputs in a separate checkout or copy of the figure folder. The same score table can appear in multiple figure directories, so update each intended consumer explicitly.
