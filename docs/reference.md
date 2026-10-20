# Pipeline reference

[README](../README.md) · [Setup](setup.md)

## Metric and inputs

S² compares sensory and motor neighborhoods over the same ordered episodes. Sensory similarity uses cosine similarity on pooled model features. For the single-arm reference data, motor distance uses DTW with position L2, SO(3) rotation-geodesic, and gripper L1 costs. The ALOHA kernel uses Euclidean distance on 14-dimensional bimanual trajectories.

The implementation uses CKNNA: it retains sensory similarity weights, represents motor neighbors with a binary mask, applies shared-neighbor masks, and normalizes an unbiased HSIC estimate. S² is therefore not just the fraction of overlapping neighbors. The separately emitted `mutual_knn_k10` column is that overlap fraction.

Keep the episode order identical across images, instructions, features, and trajectories. Reuse a motor cache only with the same subset, ordering, trajectory representation, horizon, neighbor count, and filtering settings.

## Feature files

The VLA and raw-VLM scorer reads these seven filenames:

| File | Feature position | CSV variant |
|---|---|---|
| `feats_A_vit.pt` | Vision encoder output | `vit_raw` |
| `feats_A_pre_llm.pt` | LLM input: vision | `pre_llm` |
| `feats_A_pre_llm_txt.pt` | LLM input: instruction | `pre_llm_txt` |
| `feats_A_pre_llm_vt.pt` | LLM input: vision and instruction | `pre_llm_vt` |
| `feats_A_img.pt` | LLM output: vision | `img` |
| `feats_A_txt.pt` | LLM output: instruction | `txt` |
| `feats_A.pt` | LLM output: vision and instruction | `imgtext` |

The VLA result uses `imgtext` at `h=7`; the raw-VLM result uses it at `h=3`. The `_11feat` raw-VLM cache name is historical and does not change the seven-file interface. WAM extractors have different formats; see the [WAM guide](reproduce-wam.md#available-extraction-code).

## Cache layout

Paths are relative to `DATA_STORE` from `paths.env`, except the released tables under `record/`. Counts below describe the original cached experiments.

| Cache directory                                         | Contents                                                            | Stage |
|---------------------------------------------------------|---------------------------------------------------------------------|-------|
| `$DATA_STORE/cknna_data_droid/`                      | DROID N=6000: images, `feats_B*.pt`, `actions*.pt`, `metadata.json` | Dataset preparation |
| `$DATA_STORE/cknna_data_bridge/`           | BridgeV2 N=7762: feats_B + sequences                                | Dataset preparation |
| `$DATA_STORE/cknna_data_aloha/`                      | ALOHA N=839: 14-D bimanual feats_B                                  | Dataset preparation |
| `$DATA_STORE/cknna_data_droid_7feat/<model>/`        | 7 feature files per VLA                                   | Feature extraction |
| `$DATA_STORE/cknna_data_droid_11feat/<model>/`       | 7 feature files per raw VLM (historical `_11feat` directory name)                              | Feature extraction |
| `$DATA_STORE/cknna_data_droid_15feat/<model>/`       | Model-specific WAM features; formats vary                                        | Feature extraction |
| `$DATA_STORE/cknna_data_droid_standalone_vit/`       | Features for additional standalone encoders (historical experiment)                   | Feature extraction |
| `$DATA_STORE/cknna_data_bridge_7feat/<model>/` | BridgeV2 feats_A                                                    | Feature extraction |
| `$DATA_STORE/cknna_data_aloha_15feat/<model>/`       | ALOHA feats_A for WAMs                                              | Feature extraction |
| `$DATA_STORE/record/dtw_cache/{droid,bridge,aloha}/`    | `dtw_topk_h{H}_k{K}_sym2_cuda_nowin_nopad.npy`                      | Motor kernel |
| `record/` (committed)                           | Released S² CSVs and figure scripts                          | S² scoring + Figures |

The [record index](../record/README.md) lists released tables and plots. New reproduction commands write scores to `$DATA_STORE/results/`; the plotting scripts continue to read their own committed inputs until you explicitly update a checked copy.

## Add a model

1. Choose an extractor compatible with the model, or adapt one to emit `[N, D]` feature tensors in the same episode order.
2. Store outputs under a new model directory and score it against the existing matching motor cache.
3. To compare with task success, use success rates from the same evaluation protocol as the comparison cohort.
4. Work on copies of the relevant figure data and model metadata. Check the script's model filters, feature selection, and horizon before plotting.

The released scatter scripts filter model names and cohorts; adding a row does not guarantee that every plot will include it. Running one or two models checks the pipeline but cannot reproduce a correlation reported over the full cohort.

## Resource planning

The original download instructions estimate about 1.7 TB for DROID and 124 GB for BridgeDataV2. Model checkpoints and generated features require additional space. The old README's 46-model / 50-GPU-hour total described a different set of experiments and should not be used as a budget for the current paper.

For development, use the small metric demo or a small extraction subset first. A subset changes the reference distribution and is not an exact reproduction of the reported scores.
