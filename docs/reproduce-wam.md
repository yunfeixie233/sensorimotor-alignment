# WAM results and reproduction coverage

[README](../README.md) · [Setup](setup.md) · [Figure commands](../record/README.md)

The current paper's §3.1 evaluates **6 WAMs** on ALOHA against RoboTwin 2.0 success: Vidar, LingBot-VA, Motus, Fast-WAM, FlowWAM, and Kairos. It reports `r=+0.718` using video-backbone Block 7, `h=15`, and `k=10`.

The released result bundle covers **3 WAMs**: Motus, LingBot-VA, and Vidar. These also form the paper's §3.2 video-quality comparison. Its reported S² correlation is `r=+0.999`; the released scripts use older inputs and do not exactly reproduce all current-paper numbers. The [result index](../record/README.md) identifies this distinction.

## Regenerate the released plots

Use the CPU figure environment from [Getting started](../README.md#getting-started), then run from the repository root:

```bash
python record/fig_3.3_pa_vs_video_quality/code/regen_pa_vs_video_quality.py
python record/fig_3.3_va_sr_aloha_vs_droid/code/regen_va_sr_aloha_vs_droid.py
```

These commands use the committed tables and require no new data, features, or simulator rollouts.

## Preparing to recompute WAM features

Full reproduction from raw data is incomplete in this release. The steps below describe available components and their prerequisites; they are not a complete six-model recipe.

After [Setup](setup.md), create only the environments you need:

```bash
for env in motus lingbot_va vidar; do
    conda env create -f "repro_envs/yaml/$env.yaml"
done
```

## ALOHA data

Skip downloads already present under `$WORK/datasets/aloha_lerobot`. Reuse a prepared cache only when it preserves the original episode order and sampling.

ALOHA bimanual (motor reference). Pulled from HuggingFace; N=839 episodes from 15 `lerobot/aloha_static_*` datasets:

```bash
ALOHA_DIR="${WORK}/datasets/aloha_lerobot" && mkdir -p "$ALOHA_DIR"
for d in coffee coffee_new towel screw_driver fork_pick_up battery \
         vinh_cup vinh_cup_left ziploc_slide candy thread_velcro \
         cups_open pingpong_test pro_pencil tape; do
    conda run -n starVLA hf download "lerobot/aloha_static_$d" \
        --repo-type dataset --local-dir "$ALOHA_DIR/aloha_static_$d"
done
```

(The `hf` CLI ships with `huggingface_hub >= 1.0`, which is already in the `starVLA` env. If you use an older env where `huggingface-cli` is the active binary, swap `hf` for `huggingface-cli` — the subcommand syntax is identical.)

The original experiment used 839 episodes after dropping one parquet/video mismatch. Downloading these files does not create the prepared `cknna_data_aloha` cache. The release does not include the subset-conversion script that fixes episode order, selects frames, and writes `metadata.json`, images, and `feats_B_seq.pt`.



## Available extraction code

| Extractor | Input and output |
|---|---|
| [`motus/extract_features.py`](../extractors/motus/extract_features.py) | Prepared images and instructions; pooled features in `feats_A.pt` |
| [`lingbot_va/extract_features.py`](../extractors/lingbot_va/extract_features.py) | Prepared images and instructions; `feats_A.pt`, plus `feats_A_layers.pt` with `--save_layers` |
| [`extract_va_features_vidar_droid.py`](../extractors/extract_va_features_vidar_droid.py) | Prepared multi-camera observations; files including `feats_A_block7.pt`, `feats_A_block14.pt`, and `feats_A_norm.pt` |

These scripts process recorded reference observations offline. They do not themselves run closed-loop RoboTwin evaluations. Their output formats differ; the old instruction that all three emit the same 15 feature files was incorrect.

Use each script's `--help` for checkpoint arguments and the upstream projects for checkpoint layouts: [Motus](https://github.com/jasper0314-huang/Motus), [LingBot-VA](https://github.com/LingBot-AI), and [Vidar](https://github.com/SgtVincent/Vidar). The release still needs a common adapter that selects the paper's Block 7 representation and preserves the ALOHA sample order across models.

## Motor neighbors

Given an already prepared ALOHA cache with `feats_B_seq.pt` and `metadata.json`, the released joint-space DTW command is:

```bash
conda run -n starVLA python compute/compute_dtw_aloha.py     --data_dir "$DATA_STORE/cknna_data_aloha"     --output_dir "$DATA_STORE/record/dtw_cache/aloha"     --horizons 15 --topk 10
```

The script uses Euclidean local distance for the 14-dimensional bimanual trajectories. Check that the filtered episode count and order match the sensory features. For a known unpadded cache, `--no_padding_filter` disables the stationary-frame heuristic; only use it after checking the cache.

`compute/compute_cknna.py` recognizes the VLA/VLM filenames in the [pipeline reference](reference.md#feature-files). It does not discover WAM Block 7 files or emit the WAM `feature_source` schema. Running that generic command unchanged will not reproduce the published WAM table.

## Remaining release work

- Supply the ALOHA preparation script and exact subset manifest.
- Connect the model-specific outputs to a common Block 7 scoring path.
- Add the six-model result bundle and the remaining model recipes.
- Supply the rollout feature/trajectory caches for the denoising and layer analyses; see [diagnostics](diagnostics.md#wam-denoising-and-layer-depth).

The real-robot result is a separate experiment. Its five-policy result bundle and reproduction recipe are also absent from this release.
