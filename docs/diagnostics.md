# Representation-level explanations

[README](../README.md) · [Setup](setup.md) · [Figure commands](../record/README.md)

The paper's §4 uses S² to examine modality dominance, freezing vision encoders, iterative denoising, and layer depth. Run commands from the repository root. The released figure scripts use committed data or embedded result arrays; recomputing features does not automatically update those figures.

## Modality decomposition

For §4.1, the [VLA](reproduce-vla.md) and [VLM](reproduce-vlm.md) guides produce the vision, text, and joint LLM-output features. Reuse their dataset and DTW caches.

The paper finds that, after VLA fine-tuning, task success is associated with vision-feature S² but not text-feature S². At `h=3`, the paper reports VLA correlations of `+0.694` for vision, `+0.699` for joint features, and `+0.237` for text. For raw VLMs, text correlates more strongly than vision, and joint features correlate most strongly.

To render the released comparison:

```bash
conda run -n starVLA python record/fig_4.1_vt_what_carries_2x3/code/regen_what_carries_2x3.py
```

## Frozen vision encoder

For §4.2, the paper compares matched VLM4VLA policies with frozen and trainable vision encoders. Freezing reduces S² by 41% and task success by 40%.

Prepare DROID and its motor cache with the [VLA guide](reproduce-vla.md). Skip those steps if the same subset is already present. Skip checkpoint downloads below if their directories already contain the required files.

```bash
# Download the two VLM4VLA freeze-ablation checkpoints (each ~12 GB).
VLM4VLA_DIR="${WORK}/checkpoints"
mkdir -p "$VLM4VLA_DIR"
for tag in step10k freezevis; do
    conda run -n starVLA hf download "yunfeixie/vlm4vla-qwen25vl3b-bridge-$tag" \
        --local-dir "${VLM4VLA_DIR}/vlm4vla-qwen25vl3b-bridge-$tag"
done

# Feature extraction. Each HF bundle ships one `*.fp32.pt` file; auto-pick it.
# --base_vlm_path takes the bare HuggingFace ID, no local copy needed.
for tag in step10k freezevis; do
    local_dir="${VLM4VLA_DIR}/vlm4vla-qwen25vl3b-bridge-$tag"
    ckpt_pt=$(ls "$local_dir"/*.fp32.pt 2>/dev/null | head -1)
    conda run -n starVLA python extractors/extract_7feat_vlm4vla.py \
        --ckpt_path     "$ckpt_pt" \
        --base_vlm_path Qwen/Qwen2.5-VL-3B-Instruct \
        --data_dir      "$DATA_STORE/cknna_data_droid" \
        --output_dir    "$DATA_STORE/cknna_data_droid_7feat/vlm4vla-qwen25vl3b-bridge-$tag"
done
# Score the two extracted models using the DROID motor cache from VLA-SR.
conda run -n starVLA python compute/compute_cknna.py \
    --data_dir "$DATA_STORE/cknna_data_droid" \
    --feat_dir "$DATA_STORE/cknna_data_droid_7feat" \
    --dtw_cache_dir "$DATA_STORE/record/dtw_cache/droid" \
    --output_csv "$DATA_STORE/results/cknna_freeze_ablation.csv" \
    --horizons 7 --model_filter '^vlm4vla-qwen25vl3b-bridge-(step10k|freezevis)$'


# Figures: bar chart
conda run -n starVLA python record/fig_4.2_freeze_vs_unfreeze/code/freeze-vs-unfreeze-bars.py
```

The bar-chart script contains fixed result values. Compare a new scoring run with those values before updating a copy of the plotting script.

## WAM denoising and layer depth

These correspond to **§4.3 and §4.4**, respectively. The historical folder name `fig_4.2_4.3_pa_va_layers` remains unchanged.

The paper samples five positions along each model's native denoising schedule, rather than matching noise levels across models. S² varies by about 6% at most from its initial value. Block 14 has the highest schedule-averaged S² among the sampled layers in each of the three WAMs.

Render the released plots:

```bash
conda run -n starVLA python record/fig_4.2_4.3_pa_va_layers/code/regen_pa_vs_denoising_pct.py
conda run -n starVLA python record/fig_4.2_4.3_pa_va_layers/code/regen_pa_layerwise.py
```

Recomputing these scores requires separate RoboTwin rollout feature and trajectory caches. `analysis/compute_cknna_denoise_trajectory_3model.py` accepts `--motus_root`, `--lbva_root` plus `--lbva_dense_root`, and `--vidar_server_root` plus `--vidar_client_root`. An invocation with only `--out` supplies no model inputs. The offline ALOHA features from the WAM guide do not replace these rollout caches.

Inspect the expected cache formats and arguments before recomputation:

```bash
conda run -n starVLA python analysis/compute_cknna_denoise_trajectory_3model.py --help
```

## Historical vision-encoder experiment

The release retains a standalone-encoder/ImageNet analysis under `record/fig_appendix_paradigm_imagenet`. **It is not part of the current paper.** It is also distinct from the current appendix experiment on 14 frozen backbones and downstream policy learning, for which this release lacks a matching result bundle.

For the retained experiment, prepare DROID and its motor cache with the [VLA guide](reproduce-vla.md), or reuse the same existing caches. The two extraction scripts cover different encoder families; the explicit JEPA selection below contains seven model IDs. Inspect their available model lists before interpreting an aggregate cohort count.

```bash
# Feature extraction (a): 15 non-JEPA ViTs (Reconstruction / Self-distillation / Contrastive)
conda run -n starVLA python vision_encoder/extract_finding5_features.py \
    --data_dir    "$DATA_STORE/cknna_data_droid" \
    --output_base "$DATA_STORE/cknna_data_droid_standalone_vit"

# Feature extraction (b): the seven JEPA model IDs selected below.
# This is a separate script because the jepa / vjepa2 repos under repos/ ship
# top-level src/app packages that conflict; --setting 1 runs the same
# tiled-image protocol used by extract_finding5_features.py.
conda run -n starVLA python vision_encoder/extract_jepa_features.py --setting 1 \
    --data_dir    "$DATA_STORE/cknna_data_droid" \
    --output_base "$DATA_STORE/cknna_data_droid_standalone_vit" \
    --models ijepa-vith14 vjepa-v1-vitl vjepa-v1-vith vjepa2-vitl vjepa2-vith vjepa2-vitg vjepa2-1-vitg

# S² scoring: the two extractors write to the same output_base.
conda run -n starVLA python compute/compute_cknna.py \
    --data_dir      "$DATA_STORE/cknna_data_droid" \
    --feat_dir      "$DATA_STORE/cknna_data_droid_standalone_vit" \
    --dtw_cache_dir "$DATA_STORE/record/dtw_cache/droid" \
    --output_csv    "$DATA_STORE/results/cknna_standalone_vit_complete.csv" \
    --horizons 7 --topk 10

# Figures: paradigm-vs-ImageNet scatter (joins S² with published ImageNet top-1)
conda run -n starVLA python record/fig_appendix_paradigm_imagenet/code/paradigm_imagenet_scatter.py
```

The renderer uses its released inputs. A new extraction/scoring run needs a checked input update before it changes the rendered result.

## Experimental context and original references

The frozen-encoder comparison uses a Qwen2.5-VL-3B backbone with an FCDecoder action head, fine-tuned on BridgeDataV2 under the [VLM4VLA recipe](https://arxiv.org/pdf/2601.03309). The matched variants differ in whether the vision encoder is trainable.

The denoising analysis relates to [Fast-WAM](https://arxiv.org/abs/2603.16666), which studies skipping iterative video generation. The layer analysis relates to [Demystifying Video Reasoning](https://arxiv.org/abs/2603.16870). The original rollout analysis described 50 RoboTwin tasks with 10 episodes per task, three WAMs, seven feature positions, and five schedule samples. Recomputing it requires the corresponding rollout caches, not just a destination CSV.

The retained encoder/ImageNet analysis reports per-encoder `r=+0.576` over 22 encoders with published ImageNet scores, and paradigm-mean `r=+0.907` over four groups (two-tailed `p=0.094`). This historical experiment is outside the current paper. Its group-level association does not establish a general within-family scaling law.
