# Setup and troubleshooting

[README](../README.md) · [Pipeline reference](reference.md)

For figures only, use the small CPU environment in [Getting started](../README.md#getting-started). Feature extraction and the DTW kernels use NVIDIA CUDA; the model environment snapshots target Linux. Run commands from the repository root in Bash.

## Configure paths

```bash
cp paths.env.example paths.env
```

Edit `paths.env`: set `WORK` to this checkout, choose `DATA_STORE` for generated caches, and set the raw dataset paths for the experiments you will run. Check `CONDA_ROOT` and `HF_HOME` for your machine. Then load the settings:

```bash
set -a
source paths.env
set +a
mkdir -p "$DATA_STORE" "$RECORD_DIR"
```

`paths.env` is ignored by Git. Some checkpoints require accepting a Hugging Face license and authenticating; use `hf auth login` in the relevant environment or supply `HF_TOKEN` through your shell.

## Create the shared environment

Use `starVLA` for dataset preparation, DTW, S² scoring, StarVLA extraction, and raw VLM extraction:

```bash
conda env create -f repro_envs/yaml/starVLA.yaml
conda run -n starVLA python -m pip install adjustText
conda run -n starVLA python tests/test_imports.py
```

Create additional environments only for the model families you will use:

| Model family | Environment |
|---|---|
| StarVLA and raw VLMs | `starVLA` |
| CogACT | `cogact` |
| GR00T-N1.5 / N1.6 | `groot_libero` |
| Pi0 | `pi0fast_env` |
| SpatialVLA | `spatialvla_env` |
| Motus | `motus` |
| LingBot-VA | `lingbot_va` |
| Vidar | `vidar` |
| RoboTwin simulator | `robotwin` |

Each experiment guide includes its environment commands. The packages use different PyTorch, CUDA, Transformers, and Diffusers versions; keep them in their respective environments. See [environment snapshots](../repro_envs/README.md) for version pins.

## Small metric demo

The included 20-episode example needs no model checkpoint or dataset download:

```bash
conda run -n starVLA python compute/example/run_example.py \
    --feats_A compute/example/example_feats.pt \
    --trajs compute/example/example_trajs.pt
```

This command computes DTW on CUDA. `--device cpu` moves the scoring stage to CPU but does not replace the CUDA DTW kernel; CPU scoring also requires a matching precomputed `--dtw_cache`. See [example inputs and metric details](../compute/example/README.md).

## Validate the resources you installed

After configuring and preparing the external datasets:

```bash
conda run -n starVLA python tests/test_paths.py
```

This checks 10 bundled-code paths and 7 external paths, including both raw datasets. Missing datasets can make it fail on a figures-only or single-dataset installation. It is not a prerequisite for the CPU figure example.

For a full model setup, `bash tests/verify_envs.sh` checks all listed environments. After preparing the DROID cache, `bash extractors/run_all7_smoke.sh` runs a 10-sample extraction check across the raw VLMs and requires their dependencies and checkpoint access.

## Conda resolution or package errors

The files under `repro_envs/` are environment snapshots, not guaranteed portable installers. If Conda resolution fails, use the Python, PyTorch, CUDA, and package versions in the matching snapshot to create that environment. Inspect machine-local `file://` dependencies before using a pip requirements file.

- Keep the environment's Transformers version: `cogact` and `groot_libero` use 4.x, while the released `starVLA` snapshot uses 5.3.0.
- Build or install `flash-attn` for the exact PyTorch/CUDA combination. An `undefined symbol` import error commonly indicates a mismatch.
- After installing model dependencies, check `torch.__version__`, `torch.version.cuda`, and `torch.cuda.is_available()` in that environment before extracting features.

## SAPIEN or RoboTwin rendering errors

The released CSV plots and offline reference-dataset extraction do not need a simulator. For separate RoboTwin rollouts, verify that `nvidia-smi` and `vulkaninfo --summary` identify the intended NVIDIA driver and GPU, and install the NVIDIA Vulkan/EGL components required by the upstream RoboTwin instructions. The [WAM guide](reproduce-wam.md) explains which rollout caches are still needed for the denoising analysis.

## A figure or score differs from the paper

Check the [release coverage table](../record/README.md) first. Some released figures predate the current manuscript. For a new S² run, compare the model IDs, feature position, horizon, episode order, and neighbor count with the committed CSV. Use a separate output directory until the values have been checked.

## Pip fallback template

If Conda cannot resolve a model environment, the original fallback creates an environment with that snapshot's Python version, installs its CUDA PyTorch pair, then installs the matching requirements. Set the variables below from the relevant [environment snapshot](../repro_envs/README.md), and review machine-local package references first:

```bash
: "${REPRO_ENV:?Set the environment name from repro_envs}"
: "${REPRO_PYTHON:?Set its Python version}"
: "${REPRO_TORCH:?Set its PyTorch version}"
: "${REPRO_TORCHVISION:?Set its torchvision version}"
: "${REPRO_CUDA_INDEX:?Set its CUDA wheel suffix, for example cu121}"
conda create -n "$REPRO_ENV" "python=$REPRO_PYTHON" -y
conda run -n "$REPRO_ENV" python -m pip install \
    "torch==$REPRO_TORCH" "torchvision==$REPRO_TORCHVISION" \
    --index-url "https://download.pytorch.org/whl/$REPRO_CUDA_INDEX"
conda run -n "$REPRO_ENV" python -m pip install \
    -r "repro_envs/$REPRO_ENV.requirements.txt" --no-cache-dir
```

The earlier README warned that model dependency installation, including `lerobot`, `openvla`, or `flax`, could replace a CUDA PyTorch installation. Treat this as an environment check rather than a claim about every package installation: if the CUDA build changed, reinstall the matching PyTorch pair and rebuild `flash-attn` for that version.

## Hugging Face access check

After accepting the required model licenses and logging in, check the active account:

```bash
conda run -n starVLA hf auth whoami
```

For an environment with the older CLI, use `huggingface-cli whoami`. CogACT's Llama-2 dependency and the PaliGemma models require the relevant access.

## Optional RoboTwin host setup

This is for separate simulator rollouts, not offline ALOHA scoring or figure rendering. Skip cloning and environment creation when they already exist:

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin "${WORK}/RoboTwin"
conda env create -f repro_envs/yaml/robotwin.yaml
```

The original Linux setup copied the NVIDIA EGL vendor configuration into a directory visible to SAPIEN. On hosts with the source file below, the corresponding commands are:

```bash
sudo mkdir -p /usr/share/glvnd/egl_vendor.d
sudo cp /etc/glvnd/egl_vendor.d/10_nvidia.json /usr/share/glvnd/egl_vendor.d/
sudo apt-get install -y libegl1
conda run -n robotwin python -c "import sapien; print(sapien.__version__)"
```

If the source JSON is absent, check the installed NVIDIA EGL driver configuration. The original fallback JSON named `libEGL_nvidia.so.0`; creating that file does not install the driver library. `vulkaninfo --summary` should identify NVIDIA rather than `llvmpipe`, with a driver consistent with `nvidia-smi`.

## Larger metric example

After preparing DROID, CogACT-Base vision features, and the `h=40` DTW cache, the supplied helper can regenerate the full example:

```bash
conda run -n starVLA python compute/example/prepare_example_data.py
conda run -n starVLA python compute/example/run_example.py \
    --feats_A compute/example/full_feats_img.pt \
    --trajs compute/example/full_trajs_h40.pt \
    --dtw_cache compute/example/dtw_topk_h40_k10.npy \
    --device cpu
```

The helper also rewrites the two small fixtures. The full tensors and cache are generated locally; they are not included in a fresh clone. The old README estimated about 130 MB for these files. The small fixtures are about 350 KB.
