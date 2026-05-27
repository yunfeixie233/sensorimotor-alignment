# Conda environment snapshots

`pip freeze` of every conda environment used to reproduce the `cknna_vla` paper,
captured 2026-05-16 on the verified machine. One file per environment.

These are **exact-pin references**, not turnkey installers — `pip install -r`
will work for most of them but a few entries need care:

- `torch==X` lines: install torch first with the matching CUDA wheel index
  (`--index-url https://download.pytorch.org/whl/cuXXX`), then the rest.
- `flash_attn==X`: must match the env's torch ABI; build with `--no-cache-dir`
  or use a prebuilt wheel for that exact torch.
- `packaging @ file://...`: a conda-built artifact; pip will substitute the
  PyPI version harmlessly.

See `../README.md` ([Setup](../README.md#setup)) for which model family uses which env and the per-env install gotchas.

| File | Env | torch / key versions |
|------|-----|----------------------|
| `starVLA.requirements.txt`     | RLDS loaders + DTW/CKNNA/figures + StarVLA + raw VLMs + ViTs | torch 2.6.0, transformers 5.3.0, tensorflow 2.21.0 |
| `cogact.requirements.txt`      | CogACT, OpenVLA, Prismatic | torch 2.5.1+cu121, transformers 4.40.1 |
| `groot_libero.requirements.txt`| GR00T-N1.5 / N1.6 | torch 2.5.1+cu121, transformers 4.51.3 |
| `pi0fast_env.requirements.txt` | Pi0 | torch 2.5.1+cu121, transformers 4.57.6 |
| `spatialvla_env.requirements.txt` | SpatialVLA | torch 2.5.1+cu121, transformers 4.47.0 |
| `openvla_env.requirements.txt` | OpenVLA (alt) | torch 2.5.1+cu121, transformers 4.40.2 |
| `simpler_env.requirements.txt` | RT-1-X, Octo | TF 2.15 / JAX 0.4.20 |
| `motus.requirements.txt`       | Motus WAM + RoboTwin eval | torch 2.7.1+cu128 |
| `lingbot_va.requirements.txt`  | LingBot-VA WAM | torch 2.7.1+cu128 |
| `vidar.requirements.txt`       | Vidar WAM (python 3.12) | torch 2.7.1+cu128 |
| `robotwin.requirements.txt`    | RoboTwin 2.0 simulation | — |
