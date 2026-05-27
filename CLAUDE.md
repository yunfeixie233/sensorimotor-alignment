# CLAUDE.md

This file gives Claude Code (and any other coding agent) a fast-orientation
map of this repository. **Always read [README.md](README.md) first for the
user-facing intro** — this file only documents conventions an agent needs to
move around the codebase efficiently.

## Project at a glance

- **Goal**: compute `S² = CKNNA(K_sensory, K_motor)`, a training-free offline
  metric that correlates with closed-loop robot policy success rate.
- **Three model families covered**: VLA (vision-language-action),
  VLM (vision-language, raw pretrained), WAM (world-action / video models).
- **Three datasets covered**: DROID (single-arm Franka), BridgeDataV2 (WidowX),
  ALOHA bimanual.
- **Five-stage pipeline**: (1) dataset prep → (2) feature extraction → (3) motor
  kernel (DTW) → (4) sensorimotor alignment (S² via CKNNA) → (5) figures.

## Repository layout (what to edit, what to read)

```
README.md              user-facing intro + environment setup + experiment recipes
CLAUDE.md              this file
paths.env              ONE-FILE machine config — edit before running anything
config.py              Python loader for paths.env (do not import sibling scripts; they import this)

data/                  Stage 1: RLDS / LeRobot loaders (TF dependency)
extractors/            Stage 2: feature extraction, one file per model family
  extract_7feat_<family>.py     — canonical, do not create new families ad-hoc
  extract_all_features.py       — 8 raw VLMs in a single forward pass
  run_all7_smoke.sh             — N=10 smoke driver across all 8 raw VLMs
vision_encoder/        Stage 2: 23 standalone ViT extractors (paradigm experiment)
compute/               Stage 3 (compute_dtw.py / compute_dtw_aloha.py) + Stage 4 (compute_cknna.py)
                       + example/ self-contained 350 KB CKNNA demo
analysis/              Stage 5 supporting analyses (perception metrics, visual confound, …)

record/                committed Stage-4 CSVs + figure-regen code, one folder per paper figure
  fig_<section>_<slug>/
    README.md          paper section, figure label, filter applied, sanity-check r
    data/              cknna_*_complete.csv (all models × features × horizons)
                       figure_subset.csv (exactly the rows plotted in the paper)
                       <SR>.csv (success-rate / benchmark ground truth)
    code/              regen script(s), self-contained
    <figure>.{pdf,png} rendered output
  shared/              google_palette.py, scatter_labels.py

repos/                 vendored model code (~180 MB); each model family gets one subdir
repro_envs/yaml/       conda env yamls (11 envs)
repro_envs/*.txt       matching pip-freeze snapshots
tests/                 test_paths.py / test_imports.py / verify_envs.sh
```

## Conventions an agent must respect

### 1. The eleven conda envs are mutually incompatible

Activate via `conda run -n <env>` — **never `pip install` into the wrong env**.

| If you need to load…                          | Use env             |
|-----------------------------------------------|---------------------|
| Stage A / C / D / E (loaders, DTW, CKNNA, figures) | `starVLA`           |
| StarVLA ×5, 8 raw VLMs, 23 standalone ViTs    | `starVLA`           |
| CogACT, OpenVLA, raw Prismatic                | `cogact`            |
| GR00T-N1.5, GR00T-N1.6                        | `groot_libero`      |
| Pi0                                           | `pi0fast_env`       |
| SpatialVLA                                    | `spatialvla_env`    |
| RT-1-X, Octo                                  | `simpler_env`       |
| Motus / LingBot-VA / Vidar                    | `motus`/`lingbot_va`/`vidar` |
| RoboTwin sim                                  | `robotwin`          |

### 2. Paths come from `paths.env` only

Never hard-code dataset, checkpoint, or cache paths in Python — read them via
`config.py`. The one place absolute paths exist is **paper-repo figure
scripts** (`vla_idea/writing/cknna_vla/figures/*.py`), which intentionally
hard-code `/home/ubuntu/vla/cknna_project/...` so they're `cd`-robust. On a
fresh machine either symlink or edit those constants.

### 3. CSVs in `record/fig_*/data/` are the source of truth

The paper figures read these CSVs directly. **Do not rewrite them** unless
you re-ran Stage 4. If you re-run Stage 4 and the CSVs diverge from the
committed values by more than 1e-4, treat that as a bug. Each fig folder
ships both the complete cohort CSV and the small `figure_subset.csv`
(exactly the rows plotted in the paper).

### 4. Adding a new model family

1. Create `extractors/extract_7feat_<family>.py` mirroring an existing one
   (e.g. `extract_7feat_starvla.py`).
2. Make sure the 7 features (`vit_raw`, `pre_llm`, `pre_llm_txt`,
   `pre_llm_vt`, `img`, `txt`, `imgtext`) are saved as `feats_A_<variant>.pt`.
3. Pick an existing env that the model is compatible with, or add a new yaml
   under `repro_envs/yaml/` and a matching `.requirements.txt`.

### 5. Stage A loaders need TensorFlow

`data/load_*_rlds.py` use `tensorflow_datasets`. The `starVLA` env ships
`tensorflow==2.21.0` + `tensorflow_datasets==4.9.9` alongside torch 2.6 —
do not split them.

## Common tasks → exact commands

```bash
# Quick: confirm install
set -a && source paths.env && set +a
conda run -n starVLA python tests/test_paths.py
conda run -n starVLA python tests/test_imports.py

# Self-contained pipeline demo (350 KB pre-built features ship in compute/example/)
conda run -n starVLA python compute/example/run_example.py \
    --feats_A compute/example/example_feats.pt \
    --trajs   compute/example/example_trajs.pt

# Regenerate one paper figure (every fig has its own self-contained folder under record/)
conda run -n starVLA python record/fig_3.1_pa_predicts_sr_1x3/code/combined_3benchmarks_scatter.py

# Full Stages 3 + 4 for VLA-SR on DROID
set -a && source paths.env && set +a
conda run -n starVLA python compute/compute_dtw.py …
conda run -n starVLA python compute/compute_cknna.py …
```

## Where the paper sits

- LaTeX source: `vla_idea/writing/cknna_vla/main_single.tex`
- Compiled PDF: `vla_idea/writing/cknna_vla/main_single.pdf`
- Figure scripts: ship inside this repo at `record/fig_*/code/<regen>.py`. The
  paper repo at `vla_idea/writing/cknna_vla/figures/` holds an older copy used
  to compile the paper standalone; the canonical version is here in `record/`.

## Tests

```bash
conda run -n starVLA python tests/test_paths.py       # 18 required paths OK + 6 optional may be MISSING pre-extraction
conda run -n starVLA python tests/test_imports.py     # 10 modules import
bash tests/verify_envs.sh                             # 11 OK / 0 failed (all conda envs)
bash extractors/run_all7_smoke.sh                     # N=10 across 8 raw VLMs (GPU, requires HF_TOKEN)
```

## Style rules for prose / docs in this repo

- READMEs use em-dashes sparingly; ai-content-checklist patterns apply.
- Numeric paper claims must match `record/fig_*/data/*.csv` to 1e-4.
- Terminology audit: VLA ≠ VA ≠ VLM (three families, defined on first use).
- Feature positions are named **ViT-out / LLM-in / LLM-out × {V, T, V+T}** in
  prose; on-disk filenames use the legacy slugs `vit_raw / pre_llm / pre_llm_txt
  / pre_llm_vt / img / txt / imgtext`.
