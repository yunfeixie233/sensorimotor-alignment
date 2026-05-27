# DTW-CKNNA: Trajectory-Aware Representation Alignment

Self-contained implementation of **DTW + Asymmetric Platonic CKNNA** — a metric that measures whether a model's internal representations encode information about physical trajectories.

## What is CKNNA?

**CKNNA** (Centered Kernel Nearest-Neighbor Alignment) measures the alignment between two spaces:
- **Feature space**: model internal representations (e.g., VLM hidden states)
- **Trajectory space**: ground-truth proprioceptive trajectories (robot poses over time)

High CKNNA = the model's representations encode trajectory-relevant structure.

## Pipeline

```
feats_A [N, D]              trajs [N, T, 7]
     │                            │
 cosine similarity         pairwise DTW distance
     │                     (SO(3) geodesic cost)
     │                            │
 top-k → mask_K             top-k → mask_L
 + keep K_sim                     │
     │                            │
     └──────────┬─────────────────┘
                │
         mask_inter = mask_K * mask_L
                │
  HSIC(inter*K_sim, inter*mask_L)     ← numerator
  ─────────────────────────────────
  √(HSIC(mask_K*K_sim, ...) ·        ← denominator
    HSIC(mask_L, mask_L))
                │
           CKNNA value
```

## Input Format

**Features** (`--feats_A`): `.pt` file → `torch.Tensor [N, D]`
- `N` = number of episodes
- `D` = feature dimension
- One feature vector per episode (e.g., VLM hidden state at a single frame)

**Trajectories** (`--trajs`): `.pt` file → `torch.Tensor [N, T, 7]`
- `N` = number of episodes (must match features)
- `T` = trajectory length (number of timesteps)
- `7` = `(x, y, z, roll, pitch, yaw, gripper)`
  - Euler angles in **radians**, `"xyz"` convention
  - Gripper is a scalar (e.g., 0=open, 1=closed)

## Usage

```bash
# Run on the included small example (N=20, T=40)
python run_example.py --feats_A example_feats.pt --trajs example_trajs.pt

# With precomputed DTW cache (skips Phase 1)
python run_example.py --feats_A my_feats.pt --trajs my_trajs.pt \
                      --dtw_cache my_dtw_topk.npy

# Validate against a known reference value
python run_example.py --feats_A feats.pt --trajs trajs.pt \
                      --ref_cknna 0.047826
```

## Example Output

```
feats_A: [20, 4096]  (from example_feats.pt)
trajs:   [20, 40, 7]  (from example_trajs.pt)

Phase 1: Computing DTW top-k (N=20, T=40, k=10)
    DTW done: 20 rows in 1.0s

Phase 2: Feature cosine similarity (D=4096)

Phase 3: Asymmetric Platonic CKNNA

========================================
  CKNNA      = 0.681878
  mutual_kNN = 0.515000
  Time       = 0.0s
========================================
```

## Requirements

- Python 3.8+
- PyTorch (with CUDA)
- NumPy (<2.0)
- SciPy
- Numba (for CUDA DTW kernel)

```bash
pip install torch numpy scipy numba
```

## Files

| File | Description |
|------|-------------|
| `run_example.py` | Self-contained script — all DTW + HSIC + CKNNA functions inlined |
| `prepare_example_data.py` | Extract example data from DROID experiment (optional) |
| `example_feats.pt` | Small example: `[20, 4096]` CogACT-base features |
| `example_trajs.pt` | Small example: `[20, 40, 7]` proprio trajectories |

## Key Design Choices

- **DTW variant**: GPU symmetric2 kernel (numba CUDA), SO(3) geodesic local cost, uniform weights (1,1,1), no Sakoe-Chiba window
- **K-side (features)**: continuous cosine similarity — naturally well-calibrated
- **L-side (trajectories)**: binary k-NN mask — DTW distances have no natural scale
- **Intersection mask**: focuses HSIC on pairs where both spaces agree on neighborhood
- **HSIC**: Song et al. 2012 unbiased estimator (Eq. 5)
