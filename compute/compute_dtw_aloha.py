"""
Phase 6: Compute DTW top-k neighbor indices on ALOHA 14-D joint-qpos action
trajectories.

Parallel to compute_dtw.py but with a plain Euclidean L2 local cost (no SO(3)
geodesic, no pos/rot/grip split), since ALOHA action is a 14-D joint vector
(2x 7-DoF arms) rather than a 7-D pose+grip.

Reads:
    - {data_dir}/feats_B_seq.pt   — [N, T_max+1, 14] action trajectories
    - {data_dir}/metadata.json    — for valid-task filtering

Outputs:
    - {output_dir}/dtw_topk_h{H}_k{K}_sym2_cuda_nowin_nopad.npy  (one per horizon)

Usage:
    conda run -n starVLA python compute_dtw_aloha.py \
        --data_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_aloha \
        --output_dir /lambda/nfs/vla/cache/cknna_data_store/record/dtw_cache/aloha \
        --horizons 1 3 7 15 25 40 75 110 150 220 \
        --topk 10
"""

import argparse
import json
import os
import time

import numpy as np
import torch


TOPK = 10


# ── Padding detection (same as compute_dtw.py) ───────────────────────────────

def compute_max_valid_horizon(feats_B_seq):
    diff = (feats_B_seq[:, 1:] != feats_B_seq[:, :-1]).any(dim=-1)
    positions = torch.arange(diff.shape[1]).unsqueeze(0)
    masked_pos = positions * diff.long()
    has_any = diff.any(dim=1)
    rightmost = masked_pos.max(dim=1).values
    max_avail = rightmost + 2
    max_avail[~has_any] = 1
    return (max_avail - 1).int()


# ── GPU DTW kernel (sym2, dimension-agnostic) ────────────────────────────────

def _make_sym2_cuda_kernel():
    from numba import cuda

    @cuda.jit
    def _harddtw_sym2(D, bandwidth, max_i, max_j, n_passes, R):
        b = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        for p in range(n_passes):
            i = tid + 1
            j = p - tid + 1
            if 1 <= i <= max_i and 1 <= j <= max_j:
                if not (bandwidth > 0 and abs(i - j) > bandwidth):
                    d = D[b, i - 1, j - 1]
                    diag = R[b, i - 1, j - 1] + d
                    vert = R[b, i - 1, j]
                    horz = R[b, i, j - 1]
                    best = diag if diag < vert else vert
                    best = best if best < horz else horz
                    R[b, i, j] = d + best
            cuda.syncthreads()

    return _harddtw_sym2


_SYM2_KERNEL = None


def _build_euclidean_cost_matrices_gpu(traj_i, all_trajs):
    """Euclidean L2 cost between traj_i (T, D) and each of all_trajs (N, T, D).

    Returns (N, T, T) tensor of pairwise timestep distances.
    """
    diff = traj_i[None, :, None, :] - all_trajs[:, None, :, :]   # (N, T, T, D)
    return torch.sqrt((diff ** 2).sum(dim=3))                    # (N, T, T)


def compute_dtw_topk_gpu(trajs_np, topk):
    """Pairwise hard-DTW top-k on (N, T, D) trajectories using sym2 kernel."""
    global _SYM2_KERNEL
    if _SYM2_KERNEL is None:
        _SYM2_KERNEL = _make_sym2_cuda_kernel()
    from numba import cuda as numba_cuda

    N, T, D = trajs_np.shape
    assert T <= 1024, f"sym2 kernel needs T<=1024 (CUDA threads/block), got T={T}"
    device = torch.device("cuda")

    all_trajs = torch.tensor(trajs_np.astype(np.float32), device=device)  # (N, T, D)

    topk_indices = np.empty((N, topk), dtype=np.int64)
    n_passes = 2 * T - 1
    R_buf = torch.empty((N, T + 2, T + 2), device=device, dtype=torch.float32)

    t0 = time.time()
    with torch.no_grad():
        for i in range(N):
            D_mat = _build_euclidean_cost_matrices_gpu(all_trajs[i], all_trajs)

            R_buf.fill_(float("inf"))
            R_buf[:, 0, 0] = -D_mat[:, 0, 0]

            _SYM2_KERNEL[N, T](
                numba_cuda.as_cuda_array(D_mat),
                0.0, T, T, n_passes,
                numba_cuda.as_cuda_array(R_buf),
            )

            distances = R_buf[:, -2, -2]
            distances[i] = float("inf")
            _, topk_idx = torch.topk(distances, topk, largest=False)
            topk_indices[i] = topk_idx.cpu().numpy()

            done = i + 1
            if done % 100 == 0 or done == N:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (N - done) / rate if rate > 0 else 0
                print(f"  [{done}/{N}]  {rate:.1f} rows/s  "
                      f"ETA {eta / 60:.1f} min", flush=True)

    elapsed = time.time() - t0
    print(f"  Done: {N} rows in {elapsed / 60:.1f} min")
    return topk_indices


def compute_dtw_topk_h1(trajs_np, topk):
    """h=1 fast path: pairwise Euclidean on first-frame action."""
    N = len(trajs_np)
    data = trajs_np.reshape(N, -1)  # (N, D) since T=1
    diff = data[:, None, :] - data[None, :, :]
    dist_mat = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    topk_indices = np.argpartition(dist_mat, topk, axis=1)[:, :topk]
    return topk_indices.astype(np.int64)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: DTW top-k on ALOHA 14-D joint-action trajectories.")
    parser.add_argument("--data_dir", required=True,
                        help="Dir with feats_B_seq.pt and metadata.json")
    parser.add_argument("--output_dir", required=True,
                        help="Where to save dtw_topk_*.npy cache files")
    parser.add_argument("--horizons", type=int, nargs="+",
                        default=[1, 3, 7, 15, 25, 40, 75, 110, 150, 220])
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--no_padding_filter", action="store_true",
                        help="Skip the consecutive-identical-frames padding detector. "
                             "Use when the dataset is known-clean (e.g., cknna_data_aloha, "
                             "which was built from real lerobot recordings with no padding). "
                             "At h=220 the default detector false-positives on 4 episodes that "
                             "happen to have ≥2 stationary action frames, dropping N from 839→835 "
                             "and causing a size mismatch against the feature tensors (839, D) in "
                             "compute_cknna.py. This flag disables the filter → N=839 at all horizons.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading trajectories...")
    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        meta = json.load(f)
    task_descriptions = meta["task_descriptions"]
    valid_mask = torch.tensor(
        [bool(t and t.strip()) for t in task_descriptions], dtype=torch.bool
    )
    feats_B_seq = torch.load(
        os.path.join(args.data_dir, "feats_B_seq.pt"), weights_only=True
    ).float()
    print(f"feats_B_seq shape: {tuple(feats_B_seq.shape)}")

    if args.no_padding_filter:
        print("  [no_padding_filter] skipping padding detection — "
              "using all N samples at every horizon")
        max_valid_h = torch.full((feats_B_seq.shape[0],),
                                 feats_B_seq.shape[1] - 1, dtype=torch.int32)
    else:
        max_valid_h = compute_max_valid_horizon(feats_B_seq)

    for h in args.horizons:
        out_path = os.path.join(
            args.output_dir,
            f"dtw_topk_h{h}_k{args.topk}_sym2_cuda_nowin_nopad.npy")

        if os.path.exists(out_path):
            topk = np.load(out_path)
            print(f"h={h}: CACHED ({out_path}), shape={topk.shape}")
            continue

        seq_mask = valid_mask & (max_valid_h >= h)
        N_h = seq_mask.sum().item()
        # Slice timesteps [1:h+1] to mirror compute_dtw.py convention:
        # index 0 is the "current" frame, 1..h are the next h future frames.
        traj = feats_B_seq[seq_mask].numpy()[:, 1:h + 1, :]
        print(f"\nh={h}: N={N_h}, traj shape={traj.shape}")

        if h == 1:
            topk = compute_dtw_topk_h1(traj, args.topk)
        else:
            topk = compute_dtw_topk_gpu(traj, args.topk)

        np.save(out_path, topk)
        print(f"  Saved: {out_path}")

    del feats_B_seq
    print("\nDone.")


if __name__ == "__main__":
    main()
