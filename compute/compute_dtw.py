"""
Phase 1: Compute DTW top-k neighbor indices on proprioceptive trajectories.

Reads:
    - {data_dir}/feats_B_seq.pt   — [N, T_max+1, 7] proprio trajectories
    - {data_dir}/metadata.json    — for valid-task filtering

Outputs:
    - {output_dir}/dtw_topk_h{H}_k{K}_sym2_cuda_nowin_nopad.npy  (one per horizon)

Method:
    SO(3) geodesic local cost (pos L2 + quat geodesic + grip L1),
    uniform weights (1,1,1), hard DTW symmetric2 step pattern,
    no Sakoe-Chiba window. GPU-accelerated via numba.cuda.

This is the EXPENSIVE step (hours on GPU for N=6000).
Run once; subsequent scripts read the cached .npy files.

Usage:
    conda run -n starVLA python compute_dtw.py \
        --data_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid \
        --output_dir /lambda/nfs/vla/cache/cknna_data_store/record/dtw_cache/droid \
        --horizons 1 3 7 15 25 40 75 110 150 220 \
        --topk 10

Typical runtime: ~2-4 hours on A100 for N=6000, 10 horizons.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation


TOPK = 10
WEIGHTS = (1.0, 1.0, 1.0)  # uniform: pos, rot, grip


# ── Padding detection ─────────────────────────────────────────────────────────

def compute_max_valid_horizon(feats_B_seq):
    """Per-sample max valid horizon (detects right-padding by exact equality)."""
    diff = (feats_B_seq[:, 1:] != feats_B_seq[:, :-1]).any(dim=-1)
    positions = torch.arange(diff.shape[1]).unsqueeze(0)
    masked_pos = positions * diff.long()
    has_any = diff.any(dim=1)
    rightmost = masked_pos.max(dim=1).values
    max_avail = rightmost + 2
    max_avail[~has_any] = 1
    return (max_avail - 1).int()


# ── GPU DTW kernel (symmetric2 step pattern) ─────────────────────────────────

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


def _build_pose_cost_matrices_gpu(pos_i, grip_i, quats_i,
                                  all_pos, all_grip, all_quats, weights):
    w_pos, w_rot, w_grip = weights
    d_pos = torch.sqrt(
        ((pos_i[None, :, None, :] - all_pos[:, None, :, :]) ** 2).sum(dim=3)
    )
    dots = torch.abs(
        (quats_i[None, :, None, :] * all_quats[:, None, :, :]).sum(dim=3)
    )
    torch.clamp_(dots, 0.0, 1.0)
    sin_half = torch.sqrt(1.0 - dots * dots)
    d_rot = 2.0 * torch.atan2(sin_half, dots)
    d_grip = torch.abs(grip_i[None, :, None] - all_grip[:, None, :])
    return w_pos * d_pos + w_rot * d_rot + w_grip * d_grip


def compute_dtw_topk_gpu(trajs_np, topk):
    """Compute pairwise DTW top-k on GPU using symmetric2 kernel."""
    global _SYM2_KERNEL
    if _SYM2_KERNEL is None:
        _SYM2_KERNEL = _make_sym2_cuda_kernel()
    from numba import cuda as numba_cuda

    N, T, _ = trajs_np.shape
    device = torch.device("cuda")

    all_eulers = trajs_np[:, :, 3:6].reshape(-1, 3)
    xyzw = Rotation.from_euler("xyz", all_eulers).as_quat()
    wxyz = np.empty_like(xyzw)
    wxyz[:, 0] = xyzw[:, 3]
    wxyz[:, 1:4] = xyzw[:, 0:3]
    all_quats_np = wxyz.reshape(N, T, 4).astype(np.float32)

    all_pos = torch.tensor(trajs_np[:, :, :3].astype(np.float32), device=device)
    all_grip = torch.tensor(trajs_np[:, :, 6].astype(np.float32), device=device)
    all_quats = torch.tensor(all_quats_np, device=device)

    topk_indices = np.empty((N, topk), dtype=np.int64)
    n_passes = 2 * T - 1
    R_buf = torch.empty((N, T + 2, T + 2), device=device, dtype=torch.float32)

    t0 = time.time()
    with torch.no_grad():
        for i in range(N):
            D = _build_pose_cost_matrices_gpu(
                all_pos[i], all_grip[i], all_quats[i],
                all_pos, all_grip, all_quats, WEIGHTS,
            )
            R_buf.fill_(float("inf"))
            R_buf[:, 0, 0] = -D[:, 0, 0]

            _SYM2_KERNEL[N, T](
                numba_cuda.as_cuda_array(D),
                0.0, T, T, n_passes,
                numba_cuda.as_cuda_array(R_buf),
            )

            distances = R_buf[:, -2, -2]
            distances[i] = float("inf")
            _, topk_idx = torch.topk(distances, topk, largest=False)
            topk_indices[i] = topk_idx.cpu().numpy()

            done = i + 1
            if done % 500 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (N - done) / rate
                print(f"  [{done}/{N}]  {rate:.1f} rows/s  "
                      f"ETA {eta / 60:.1f} min", flush=True)

    elapsed = time.time() - t0
    print(f"  Done: {N} rows in {elapsed / 60:.1f} min")
    return topk_indices


def compute_dtw_topk_h1(trajs_np, topk):
    """Fast path for h=1: DTW on length-1 = pairwise pose distance."""
    w_pos, w_rot, w_grip = WEIGHTS
    N = len(trajs_np)
    data = trajs_np.reshape(N, -1)
    pos = data[:, :3]
    grip = data[:, 6]

    eulers = data[:, 3:6]
    xyzw = Rotation.from_euler("xyz", eulers).as_quat()
    quats = np.empty_like(xyzw)
    quats[:, 0] = xyzw[:, 3]
    quats[:, 1:4] = xyzw[:, 0:3]

    d_pos = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))
    dots = np.abs((quats[:, None, :] * quats[None, :, :]).sum(axis=2))
    np.clip(dots, 0.0, 1.0, out=dots)
    d_rot = 2.0 * np.arctan2(np.sqrt(1.0 - dots * dots), dots)
    d_grip = np.abs(grip[:, None] - grip[None, :])

    dist_mat = w_pos * d_pos + w_rot * d_rot + w_grip * d_grip
    np.fill_diagonal(dist_mat, np.inf)
    topk_indices = np.argpartition(dist_mat, topk, axis=1)[:, :topk]
    return topk_indices.astype(np.int64)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Compute DTW top-k on proprio trajectories.")
    parser.add_argument("--data_dir", required=True,
                        help="Dir with feats_B_seq.pt and metadata.json")
    parser.add_argument("--output_dir", required=True,
                        help="Where to save dtw_topk_*.npy cache files")
    parser.add_argument("--horizons", type=int, nargs="+",
                        default=[1, 3, 7, 15, 25, 40, 75, 110, 150, 220])
    parser.add_argument("--topk", type=int, default=TOPK)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load trajectories
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
    max_valid_h = compute_max_valid_horizon(feats_B_seq)

    for h in args.horizons:
        out_path = os.path.join(
            args.output_dir,
            f"dtw_topk_h{h}_k{args.topk}_sym2_cuda_nowin_nopad.npy")

        if os.path.exists(out_path):
            topk = np.load(out_path)
            print(f"h={h}: CACHED ({out_path}), shape={topk.shape}")
            continue

        # Filter to valid samples for this horizon
        seq_mask = valid_mask & (max_valid_h >= h)
        N_h = seq_mask.sum().item()
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
