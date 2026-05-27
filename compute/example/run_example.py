"""
Self-contained DTW + Asymmetric Platonic CKNNA computation.

All functions are inlined — no external imports beyond standard libraries
(torch, numpy, scipy, numba). This script can be run standalone.

Input format:
  --feats_A  Path to a .pt file containing a torch tensor of shape [N, D]
             where N = number of episodes, D = feature dimension.
             These are the model's internal representations (e.g., VLM hidden
             states extracted at a single frame per episode). Each row is one
             episode's feature vector. Must be L2-normalizable (not all zeros).

  --trajs    Path to a .pt file containing a torch tensor of shape [N, T, 7]
             where N = number of episodes (must match feats_A),
                   T = trajectory length (number of timesteps),
                   7 = (x, y, z, roll, pitch, yaw, gripper).
             These are proprioceptive trajectories in SE(3) pose space.
             Euler angles (roll, pitch, yaw) are in radians, "xyz" convention.
             Gripper is a scalar (e.g., 0=open, 1=closed).

Output:
  Prints CKNNA value (float in [0, 1]) and mutual k-NN overlap.
  Higher CKNNA = stronger alignment between feature space and trajectory space.

Example:
  # Compute CKNNA on the small 20-sample example
  python run_example.py --feats_A example_feats.pt --trajs example_trajs.pt

  # Verify against previous experiment (N=6000, uses precomputed DTW cache)
  python run_example.py --feats_A full_feats_img.pt --trajs full_trajs_h40.pt \\
                        --dtw_cache dtw_topk_h40_k10.npy --ref_cknna 0.047826

Pipeline:
  Phase 1: Pairwise DTW on trajectories using SO(3) geodesic local cost
           → top-k=10 nearest neighbors → binary mask_L [N, N]
  Phase 2: Cosine similarity on features → top-k=10 → binary mask_K [N, N]
           + continuous similarity matrix K_sim [N, N]
  Phase 3: Asymmetric Platonic CKNNA =
           HSIC(mask_inter * K_sim, mask_inter * mask_L) /
           sqrt(HSIC(mask_K * K_sim, mask_K * K_sim) * HSIC(mask_L, mask_L))
           where mask_inter = mask_K * mask_L (intersection)
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

TOPK = 10


# ============================================================================
# DTW: GPU symmetric2 kernel with SO(3) geodesic cost
# ============================================================================

def build_pose_cost_matrices_gpu(pos_i, grip_i, quats_i,
                                 all_pos, all_grip, all_quats, weights):
    """Build [N, T_i, T_j] local cost matrices on GPU.

    Motivation: DTW needs a "local cost" between every pair of frames (t1, t2)
    from two trajectories. A robot pose has 3 components — position, rotation,
    and gripper — each needing its own distance metric. We compute all N cost
    matrices in one vectorized call (trajectory i vs all N trajectories).

    Local cost between two frames:
      d = w_pos * ||pos1 - pos2||_2           ← Euclidean on R³
        + w_rot * geodesic(quat1, quat2)      ← SO(3) geodesic angle
        + w_grip * |grip1 - grip2|            ← scalar difference

    Why SO(3) geodesic instead of Euclidean on Euler angles:
      Euler angles wrap around (e.g., 179° and -179° are 2° apart physically
      but 358° apart numerically). The quaternion geodesic measures the true
      shortest rotation path on the SO(3) manifold: θ = 2·atan2(sin(θ/2), cos(θ/2)).

    Why atan2 instead of arccos:
      The equivalent formula arccos(2·|q1·q2|² - 1) requires clamping to avoid
      NaN at the boundaries. atan2 is numerically stable without clamping.

    Args:
        pos_i:      [T, 3]    — positions of query trajectory i
        grip_i:     [T]       — gripper values of query trajectory i
        quats_i:    [T, 4]    — quaternions [w,x,y,z] of query trajectory i
        all_pos:    [N, T, 3] — positions of all N trajectories
        all_grip:   [N, T]    — gripper values of all N trajectories
        all_quats:  [N, T, 4] — quaternions of all N trajectories
        weights:    (w_pos, w_rot, w_grip) — cost component weights

    Returns:
        [N, T, T] float32 — cost[n, t1, t2] = local cost between frame t1 of
                             trajectory i and frame t2 of trajectory n
    """
    w_pos, w_rot, w_grip = weights

    # --- Component 1: Position distance (Euclidean on R³) ---
    # pos_i is [T, 3], all_pos is [N, T, 3].
    # Broadcasting: [1, T_i, 1, 3] - [N, 1, T_j, 3] → [N, T_i, T_j, 3]
    # Then sum over xyz and sqrt → [N, T_i, T_j] pairwise Euclidean distances.
    d_pos = torch.sqrt(
        ((pos_i[None, :, None, :] - all_pos[:, None, :, :]) ** 2).sum(dim=3)
    )

    # --- Component 2: Rotation distance (SO(3) geodesic) ---
    # Quaternion dot product: |q1 · q2| = |cos(θ/2)| where θ is rotation angle.
    # abs() handles the double-cover ambiguity: q and -q represent the same rotation.
    dots = torch.abs(
        (quats_i[None, :, None, :] * all_quats[:, None, :, :]).sum(dim=3)
    )
    # Clamp to [0, 1] for numerical safety (floating-point can exceed 1.0 slightly)
    torch.clamp_(dots, 0.0, 1.0)
    # sin(θ/2) = sqrt(1 - cos²(θ/2))
    sin_half = torch.sqrt(1.0 - dots * dots)
    # θ = 2 · atan2(sin(θ/2), cos(θ/2))  — geodesic angle in [0, π] radians
    d_rot = 2.0 * torch.atan2(sin_half, dots)

    # --- Component 3: Gripper distance (absolute difference) ---
    # Gripper is a scalar (e.g., 0=open, 1=closed), so |g1 - g2| suffices.
    d_grip = torch.abs(grip_i[None, :, None] - all_grip[:, None, :])

    # --- Combined local cost: weighted sum ---
    # With uniform weights (1,1,1), each component contributes equally.
    return w_pos * d_pos + w_rot * d_rot + w_grip * d_grip


def make_sym2_cuda_kernel():
    """Create numba CUDA kernel for hard DTW with symmetric2 step pattern.

    Motivation: DTW finds the minimum-cost monotone alignment between two
    sequences. The "symmetric2" step pattern is the standard one (used by
    dtw-python's default). From any cell (i,j), you can step to:
      - (i+1, j+1): diagonal — both sequences advance — costs 2×D (favored)
      - (i+1, j):   vertical — sequence 1 advances, seq 2 repeats — costs 1×D
      - (i, j+1):   horizontal — seq 2 advances, seq 1 repeats — costs 1×D
    The diagonal step costs 2×D to avoid bias: without it, the diagonal path
    through a TxT matrix has T steps while vert/horiz has 2T, unfairly penalizing
    the straight-through path.

    Why GPU: Pairwise DTW is O(N² · T²). For N=6000, T=40, that's 5.76×10¹⁰
    operations. The anti-diagonal trick parallelizes across T threads per pair:
    all cells on the same anti-diagonal are independent (they only depend on
    previous anti-diagonals), so T cells compute simultaneously.

    Recurrence: R[i,j] = D[i,j] + min(R[i-1,j-1] + D[i,j], R[i-1,j], R[i,j-1])
    """
    from numba import cuda

    @cuda.jit
    def harddtw_sym2(D, bandwidth, max_i, max_j, n_passes, R):
        # Each CUDA block processes one trajectory pair (batch element)
        b = cuda.blockIdx.x
        # Each thread handles one cell on the current anti-diagonal
        tid = cuda.threadIdx.x

        # Sweep through 2T-1 anti-diagonals (p=0 is top-left corner,
        # p=2T-2 is bottom-right corner)
        for p in range(n_passes):
            # Map thread id to (i, j) on anti-diagonal p.
            # R is 1-indexed (padded with inf at row 0 and col 0).
            # On anti-diagonal p, all cells satisfy i + j = p + 2.
            i = tid + 1
            j = p - tid + 1

            # Bounds check: only process valid cells within the TxT grid
            if 1 <= i <= max_i and 1 <= j <= max_j:
                # Sakoe-Chiba bandwidth constraint (disabled when bandwidth=0)
                if not (bandwidth > 0 and abs(i - j) > bandwidth):
                    # D is 0-indexed, R is 1-indexed, so D[i-1, j-1] = R's (i,j)
                    d = D[b, i - 1, j - 1]

                    # Three candidate predecessors:
                    diag = R[b, i - 1, j - 1] + d  # diagonal: accumulated + d (total 2×d for this cell)
                    vert = R[b, i - 1, j]           # vertical: seq 1 advances alone (1×d)
                    horz = R[b, i, j - 1]           # horizontal: seq 2 advances alone (1×d)

                    # Take minimum predecessor
                    best = diag if diag < vert else vert
                    best = best if best < horz else horz

                    # Store: local cost d + best predecessor
                    # This means diagonal contributes d + (prev + d) = 2d total,
                    # while vert/horz contribute d + prev = 1d total.
                    R[b, i, j] = d + best

            # CRITICAL: all threads must finish this anti-diagonal before
            # any thread starts the next one (data dependency)
            cuda.syncthreads()

    return harddtw_sym2


_SYM2_KERNEL = None


def compute_dtw_topk_gpu(trajs_np, weights, topk):
    """Pairwise DTW top-k on GPU.

    Motivation: We need to find, for each trajectory, its k most similar
    trajectories under DTW distance. This requires N² pairwise DTW computations,
    each O(T²). The GPU kernel parallelizes both within each DTW (anti-diagonal)
    and across batch elements (one CUDA block per pair).

    Strategy: For each row i, build ALL N cost matrices at once on GPU (i vs
    all N), run the DTW kernel in one launch (N blocks × T threads), then pick
    the k smallest distances. This is row-sequential (N outer iterations) but
    each iteration is fully GPU-parallel.

    Args:
        trajs_np: [N, T, 7] numpy float array — proprio trajectories
        weights:  (w_pos, w_rot, w_grip) — local cost weights
        topk:     number of nearest neighbors

    Returns:
        [N, topk] int64 numpy array — top-k neighbor indices per sample
    """
    # ---- Lazy-init the CUDA kernel (compiled once, reused) ----
    global _SYM2_KERNEL
    if _SYM2_KERNEL is None:
        _SYM2_KERNEL = make_sym2_cuda_kernel()
    from numba import cuda as numba_cuda

    N, T, _ = trajs_np.shape
    device = torch.device("cuda")

    # ---- Step A: Precompute quaternions for all frames ----
    # Motivation: The SO(3) geodesic operates on quaternions, but our input
    # is Euler angles. Converting once here avoids repeated conversion inside
    # the inner loop. scipy outputs [x,y,z,w], but we need [w,x,y,z].
    all_eulers = trajs_np[:, :, 3:6].reshape(-1, 3)  # [N*T, 3] — flatten all frames
    xyzw = Rotation.from_euler("xyz", all_eulers).as_quat()  # scipy convention: [x,y,z,w]
    wxyz = np.empty_like(xyzw)
    wxyz[:, 0] = xyzw[:, 3]      # w component
    wxyz[:, 1:4] = xyzw[:, 0:3]  # x, y, z components
    all_quats_np = wxyz.reshape(N, T, 4).astype(np.float32)

    # ---- Step B: Move data to GPU ----
    # We keep pos, grip, quats as separate tensors (not the full 7D) because
    # build_pose_cost_matrices_gpu needs them separately for the 3-component
    # distance formula.
    all_pos   = torch.tensor(trajs_np[:, :, :3].astype(np.float32), device=device)  # [N, T, 3]
    all_grip  = torch.tensor(trajs_np[:, :, 6].astype(np.float32), device=device)   # [N, T]
    all_quats = torch.tensor(all_quats_np, device=device)                            # [N, T, 4]

    # ---- Step C: Allocate output + DP buffer ----
    topk_indices = np.empty((N, topk), dtype=np.int64)
    n_passes = 2 * T - 1  # number of anti-diagonals in a TxT grid
    # R_buf is padded: R[:,0,:] and R[:,:,0] are boundaries initialized to inf.
    # The actual TxT DP fills R[:,1:T+1,1:T+1]. Final answer at R[:,T,T] = R[:,-2,-2].
    R_buf = torch.empty((N, T + 2, T + 2), device=device, dtype=torch.float32)

    # ---- Step D: Row-by-row DTW ----
    t0 = time.time()
    with torch.no_grad():
        for i in range(N):
            # D[n, t1, t2] = local cost between frame t1 of traj i and frame t2 of traj n
            # Shape: [N, T, T] — all N cost matrices computed in one vectorized GPU call.
            D = build_pose_cost_matrices_gpu(
                all_pos[i], all_grip[i], all_quats[i],
                all_pos, all_grip, all_quats, weights,
            )

            # Reset DP table: inf everywhere (boundary condition for the recurrence).
            R_buf.fill_(float("inf"))

            # Init trick: R[b,0,0] = -D[b,0,0]
            #
            # Why: The symmetric2 recurrence at cell (1,1) via diagonal is:
            #   R[1,1] = D[0,0] + (R[0,0] + D[0,0]) = D[0,0] + R[0,0] + D[0,0]
            #
            # dtw-python starts the path with cost = 1 × D[0,0] at the origin.
            # Setting R[0,0] = -D[0,0] gives:
            #   R[1,1] = D[0,0] + (-D[0,0] + D[0,0]) = D[0,0]  ✓
            #
            # Without this trick, the origin cell would contribute 2×D[0,0],
            # which is wrong — only interior diagonal steps should double-count.
            R_buf[:, 0, 0] = -D[:, 0, 0]

            # Launch kernel: N CUDA blocks (one per trajectory pair), T threads each.
            # bandwidth=0.0 disables Sakoe-Chiba window constraint.
            _SYM2_KERNEL[N, T](
                numba_cuda.as_cuda_array(D),
                0.0,       # bandwidth (0 = no constraint)
                T, T,      # max_i, max_j (grid dimensions)
                n_passes,  # 2T-1 anti-diagonals to sweep
                numba_cuda.as_cuda_array(R_buf),
            )

            # Read final DTW distances from R[T, T] = R[:, -2, -2]
            # (R is (T+2)x(T+2), so index T maps to -2)
            distances = R_buf[:, -2, -2]  # [N] — DTW distance from traj i to each traj
            distances[i] = float("inf")   # exclude self-distance (always 0)

            # Take k trajectories with smallest DTW distance
            _, topk_idx = torch.topk(distances, topk, largest=False)
            topk_indices[i] = topk_idx.cpu().numpy()

            # Progress reporting for long runs (N=6000 takes ~10 min)
            done = i + 1
            if done % 500 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (N - done) / rate
                print(f"    DTW [{done}/{N}]  {rate:.1f} rows/s  "
                      f"ETA {eta / 60:.1f} min", flush=True)

    print(f"    DTW done: {N} rows in {time.time() - t0:.1f}s")
    return topk_indices


def compute_dtw_topk_h1(trajs_np, weights, topk):
    """Fast path for T=1: DTW on single-frame = pairwise pose distance.

    Motivation: When trajectories have only 1 timestep, DTW degenerates to
    plain pairwise distance — no warping is possible. We skip the GPU kernel
    entirely and just compute the full N×N distance matrix on CPU with numpy
    broadcasting, which is faster for this case.

    Uses argpartition (O(N) partial sort) instead of argsort (O(N log N))
    since we only need the k smallest, not their order.
    """
    w_pos, w_rot, w_grip = weights
    N = len(trajs_np)
    data = trajs_np.reshape(N, -1)  # [N, 7] — single frame per trajectory

    # Euler → quaternion (same convention as GPU path)
    eulers = data[:, 3:6]
    xyzw = Rotation.from_euler("xyz", eulers).as_quat()
    quats = np.empty_like(xyzw)
    quats[:, 0] = xyzw[:, 3]
    quats[:, 1:4] = xyzw[:, 0:3]

    # All 3 distance components, fully vectorized as [N, N] matrices
    d_pos = np.sqrt(((data[:, None, :3] - data[None, :, :3]) ** 2).sum(axis=2))
    dots = np.abs((quats[:, None, :] * quats[None, :, :]).sum(axis=2))
    np.clip(dots, 0.0, 1.0, out=dots)
    d_rot = 2.0 * np.arctan2(np.sqrt(1.0 - dots * dots), dots)
    d_grip = np.abs(data[:, None, 6] - data[None, :, 6])

    dist_mat = w_pos * d_pos + w_rot * d_rot + w_grip * d_grip
    np.fill_diagonal(dist_mat, np.inf)  # exclude self
    return np.argpartition(dist_mat, topk, axis=1)[:, :topk].astype(np.int64)


def compute_dtw_topk(trajs_np, topk=TOPK, weights=(1.0, 1.0, 1.0)):
    """Compute DTW top-k neighbors. Routes to GPU kernel or T=1 fast path.

    Args:
        trajs_np: [N, T, 7] numpy array — (x, y, z, roll, pitch, yaw, gripper)
        topk:     k nearest neighbors (default: 10)
        weights:  (w_pos, w_rot, w_grip) uniform (1,1,1) by default

    Returns:
        [N, topk] int64 numpy array
    """
    N, T, D = trajs_np.shape
    assert D == 7, f"Expected 7D proprio (x,y,z,roll,pitch,yaw,gripper), got {D}D"
    if T == 1:
        return compute_dtw_topk_h1(trajs_np, weights, topk)
    else:
        return compute_dtw_topk_gpu(trajs_np, weights, topk)


# ============================================================================
# HSIC: Song et al. 2012, unbiased estimator (Eq. 5)
# ============================================================================

def hsic_unbiased(K, L):
    """Unbiased HSIC estimator (Song et al. 2012, Eq. 5).

    Motivation: HSIC (Hilbert-Schmidt Independence Criterion) measures
    statistical dependence between two kernel matrices. If K and L encode
    the same neighborhood structure, HSIC > 0. If independent, HSIC ≈ 0.
    We use it to test whether feature-space similarity (K) aligns with
    trajectory-space similarity (L).

    The "unbiased" variant (Song et al. 2012) corrects for finite-sample
    bias that makes naive HSIC positive even for independent variables.

    Args:
        K, L: [N, N] kernel/similarity matrices (diagonals will be zeroed).

    Returns:
        scalar — unbiased HSIC estimate (can be slightly negative for
                 independent inputs due to the centering correction)
    """
    m = K.shape[0]

    # Zero diagonals: self-similarity (K[i,i], L[i,i]) is always high and
    # uninformative — it would dominate HSIC if not removed.
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)

    # term1 = Σᵢⱼ K̃ᵢⱼ · L̃ⱼᵢ  (note: L is TRANSPOSED)
    # Measures direct co-occurrence: "when i→j is strong in K, is j→i strong in L?"
    # Chunked to avoid materializing a full N×N intermediate for large N.
    chunk = 2000
    term1 = torch.tensor(0.0, device=K.device, dtype=K.dtype)
    for i in range(0, m, chunk):
        end = min(i + chunk, m)
        term1 += (K_tilde[i:end] * L_tilde.T[i:end]).sum()

    # term2 = Σ(K̃) · Σ(L̃) / ((m-1)(m-2))
    # Centering correction: the expected value of term1 if K and L were
    # independent. Subtracting it makes HSIC zero-mean under the null.
    term2 = K_tilde.sum() * L_tilde.sum() / ((m - 1) * (m - 2))

    # term3 = 2 · (col_sums_K · row_sums_L) / (m-2)
    # Hub correction: penalizes "hub" samples that are neighbors of many
    # others in both spaces. Without this, a few popular samples would
    # inflate HSIC even when K and L are independent.
    # Mathematically: 2 · 1ᵀ K̃ L̃ 1 / (m-2), computed via dot product
    # of column sums of K and row sums of L to avoid N×N matmul.
    k_col = K_tilde.sum(dim=0)   # [N] — how popular is each sample as a neighbor in K?
    l_row = L_tilde.sum(dim=1)   # [N] — how much does each sample reach out in L?
    term3 = 2.0 * (k_col @ l_row) / (m - 2)

    return (term1 + term2 - term3) / (m * (m - 3))


# ============================================================================
# Asymmetric Platonic CKNNA
# ============================================================================

def build_mask_from_topk(topk_indices, N, device):
    """Convert [N, k] top-k index array → [N, N] binary float mask.

    Motivation: The DTW phase produces top-k neighbor indices (compact [N, k]),
    but HSIC needs a full [N, N] matrix. scatter_ places 1.0 at each neighbor
    position. The resulting mask is ASYMMETRIC: mask[i,j]=1 means "j is a
    DTW neighbor of i", but j may not have i as its neighbor.
    """
    topk_idx = torch.from_numpy(topk_indices).long().to(device)
    mask = torch.zeros(N, N, dtype=torch.float32, device=device)
    mask.scatter_(1, topk_idx, 1.0)  # mask[i, topk_idx[i, :]] = 1.0
    return mask


def compute_cosine_topk(feats_norm, topk, device):
    """L2-normalized features → cosine sim matrix + top-k binary mask.

    Motivation: For the K-side of CKNNA, we need BOTH:
      (a) K_sim — continuous cosine similarities (used as HSIC kernel values)
      (b) mask_K — binary top-k mask (used for intersection with mask_L)
    Most methods only need one or the other; Asymmetric Platonic needs both
    because it uses continuous values on the K-side but binary on the L-side.

    Returns:
        K_sim:  [N, N] cosine similarity (symmetric, continuous in [-1, 1])
        mask_K: [N, N] binary top-k mask (asymmetric: i→j ≠ j→i)
    """
    # Cosine similarity: for L2-normalized vectors, dot product = cosine sim
    K_sim = feats_norm @ feats_norm.T  # [N, N], symmetric, values in [-1, 1]

    # Find top-k most similar (excluding self: fill diagonal with -inf)
    sim_for_topk = K_sim.clone().fill_diagonal_(float("-inf"))
    _, topk_idx = torch.topk(sim_for_topk, topk, dim=1)

    # Convert to binary mask
    mask_K = torch.zeros(K_sim.shape[0], K_sim.shape[0], device=device)
    mask_K.scatter_(1, topk_idx, 1.0)
    return K_sim, mask_K


def asymmetric_platonic_cknna(K_sim, mask_K, mask_L):
    """Asymmetric Platonic CKNNA.

    Motivation: We want to measure "does the model's feature representation
    encode information about the physical trajectory?" This is a kernel
    alignment problem: does the feature-space kernel (K) align with the
    trajectory-space kernel (L)?

    Why "Asymmetric Platonic":
      - K-side uses CONTINUOUS cosine similarity — model features have
        well-calibrated cosine values, so HSIC can distinguish "very similar"
        (cos=0.95) from "barely in top-k" (cos=0.70).
      - L-side uses BINARY mask — DTW distances have no natural similarity
        scale (d=0.5 vs d=1.0 means nothing absolute), so any distance-to-
        similarity conversion (like exp(-d²)) would be arbitrary. Binary
        "neighbor or not" is cleaner.
      - INTERSECTION mask — focuses HSIC on pairs where BOTH spaces agree
        they're neighbors. Without it, non-neighbor pairs (value=0) dilute
        the signal.

    Args:
        K_sim:  [N, N] cosine similarity matrix (continuous)
        mask_K: [N, N] binary top-k mask from feature space
        mask_L: [N, N] binary top-k mask from DTW trajectory space

    Returns:
        cknna value (float, typically in [0, 0.15] for real data)
    """
    # Intersection: only pairs where BOTH spaces agree they're neighbors
    mask_inter = mask_K * mask_L  # element-wise product of two binary masks

    # Numerator: cross-space dependence
    # "Among mutually-agreed neighbor pairs, does feature similarity
    #  correlate with trajectory neighborhood membership?"
    sim_kl = hsic_unbiased(
        (mask_inter * K_sim).clone(),   # K-side: cosine values at intersection
        (mask_inter * mask_L).clone())  # L-side: binary 1s at intersection

    # Denominator: self-dependence of each space (for normalization)
    # Note: sim_kk uses mask_K (not mask_inter) — the full feature neighborhood
    sim_kk = hsic_unbiased(
        (mask_K * K_sim).clone(), (mask_K * K_sim).clone())
    # sim_ll uses mask_L — the full trajectory neighborhood
    sim_ll = hsic_unbiased(mask_L.clone(), mask_L.clone())

    # CKNNA = sim_kl / sqrt(sim_kk * sim_ll)
    # Analogous to Pearson correlation: normalize by geometric mean of self-dependences
    denom = (sim_kk * sim_ll).clamp(min=1e-10).sqrt()
    return (sim_kl / denom).item()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Self-contained DTW + Asymmetric Platonic CKNNA.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input format:
  feats_A:   .pt file → torch tensor [N, D]
             N = number of episodes, D = feature dimension.
             Model internal representations (one vector per episode).

  trajs:     .pt file → torch tensor [N, T, 7]
             N = number of episodes (must match feats_A),
             T = trajectory length (timesteps),
             7 = (x, y, z, roll, pitch, yaw, gripper)
             Euler angles in radians, "xyz" convention.

  dtw_cache: (optional) .npy file → numpy int64 array [N, k]
             Precomputed DTW top-k neighbor indices.
             If provided, skips DTW computation (Phase 1).

  ref_cknna: (optional) float — expected CKNNA value for validation.
             If provided, prints PASS/FAIL based on 1e-4 tolerance.

Examples:
  # Small example (20 samples, computes DTW from scratch)
  python run_example.py --feats_A example_feats.pt --trajs example_trajs.pt

  # Full reference check (6000 samples, uses cached DTW)
  python run_example.py --feats_A full_feats_img.pt --trajs full_trajs_h40.pt \\
                        --dtw_cache dtw_topk_h40_k10.npy --ref_cknna 0.047826
""")
    parser.add_argument("--feats_A", required=True,
                        help="Path to features .pt file [N, D]")
    parser.add_argument("--trajs", required=True,
                        help="Path to trajectories .pt file [N, T, 7]")
    parser.add_argument("--dtw_cache", default=None,
                        help="(Optional) Path to precomputed DTW top-k .npy file [N, k]. "
                             "Skips DTW computation if provided.")
    parser.add_argument("--topk", type=int, default=TOPK,
                        help=f"Number of nearest neighbors (default: {TOPK})")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Compute device (default: cuda)")
    parser.add_argument("--ref_cknna", type=float, default=None,
                        help="(Optional) Expected CKNNA value for validation")
    parser.add_argument("--ref_mknn", type=float, default=None,
                        help="(Optional) Expected mutual kNN value for validation")
    args = parser.parse_args()

    # ---- Load inputs ----
    feats_A = torch.load(args.feats_A, weights_only=True).float()
    trajs = torch.load(args.trajs, weights_only=True).float()

    N_f, D = feats_A.shape
    N_t, T, C = trajs.shape
    assert N_f == N_t, f"N mismatch: feats_A has {N_f}, trajs has {N_t}"
    assert C == 7, f"trajs last dim must be 7 (x,y,z,r,p,y,grip), got {C}"
    N = N_f

    print(f"feats_A: [{N}, {D}]  (from {args.feats_A})")
    print(f"trajs:   [{N}, {T}, {C}]  (from {args.trajs})")

    # ---- Phase 1: DTW top-k (or load cache) ----
    if args.dtw_cache is not None:
        print(f"\nPhase 1: Loading DTW cache from {args.dtw_cache}")
        dtw_topk = np.load(args.dtw_cache)
        assert dtw_topk.shape[0] == N, (
            f"DTW cache N={dtw_topk.shape[0]} != data N={N}")
        print(f"  DTW top-k: [{dtw_topk.shape[0]}, {dtw_topk.shape[1]}]")
    else:
        print(f"\nPhase 1: Computing DTW top-k (N={N}, T={T}, k={args.topk})")
        trajs_np = trajs.numpy()
        dtw_topk = compute_dtw_topk(trajs_np, topk=args.topk)

    mask_L = build_mask_from_topk(dtw_topk, N, args.device)

    # ---- Phase 2: Feature cosine similarity ----
    print(f"\nPhase 2: Feature cosine similarity (D={D})")
    feats_norm = F.normalize(feats_A, p=2, dim=-1).to(args.device)
    K_sim, mask_K = compute_cosine_topk(feats_norm, args.topk, args.device)

    # ---- Phase 3: Asymmetric Platonic CKNNA ----
    print("\nPhase 3: Asymmetric Platonic CKNNA")
    t0 = time.time()
    cknna = asymmetric_platonic_cknna(K_sim, mask_K, mask_L)
    mknn = ((mask_K * mask_L).sum() / (args.topk * N)).item()
    dt = time.time() - t0

    print(f"\n{'=' * 40}")
    print(f"  CKNNA      = {cknna:.6f}")
    print(f"  mutual_kNN = {mknn:.6f}")
    print(f"  Time       = {dt:.1f}s")
    print(f"{'=' * 40}")

    # ---- Optional: validate against reference ----
    if args.ref_cknna is not None:
        diff = abs(cknna - args.ref_cknna)
        ok = diff < 1e-4
        print(f"\n  Reference CKNNA: {args.ref_cknna}")
        print(f"  Diff:            {diff:.8f}")
        print(f"  {'PASS' if ok else 'FAIL'} (threshold: 1e-4)")

    if args.ref_mknn is not None:
        diff = abs(mknn - args.ref_mknn)
        ok = diff < 1e-4
        print(f"\n  Reference mknn:  {args.ref_mknn}")
        print(f"  Diff:            {diff:.8f}")
        print(f"  {'PASS' if ok else 'FAIL'} (threshold: 1e-4)")


if __name__ == "__main__":
    main()
