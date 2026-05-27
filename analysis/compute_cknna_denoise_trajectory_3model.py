"""3-model CKNNA along denoising schedule (populates layerwise-vs-denoising-step.tex).

Produces one (model, feature, step_pct) -> CKNNA value for:
  - 3 models:   Motus, LingBot-VA V25, Vidar
  - 8 features: V-D1, V-B0, V-B7, V-B14, V-B21, V-B29, V-Norm, T-D1
  - 5 step_pcts: 0, 25, 50, 75, 100 (% of each model's native denoising schedule)

Plus random baseline row (i.i.d. Gaussian features, 10-seed mean ± std).

A-side (features):
    Motus:   <images>/<task>/dense_data/<ep>_denoise_trajectory.pt   list of 5 step dicts
    LB-VA:   <real>/<prompt>/hook_features_<chunk>.pt['depth_trajectory']  list of 5 step dicts
    Vidar:   <capture>/<task>/<ep>/hook_features_chunk00.pt['features'][name][step_idx]  dict-of-dicts, 20 steps

B-side (trajectory DTW top-k over proprio endpose):
    All 3: left_endpose + right_endpose concat → 16D trajectory.
    Motus/LB-VA use Euler angles (8D/arm: pos[3]+euler[3]+grip[1]+unused[1]).
    Vidar uses quaternions in endpose — converted to Euler for the standard dualarm cost.

Imports DTW + CKNNA from lingbot-va/compute_cknna_denoise_trajectory_gpu.py (proven pipeline).

Usage:
  PYTHONNOUSERSITE=1 python compute_cknna_denoise_trajectory_3model.py \
      --motus_root   /lambda/nfs/vla/RoboTwin/policy/Motus/logs_50x10_20260418_051412/images \
      --lbva_root    /lambda/nfs/vla/lingbot-va/visualization_lingbot_50x10/real \
      --lbva_dense_root /lambda/nfs/vla/RoboTwin/results_lingbot_50x10 \
      --vidar_server_root /lambda/nfs/vla/vidar-robotwin/vidar_capture \
      --vidar_client_root /lambda/nfs/vla/vidar-robotwin/eval_result/ar/vidar_full \
      --out /lambda/nfs/vla/cknna_project/analysis/out/cknna_denoise_3model_50x10.csv

  # Task filter for quick smoke:
  ... --tasks adjust_bottle click_bell
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# Import proven GPU DTW + CKNNA from the 2026-04-08 single-model pipeline.
sys.path.insert(0, "/lambda/nfs/vla/lingbot-va")
from compute_cknna_denoise_trajectory_gpu import (  # noqa: E402
    compute_dtw_topk_gpu,
    asymmetric_platonic_cknna_gpu,
    euler_to_quat_wxyz,
)

# ---------------------------------------------------------------------------
# Canonical feature + step schema
# ---------------------------------------------------------------------------
CANON_FEATURES = [
    "V-D1", "V-B0", "V-B7", "V-B14", "V-B21", "V-B29", "V-Norm", "T-D1",
]
STEP_PCTS = [0, 25, 50, 75, 100]
TOPK = 10

# Per-model aliases → canonical. LB-VA uses "block0" etc; Motus/Vidar use "V-B0" etc.
LBVA_ALIAS = {
    "D1-V": "V-D1",
    "block0": "V-B0", "block7": "V-B7", "block14": "V-B14",
    "block21": "V-B21", "block29": "V-B29",
    "norm_out": "V-Norm",
    "D1-T": "T-D1",
}


def quat_wxyz_to_euler_xyz(wxyz: np.ndarray) -> np.ndarray:
    xyzw = np.concatenate([wxyz[..., 1:4], wxyz[..., 0:1]], axis=-1)
    return Rotation.from_quat(xyzw).as_euler("xyz")


def _mean_pool(t):
    """Reduce a per-step hook tensor to a (3072,) numpy vector."""
    if isinstance(t, np.ndarray):
        arr = t
    else:
        arr = t.detach().float().cpu().numpy()
    if arr.ndim == 3:                       # [1, L, D]
        arr = arr[0].mean(axis=0)
    elif arr.ndim == 2:                     # [L, D] or [1, D]
        arr = arr.mean(axis=0) if arr.shape[0] > 1 else arr[0]
    # else already 1-D [D]
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-model loaders. Each returns:
#     trajs:   list[np.ndarray(T_i, 16)]
#     quats:   list[np.ndarray(T_i, 8)]   (wxyz L + wxyz R)
#     feats:   dict[(canon_feat, step_pct)] -> list[np.ndarray(3072,)]
#     meta:    list[dict]    per-episode identity
# ---------------------------------------------------------------------------

def _traj_quats_from_dual_euler(seq16):
    """seq16: (T, 16) array with [pos_L(3), euler_L(3), grip_L(1), _, pos_R(3), euler_R(3), grip_R(1), _]"""
    q_L = euler_to_quat_wxyz(seq16[:, 3:6])
    q_R = euler_to_quat_wxyz(seq16[:, 11:14])
    return np.concatenate([q_L, q_R], axis=1)


def load_motus(images_root: Path, tasks_filter: List[str] | None):
    trajs, quats, metas = [], [], []
    feats: Dict[Tuple[str, int], List[np.ndarray]] = {}

    task_dirs = sorted(
        d for d in images_root.iterdir()
        if d.is_dir() and d.name not in {"dense_data", "latents"}
    )
    if tasks_filter:
        task_dirs = [d for d in task_dirs if d.name in tasks_filter]

    # Motus native: denoise_trajectory has 5 entries with step_counter ∈ {1,3,5,8,10}.
    # Map step_counter to nearest target % of schedule.
    # step_counter/10 → pct ≈ {10, 30, 50, 80, 100}; we snap to {0, 25, 50, 75, 100}.
    MOTUS_STEP_TO_PCT = {1: 0, 3: 25, 5: 50, 8: 75, 10: 100}

    for td in task_dirs:
        dd = td / "dense_data"
        if not dd.is_dir():
            continue
        ep_pkls = sorted(
            dd.glob("*_dense_obs.pkl"),
            key=lambda p: int(p.name.split("_")[0]),
        )
        for pkl in ep_pkls:
            try:
                ep = int(pkl.name.split("_")[0])
            except ValueError:
                continue
            dn = dd / f"{ep}_denoise_trajectory.pt"
            if not dn.exists():
                continue

            # --- B-side: trajectory over keyframes with endpose_left + endpose_right ---
            with open(pkl, "rb") as f:
                dense = pickle.load(f)
            kfs = [d for d in dense
                   if isinstance(d, dict)
                   and "endpose_left" in d and "endpose_right" in d
                   and d["endpose_left"] is not None]
            if len(kfs) < 3:
                continue
            seq = np.stack([
                np.concatenate([np.asarray(kf["endpose_left"], dtype=np.float32),
                                np.asarray(kf["endpose_right"], dtype=np.float32)])
                for kf in kfs
            ])
            if seq.shape[1] != 16:
                continue

            # --- A-side: denoise_trajectory list ---
            dt = torch.load(dn, weights_only=False, map_location="cpu")
            if not isinstance(dt, list) or len(dt) == 0:
                continue
            ep_feats_this = {}
            for entry in dt:
                if not isinstance(entry, dict):
                    continue
                step_counter = entry.get("step")
                pct = MOTUS_STEP_TO_PCT.get(step_counter)
                if pct is None:
                    continue
                hooks = entry.get("hooks") or entry  # support flat dict too
                for canon in CANON_FEATURES:
                    if canon in hooks:
                        ep_feats_this[(canon, pct)] = _mean_pool(hooks[canon])

            if len(ep_feats_this) == 0:
                continue
            trajs.append(seq)
            quats.append(_traj_quats_from_dual_euler(seq))
            for key, vec in ep_feats_this.items():
                feats.setdefault(key, []).append(vec)
            metas.append({"model": "motus", "task": td.name, "episode": ep})

    return trajs, quats, feats, metas


def load_lbva(real_root: Path, dense_root: Path, tasks_filter: List[str] | None):
    """LB-VA 50×10. Prompt-named dirs on server-side; task-named dirs on client-side.

    Skip task-filtering for LB-VA unless a pre-built prompt→task map is supplied —
    prompt dirs have natural-language names that don't trivially map to task names.
    The comparator averages over all available prompts (50 tasks × 10 eps = 500 total).
    """
    trajs, quats, metas = [], [], []
    feats: Dict[Tuple[str, int], List[np.ndarray]] = {}

    # LB-VA native: depth_trajectory has 5 entries with step ∈ {0, 6, 13, 19, 25}.
    # Map to {0, 25, 50, 75, 100}%. Indexes within the list rather than by step.
    # Use enumerate over the sorted list since step values can vary slightly per run.

    # Build a flat dense_obs lookup: {(task, ep_num) -> path} from client-side dir.
    # LB-VA client convention: RoboTwin/results_lingbot_50x10/stseed-*/dense_data/<task>/<ep>_dense_obs.pkl
    dense_map: Dict[Tuple[str, int], Path] = {}
    for seed_dir in dense_root.glob("stseed-*"):
        dd = seed_dir / "dense_data"
        if not dd.is_dir():
            continue
        for task_dir in dd.iterdir():
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            if tasks_filter and task not in tasks_filter:
                continue
            for pkl in task_dir.glob("*_dense_obs.pkl"):
                m = re.match(r"(\d+)_dense_obs\.pkl", pkl.name)
                if m:
                    dense_map[(task, int(m.group(1)))] = pkl

    # Enumerate server-side prompt dirs, match to dense by TIMESTAMP ORDER within
    # each task. Since we can't map prompt → task purely from the dir name,
    # iterate in creation-timestamp order and rely on the dense_map's episode
    # index within each task. Fallback: skip episodes with no dense_obs match.
    # NOTE: this pairing assumes server writes match client writes 1:1 by order.
    # For now we do a best-effort global pairing by timestamp.
    all_prompt_dirs = sorted(
        [d for d in real_root.iterdir() if d.is_dir()],
        key=lambda d: d.name.rsplit("_", 2)[-2:] if "_" in d.name else d.name,
    )

    # If we have a side-map file (task, ep) -> prompt_dir, use that.
    side_map_path = real_root.parent / "prompt_to_task.json"
    prompt_to_task: Dict[str, Tuple[str, int]] = {}
    if side_map_path.exists():
        with open(side_map_path) as f:
            raw = json.load(f)
        for k, v in raw.items():
            prompt_to_task[k] = (v["task"], int(v["episode"]))

    for pd in all_prompt_dirs:
        key = prompt_to_task.get(pd.name)
        if key is None:
            # Without a task map, we pair by timestamp position: fall back to
            # taking the N-th prompt-dir as episode (N % 10) of task (N // 10)
            # given sorted task list. This is fragile; skip for now and rely
            # on prompt_to_task.json when it exists.
            continue
        task, ep = key
        if tasks_filter and task not in tasks_filter:
            continue

        # A-side: hook_features for chunk 0
        hf_candidates = sorted(pd.glob("hook_features_*.pt"),
                               key=lambda p: int(re.search(r"_(\d+)\.", p.name).group(1)))
        if not hf_candidates:
            continue
        hf = torch.load(hf_candidates[0], map_location="cpu", weights_only=False)
        dt = hf.get("depth_trajectory")
        if not dt:
            continue

        # B-side trajectory
        dense_path = dense_map.get((task, ep))
        if dense_path is None:
            continue
        with open(dense_path, "rb") as f:
            dense = pickle.load(f)
        kfs = [d for d in dense
               if isinstance(d, dict)
               and "endpose_left" in d and "endpose_right" in d
               and d["endpose_left"] is not None]
        if len(kfs) < 3:
            continue
        seq = np.stack([
            np.concatenate([np.asarray(kf["endpose_left"], dtype=np.float32),
                            np.asarray(kf["endpose_right"], dtype=np.float32)])
            for kf in kfs
        ])
        if seq.shape[1] != 16:
            continue

        ep_feats_this = {}
        for pos_idx, entry in enumerate(dt):
            if pos_idx >= len(STEP_PCTS):
                break
            pct = STEP_PCTS[pos_idx]
            hooks = entry.get("hooks", {})
            for raw_name, vec in hooks.items():
                canon = LBVA_ALIAS.get(raw_name)
                if canon is None:
                    continue
                ep_feats_this[(canon, pct)] = _mean_pool(vec)

        if len(ep_feats_this) == 0:
            continue
        trajs.append(seq)
        quats.append(_traj_quats_from_dual_euler(seq))
        for k, v in ep_feats_this.items():
            feats.setdefault(k, []).append(v)
        metas.append({"model": "lbva_v25", "task": task, "episode": ep})

    return trajs, quats, feats, metas


def _vidar_endpose_to_arm8d(ep_dict: dict) -> np.ndarray | None:
    """Vidar endpose = [x, y, z, qw, qx, qy, qz]. Convert to 8D per arm:
    [pos(3), euler(3), grip(1), unused(1)]."""
    left = ep_dict.get("endpose", {}).get("left_endpose")
    right = ep_dict.get("endpose", {}).get("right_endpose")
    grip_l = ep_dict.get("endpose", {}).get("left_gripper", 0.0)
    grip_r = ep_dict.get("endpose", {}).get("right_gripper", 0.0)
    if left is None or right is None:
        return None
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.shape[-1] != 7 or right.shape[-1] != 7:
        return None
    eul_l = quat_wxyz_to_euler_xyz(left[3:7][None])[0].astype(np.float32)
    eul_r = quat_wxyz_to_euler_xyz(right[3:7][None])[0].astype(np.float32)
    arm_L = np.array([left[0], left[1], left[2],
                      eul_l[0], eul_l[1], eul_l[2], grip_l, 0.0], dtype=np.float32)
    arm_R = np.array([right[0], right[1], right[2],
                      eul_r[0], eul_r[1], eul_r[2], grip_r, 0.0], dtype=np.float32)
    return np.concatenate([arm_L, arm_R])


def load_vidar(server_root: Path, client_root: Path, tasks_filter: List[str] | None):
    trajs, quats, metas = [], [], []
    feats: Dict[Tuple[str, int], List[np.ndarray]] = {}

    # Vidar native: 20 cond steps. Sub-sample 5 evenly: {0, 5, 10, 15, 19}.
    VIDAR_STEP_INDICES = [0, 5, 10, 15, 19]
    VIDAR_INDEX_TO_PCT = dict(zip(VIDAR_STEP_INDICES, STEP_PCTS))

    task_dirs = sorted(d for d in server_root.iterdir() if d.is_dir())
    if tasks_filter:
        task_dirs = [d for d in task_dirs if d.name in tasks_filter]

    for td in task_dirs:
        task = td.name
        ep_dirs = sorted(d for d in td.iterdir() if d.is_dir() and d.name.startswith("episode"))
        for ed in ep_dirs:
            m = re.match(r"episode(\d+)", ed.name)
            if not m:
                continue
            ep = int(m.group(1))

            # A-side
            hf_path = ed / "hook_features_chunk00.pt"
            if not hf_path.exists():
                continue
            payload = torch.load(hf_path, map_location="cpu", weights_only=False)
            feats_dict = payload.get("features", {})

            # B-side: client-side dense_obs.pkl (note nested dir: <task>/<task>/)
            dense_path = client_root / task / task / ed.name / "dense_obs.pkl"
            if not dense_path.exists():
                # older runs saved to a flatter path
                dense_path = client_root / task / ed.name / "dense_obs.pkl"
            if not dense_path.exists():
                continue
            with open(dense_path, "rb") as f:
                dense = pickle.load(f)

            sorted_keys = sorted(dense.keys())
            seq_list = []
            for k in sorted_keys:
                arm16 = _vidar_endpose_to_arm8d(dense[k])
                if arm16 is not None:
                    seq_list.append(arm16)
            if len(seq_list) < 3:
                continue
            seq = np.stack(seq_list)
            if seq.shape[1] != 16:
                continue

            ep_feats_this = {}
            for canon in CANON_FEATURES:
                step_map = feats_dict.get(canon)
                if step_map is None:
                    continue
                for idx, pct in VIDAR_INDEX_TO_PCT.items():
                    t = step_map.get(idx)
                    if t is None:
                        continue
                    ep_feats_this[(canon, pct)] = _mean_pool(t)

            if len(ep_feats_this) == 0:
                continue
            trajs.append(seq)
            quats.append(_traj_quats_from_dual_euler(seq))
            for k, v in ep_feats_this.items():
                feats.setdefault(k, []).append(v)
            metas.append({"model": "vidar", "task": task, "episode": ep})

    return trajs, quats, feats, metas


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def compute_for_model(model_name, trajs, quats, feats, k=TOPK, random_seeds=10):
    """Returns list of dicts with cknna + cos_dist + random baseline per cell."""
    N = len(trajs)
    if N < 5:
        print(f"  [{model_name}] only {N} episodes, skipping")
        return []

    t0 = time.time()
    print(f"  [{model_name}] DTW top-{k} over N={N}...")
    dtw_topk = compute_dtw_topk_gpu(trajs, quats, topk=k)
    print(f"  [{model_name}] DTW done in {time.time()-t0:.1f}s")

    rows = []
    for feat_name in CANON_FEATURES:
        for pct in STEP_PCTS:
            key = (feat_name, pct)
            if key not in feats:
                rows.append({
                    "model": model_name, "feature": feat_name, "step_pct": pct,
                    "cknna": float("nan"), "cos_dist": float("nan"),
                    "n_eps": N, "n_with_feat": 0,
                    "random_mean": float("nan"), "random_std": float("nan"),
                })
                continue
            vecs = feats[key]
            if len(vecs) != N:
                # Only episodes where feature was captured. DTW top-k was built
                # from ALL N trajectories; fairness-wise we need to align by
                # restricting DTW top-k to the subset that has the feature.
                # For simplicity, warn and skip if the gap is large (>5%).
                frac = len(vecs) / N
                if frac < 0.95:
                    print(f"    WARN {feat_name}@{pct}%: only {len(vecs)}/{N} eps have feature")
                # Truncate trajectories isn't trivial without re-running DTW;
                # skip this cell rather than bias.
                rows.append({
                    "model": model_name, "feature": feat_name, "step_pct": pct,
                    "cknna": float("nan"), "cos_dist": float("nan"),
                    "n_eps": N, "n_with_feat": len(vecs),
                    "random_mean": float("nan"), "random_std": float("nan"),
                })
                continue

            mat = np.stack(vecs)        # (N, 3072)
            cknna = asymmetric_platonic_cknna_gpu(mat, dtw_topk, k=k)
            normed = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
            cs = 1.0 - normed @ normed.T
            np.fill_diagonal(cs, np.nan)
            cos_dist = float(np.nanmean(cs))

            # Random baseline
            rng = np.random.default_rng(0)
            rand_scores = []
            D = mat.shape[1]
            for seed in range(random_seeds):
                rng2 = np.random.default_rng(seed + 12345)
                rand_mat = rng2.standard_normal(size=(N, D)).astype(np.float32)
                rand_scores.append(asymmetric_platonic_cknna_gpu(rand_mat, dtw_topk, k=k))
            rand_arr = np.array(rand_scores)

            rows.append({
                "model": model_name, "feature": feat_name, "step_pct": pct,
                "cknna": float(cknna), "cos_dist": cos_dist,
                "n_eps": N, "n_with_feat": len(vecs),
                "random_mean": float(rand_arr.mean()), "random_std": float(rand_arr.std()),
            })
            print(f"    {feat_name:8s}@{pct:3d}%  CKNNA={cknna:+.4f}  "
                  f"random={rand_arr.mean():+.4f}±{rand_arr.std():.4f}  "
                  f"cos_dist={cos_dist:.5f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motus_root", type=Path)
    ap.add_argument("--lbva_root", type=Path)
    ap.add_argument("--lbva_dense_root", type=Path)
    ap.add_argument("--vidar_server_root", type=Path)
    ap.add_argument("--vidar_client_root", type=Path)
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Optional task-name filter (e.g. adjust_bottle click_bell)")
    ap.add_argument("--k", type=int, default=TOPK)
    ap.add_argument("--random_seeds", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    all_rows = []

    if args.motus_root:
        print(f"=== Loading Motus from {args.motus_root} ===")
        trajs, quats, feats, metas = load_motus(args.motus_root, args.tasks)
        print(f"  loaded {len(trajs)} episodes, feature keys: {len(feats)}")
        all_rows += compute_for_model("motus", trajs, quats, feats,
                                      k=args.k, random_seeds=args.random_seeds)

    if args.lbva_root and args.lbva_dense_root:
        print(f"=== Loading LB-VA from {args.lbva_root} + {args.lbva_dense_root} ===")
        trajs, quats, feats, metas = load_lbva(args.lbva_root, args.lbva_dense_root, args.tasks)
        print(f"  loaded {len(trajs)} episodes, feature keys: {len(feats)}")
        all_rows += compute_for_model("lbva_v25", trajs, quats, feats,
                                      k=args.k, random_seeds=args.random_seeds)

    if args.vidar_server_root and args.vidar_client_root:
        print(f"=== Loading Vidar from {args.vidar_server_root} + {args.vidar_client_root} ===")
        trajs, quats, feats, metas = load_vidar(args.vidar_server_root, args.vidar_client_root, args.tasks)
        print(f"  loaded {len(trajs)} episodes, feature keys: {len(feats)}")
        all_rows += compute_for_model("vidar", trajs, quats, feats,
                                      k=args.k, random_seeds=args.random_seeds)

    # Save CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "feature", "step_pct", "cknna", "cos_dist",
                  "n_eps", "n_with_feat", "random_mean", "random_std"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
