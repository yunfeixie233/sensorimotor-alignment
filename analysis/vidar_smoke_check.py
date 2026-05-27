"""Post-smoke-run sanity checks for VIDAR capture.

Run after `run_eval_client_only.sh` has completed at least 2 episodes with
SAVE_* flags on. Verifies:

  1. hook_features_chunk{CC}.pt has 20 step-indexed entries × 8 feature names
  2. pred_frames/chunk{CC}/*.png saved as PNG uint8 (not JPEG)
  3. gt_frames/chunk{CC}/t{SSS}_head.png at keyframes {0, 4, 8, 16, 32, 48, 60}
  4. dense_obs.pkl loadable with expected schema
  5. frame[0] teacher-forcing: if SSIM(f000.png, obs_input) > 0.99,
     Protocol B fair-window shifts from f[3]/[7] → f[4]/[8]

Usage:
  PYTHONNOUSERSITE=1 python vidar_smoke_check.py \
      --capture_dir /lambda/nfs/vla/vidar-robotwin/vidar_capture \
      --task adjust_bottle
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    g1 = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    g2 = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(g1, (7, 7), 1.5)
    mu2 = cv2.GaussianBlur(g2, (7, 7), 1.5)
    s1 = cv2.GaussianBlur(g1 ** 2, (7, 7), 1.5) - mu1 ** 2
    s2 = cv2.GaussianBlur(g2 ** 2, (7, 7), 1.5) - mu2 ** 2
    s12 = cv2.GaussianBlur(g1 * g2, (7, 7), 1.5) - mu1 * mu2
    return float(((2 * mu1 * mu2 + C1) * (2 * s12 + C2) /
                  ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))).mean())


def check_one_episode(ep_dir: Path, expected_cond_steps: int = 20) -> dict:
    import torch  # noqa

    result = {"episode": ep_dir.name, "errors": [], "warnings": []}

    # (1) hook features
    feat_files = sorted(ep_dir.glob("hook_features_chunk*.pt"))
    if not feat_files:
        result["errors"].append("no hook_features_chunk*.pt found")
    else:
        payload = torch.load(feat_files[0], map_location="cpu")
        feats = payload.get("features", {})
        cond_steps = payload.get("num_cond_steps", 0)
        result["hook_chunks"] = len(feat_files)
        result["num_cond_steps"] = cond_steps
        result["feature_names"] = sorted(feats.keys())
        if cond_steps != expected_cond_steps:
            result["warnings"].append(
                f"num_cond_steps={cond_steps} (expected {expected_cond_steps})"
            )
        expected_names = {"V-D1", "V-B0", "V-B7", "V-B14", "V-B21", "V-B29",
                          "V-Norm", "T-D1"}
        if set(feats.keys()) != expected_names:
            result["errors"].append(
                f"feature names mismatch: got {set(feats.keys())}"
            )
        # Shape + NaN check on first-chunk last-step entry.
        for name, step_dict in feats.items():
            if not step_dict:
                result["errors"].append(f"{name}: empty step dict")
                continue
            last = max(step_dict.keys())
            t = step_dict[last]
            if t.dim() != 1 or t.shape[0] != 3072:
                result["warnings"].append(f"{name}: shape {tuple(t.shape)}")
            if torch.isnan(t).any():
                result["errors"].append(f"{name}: NaN in last-step feature")

    # (2) pred frames are PNG
    pred_root = ep_dir / "pred_frames"
    if not pred_root.is_dir():
        result["errors"].append("pred_frames/ missing")
    else:
        sample = next(pred_root.rglob("f*.png"), None)
        if sample is None:
            result["errors"].append("no pred PNGs found")
        else:
            head = sample.read_bytes()[:8]
            if head[:4] != b"\x89PNG":
                result["errors"].append(f"{sample} is not a PNG")
            result["pred_frames_sample"] = str(sample)

    # (3) GT frames exist at keyframes
    gt_root = ep_dir / "gt_frames"
    if not gt_root.is_dir():
        result["errors"].append("gt_frames/ missing")
    else:
        chunk0 = gt_root / "chunk00"
        if chunk0.is_dir():
            steps = sorted(
                int(p.stem.split("_")[0][1:])
                for p in chunk0.glob("t*_head.png")
            )
            result["gt_chunk0_steps"] = steps
            expected = {0, 4, 8, 16, 32, 48, 60}
            missing = expected - set(steps)
            if missing:
                result["warnings"].append(f"gt chunk0 missing steps {missing}")

    # (4) dense_obs.pkl
    dense_path = ep_dir / "dense_obs.pkl"
    if dense_path.exists():
        try:
            with open(dense_path, "rb") as f:
                dense = pickle.load(f)
            result["dense_obs_keys"] = len(dense)
        except Exception as e:
            result["errors"].append(f"dense_obs.pkl load failed: {e}")
    else:
        result["warnings"].append("dense_obs.pkl missing")

    # (5) frame[0] teacher-forcing test
    pred_f000 = ep_dir / "pred_frames" / "chunk00" / "f000.png"
    gt_t000 = ep_dir / "gt_frames" / "chunk00" / "t000_head.png"
    if pred_f000.exists() and gt_t000.exists():
        pred = cv2.imread(str(pred_f000))[..., ::-1]
        gt = cv2.imread(str(gt_t000))[..., ::-1]
        # Head-crop VIDAR pred to match GT resolution.
        pred_head = pred[:480, :640, :]
        gt_resized = cv2.resize(gt, (pred_head.shape[1], pred_head.shape[0]),
                                interpolation=cv2.INTER_AREA)
        s = _ssim(pred_head, gt_resized)
        result["frame0_ssim_vs_obs"] = s
        if s > 0.99:
            result["frame0_teacher_forced"] = True
            result["warnings"].append(
                "frame[0] SSIM > 0.99: Protocol B fair window should shift "
                "from f[3]/f[7] to f[4]/f[8]"
            )
        else:
            result["frame0_teacher_forced"] = False

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture_dir", type=Path, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--max_eps", type=int, default=2)
    args = ap.parse_args()

    task_dir = args.capture_dir / args.task
    if not task_dir.is_dir():
        print(f"ERROR: task dir missing: {task_dir}")
        return

    ep_dirs = sorted(task_dir.glob("episode*"))[: args.max_eps]
    reports = [check_one_episode(ep) for ep in ep_dirs]
    import json
    print(json.dumps(reports, indent=2, default=str))

    any_errors = any(r["errors"] for r in reports)
    if any_errors:
        print("\n[FAIL] at least one episode had errors")
        exit(1)
    print("\n[PASS] smoke test clean")


if __name__ == "__main__":
    main()
