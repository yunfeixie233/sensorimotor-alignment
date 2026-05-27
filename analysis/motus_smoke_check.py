#!/usr/bin/env python
"""Validate Motus smoke-run artifacts for the 2026-04-18 50x10 pre-flight.

Mirrors `cknna_project/analysis/vidar_smoke_check.py` — checks that the
Motus capture produces the schema Table A (layerwise-vs-denoising-step)
and Table B (chunk0-fair-all3) will consume.

Usage:
    python motus_smoke_check.py <log_dir>

<log_dir> should be the RoboTwin log_dir from smoke_1ep_adjust_bottle.sh
(e.g. logs_smoke_1ep_YYYYmmdd_HHMMSS). The script walks into the per-task
subdirectory and inspects:

  dense_data/0_hook_features.pt           # chunk-0 last-step features
  dense_data/0_hook_features_allsteps.pt  # per-chunk last-step features
  dense_data/0_denoise_trajectory.pt      # NEW: per-step block features
  dense_data/0_dense_obs.pkl              # per-4-step GT trajectory
  latents/pred_frames_0_*.pt              # 8-frame decoded pred (all chunks)
  latents/cond_frame_0_*.pt               # per-chunk condition frame
  latents/latents_0_*.pt                  # raw video latent
  episode_0000_step_*.png                 # frame grid (cond + 4 pred)

Prints PASS/FAIL and a per-artifact summary.
"""
import re
import sys
import pickle
from pathlib import Path

import torch


EXPECTED_FEATURES = {
    'V-D1', 'V-B0', 'V-B7', 'V-B14', 'V-B21', 'V-B29', 'V-Norm',
    'A-D1', 'A-B0', 'A-B7', 'A-B14', 'A-B21', 'A-B29', 'A-Norm',
    'T-D1', 'U-B14', 'U-B29',
}
TRAJECTORY_BLOCKS = {'V-D1', 'V-B0', 'V-B7', 'V-B14', 'V-B21', 'V-B29', 'V-Norm', 'T-D1'}


def find_capture_dir(log_dir: Path) -> Path:
    """Locate the dense_data/latents/grid root."""
    candidates = list(log_dir.rglob('dense_data'))
    if not candidates:
        raise SystemExit(f"No dense_data/ dir found under {log_dir}")
    for c in candidates:
        if any(c.glob('*_hook_features.pt')):
            return c.parent  # parent holds dense_data/ + latents/ + frame grids
    return candidates[0].parent


def detect_episode_id(cap: Path) -> int:
    """Motus uses episode_count which may be 0 or 1 at smoke. Auto-detect."""
    files = list((cap / 'dense_data').glob('*_hook_features.pt'))
    if not files:
        return 0
    m = re.match(r'(\d+)_hook_features\.pt', files[0].name)
    return int(m.group(1)) if m else 0


def check_hook_features(p: Path, ep: int) -> bool:
    ok = True
    f = p / "dense_data" / f"{ep}_hook_features.pt"
    if not f.exists():
        print(f"  [FAIL] missing {f}")
        return False
    d = torch.load(f, weights_only=True)
    keys = set(d.keys())
    missing = EXPECTED_FEATURES - keys
    extra = keys - EXPECTED_FEATURES
    dims = {k: tuple(v.shape) for k, v in d.items()}
    print(f"  [hook_features]        keys={len(keys)}, missing={sorted(missing) or 'none'}, "
          f"extra={sorted(extra) or 'none'}")
    print(f"    shapes: {dims}")
    if missing:
        ok = False
    for k, v in d.items():
        if torch.isnan(v).any() or torch.isinf(v).any():
            print(f"  [FAIL] {k} has NaN/Inf")
            ok = False
    return ok


def check_allsteps(p: Path, ep: int) -> int:
    f = p / "dense_data" / f"{ep}_hook_features_allsteps.pt"
    if not f.exists():
        print(f"  [WARN] missing {f}")
        return 0
    d = torch.load(f, weights_only=True)
    print(f"  [hook_features_allsteps] num_chunks={len(d)}, "
          f"keys_per_chunk={sorted(d[0]['features'].keys()) if d else []}")
    return len(d)


def check_denoise_trajectory(p: Path, ep: int) -> bool:
    f = p / "dense_data" / f"{ep}_denoise_trajectory.pt"
    if not f.exists():
        print(f"  [FAIL] missing {f} — SAVE_DENOISE_TRAJECTORY patch did not fire")
        return False
    d = torch.load(f, weights_only=True)
    print(f"  [denoise_trajectory]   num_steps={len(d)}")
    if not d:
        print(f"  [FAIL] empty trajectory list")
        return False
    step0 = d[0]
    feats = {k for k in step0.keys() if k not in {'step', 't'}}
    missing = TRAJECTORY_BLOCKS - feats
    print(f"    step entries: {[e.get('step') for e in d]}")
    print(f"    t values:     {[round(e.get('t', -1), 1) for e in d]}")
    print(f"    features at step[0]: {sorted(feats)}")
    if missing:
        print(f"  [FAIL] missing block features in trajectory: {sorted(missing)}")
        return False
    # Shape + finiteness spot-check
    for e in d:
        for k, v in e.items():
            if k in {'step', 't'}:
                continue
            if not isinstance(v, torch.Tensor):
                print(f"  [FAIL] {k} is not a tensor"); return False
            if tuple(v.shape) != (3072,):
                print(f"  [WARN] {k} shape {tuple(v.shape)} != (3072,)")
            if torch.isnan(v).any() or torch.isinf(v).any():
                print(f"  [FAIL] {k} at step {e.get('step')} has NaN/Inf")
                return False
    return True


def check_dense_obs(p: Path, ep: int) -> bool:
    f = p / "dense_data" / f"{ep}_dense_obs.pkl"
    if not f.exists():
        print(f"  [FAIL] missing {f}")
        return False
    with open(f, 'rb') as fh:
        d = pickle.load(fh)
    kf = [e for e in d if 'cam_high' in e and e.get('cam_high') is not None]
    print(f"  [dense_obs]            total_steps={len(d)}, keyframes_with_cams={len(kf)}")
    if not kf:
        print(f"  [FAIL] no keyframes with cam_high")
        return False
    e0 = kf[0]
    shape_h = e0['cam_high'].shape if e0.get('cam_high') is not None else None
    print(f"    cam_high shape: {shape_h}")
    print(f"    endpose_left dim: {len(e0.get('endpose_left', []))}, "
          f"endpose_right dim: {len(e0.get('endpose_right', []))}")
    return True


def check_latents(p: Path, ep: int) -> bool:
    lat_dir = p / "latents"
    if not lat_dir.exists():
        print(f"  [WARN] missing {lat_dir} — SAVE_LATENTS=1 not active")
        return False
    pred = sorted(lat_dir.glob(f'pred_frames_{ep}_*.pt'))
    cond = sorted(lat_dir.glob(f'cond_frame_{ep}_*.pt'))
    raw  = sorted(lat_dir.glob(f'latents_{ep}_*.pt'))
    print(f"  [latents]              pred={len(pred)}, cond={len(cond)}, raw_latent={len(raw)}")
    if not pred:
        return False
    p0 = torch.load(str(pred[0]), weights_only=True)
    print(f"    pred_frames_{ep}_0.pt shape={tuple(p0.shape)}, dtype={p0.dtype}, "
          f"range=[{p0.min():.3f}, {p0.max():.3f}]")
    return True


def check_frame_grids(p: Path, ep: int) -> int:
    grids = sorted(p.glob(f'episode_{ep:04d}_step_*.png'))
    print(f"  [frame_grids]          count={len(grids)}")
    return len(grids)


def main():
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(1)
    log_dir = Path(sys.argv[1]).resolve()
    print(f"[motus_smoke_check] log_dir = {log_dir}")
    cap = find_capture_dir(log_dir)
    ep = detect_episode_id(cap)
    print(f"[motus_smoke_check] capture root = {cap}")
    print(f"[motus_smoke_check] episode_id = {ep}\n")

    checks = {
        'hook_features':      check_hook_features(cap, ep),
        'denoise_trajectory': check_denoise_trajectory(cap, ep),
        'dense_obs':          check_dense_obs(cap, ep),
        'latents':            check_latents(cap, ep),
    }
    check_allsteps(cap, ep)
    check_frame_grids(cap, ep)

    print()
    passed = all(checks.values())
    verdict = "PASS" if passed else "FAIL"
    print(f"[motus_smoke_check] {verdict}")
    for k, v in checks.items():
        print(f"    {'+' if v else '-'} {k}")
    sys.exit(0 if passed else 2)


if __name__ == '__main__':
    main()
