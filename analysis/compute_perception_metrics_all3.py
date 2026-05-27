"""Cross-model perception metrics for Motus + LingBot-VA (V25/V5) + VIDAR.

Two protocols:
  --protocol natural     Every chunk × every pred offset × each model's own GT.
                         Reproduces the Motus-vs-LB-VA "inversion" story.
  --protocol controlled  Chunk 0 + t∈{4,8} + head-crop 256×320.
                         Reproduces the 2026-04-07 "saturation" story.

Metrics:
  MSE, SSIM, PSNR    (per-pair pixel distortion)
  CMMD               (distributional perceptual distance, Jayasumana et al. 2024).
                     Replaces FID because FID is rank-deficient below ~10K samples
                     and violates its Gaussian assumption on Inception embeddings.

FVD is intentionally dropped. Cross-model clip-length alignment (Motus 9 / LB-VA 5 /
VIDAR 61) is intractable, and content-debiased-FVD (Ge et al. 2024) shows classical
FVD is dominated by per-frame appearance anyway.

Usage:
  PYTHONNOUSERSITE=1 python compute_perception_metrics_all3.py \
      --protocol controlled \
      --motus_dir   /lambda/nfs/vla/motus/logs_easy_turnswitch/images/turn_switch/dense_data \
      --lbva_v25_dir /lambda/nfs/vla/lingbot-va/decoded/turn_switch_v25 \
      --lbva_v5_dir  /lambda/nfs/vla/lingbot-va/decoded/turn_switch_v5 \
      --vidar_dir    /lambda/nfs/vla/vidar-robotwin/vidar_capture/turn_switch \
      --task turn_switch \
      --out_json all3_controlled_turn_switch.json

Sanity check (no VIDAR yet):
  Protocol natural should reproduce chunk0-fair-all3.tex MSE ≈ 319 vs 41.
  Protocol controlled should reproduce 2026-04-07 Motus 13.8 < V25 33.1 ≈ V5 32.1.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
METRIC_SIZE = 256       # canonical height for all metrics
METRIC_WIDTH = 320      # canonical width
# Motus T-shape head-cam region (empirically verified in 2026-03-25 notes).
MOTUS_HEAD_TOP = 12
MOTUS_HEAD_BOTTOM = 252
FRAME_WIDTH = 320
# VIDAR 640×736 T-shape: head is the top portion, 480px tall × 640 wide.
VIDAR_HEAD_TOP = 0
VIDAR_HEAD_BOTTOM = 480
VIDAR_HEAD_WIDTH = 640


# ---------------------------------------------------------------------
# Pixel-distortion primitives
# ---------------------------------------------------------------------
def compute_ssim(img1: np.ndarray, img2: np.ndarray, win_size: int = 7) -> float:
    g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float64)
    g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(g1, (win_size, win_size), 1.5)
    mu2 = cv2.GaussianBlur(g2, (win_size, win_size), 1.5)
    s1 = cv2.GaussianBlur(g1 ** 2, (win_size, win_size), 1.5) - mu1 ** 2
    s2 = cv2.GaussianBlur(g2 ** 2, (win_size, win_size), 1.5) - mu2 ** 2
    s12 = cv2.GaussianBlur(g1 * g2, (win_size, win_size), 1.5) - mu1 * mu2
    num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2)
    return float((num / den).mean())


def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


# ---------------------------------------------------------------------
# CMMD (Jayasumana et al., Rethinking FID, arXiv 2401.09603)
# ---------------------------------------------------------------------
def _gaussian_rbf_mmd2(x: np.ndarray, y: np.ndarray, sigma: float = 10.0) -> float:
    """Unbiased CMMD^2 with Gaussian RBF kernel k(a, b) = exp(-||a-b||^2 / (2 sigma^2)).

    Expects L2-normalized CLIP-style embeddings, (N, D).
    """
    gamma = 1.0 / (2.0 * sigma * sigma)
    xx = x @ x.T
    yy = y @ y.T
    xy = x @ y.T
    dx = np.diag(xx)[:, None] + np.diag(xx)[None, :] - 2 * xx
    dy = np.diag(yy)[:, None] + np.diag(yy)[None, :] - 2 * yy
    dxy = np.diag(xx)[:, None] + np.diag(yy)[None, :] - 2 * xy

    m, n = x.shape[0], y.shape[0]
    kxx = np.exp(-gamma * dx)
    kyy = np.exp(-gamma * dy)
    kxy = np.exp(-gamma * dxy)

    # Unbiased estimator: exclude diagonal for within-sample terms.
    kxx_off = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    kyy_off = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
    kxy_mean = kxy.sum() / (m * n)
    return float(kxx_off + kyy_off - 2 * kxy_mean)


def compute_cmmd(pred_frames: Iterable[np.ndarray],
                 gt_frames: Iterable[np.ndarray],
                 model_name: str = "ViT-L/14") -> float:
    """Encode each frame via CLIP image encoder, L2-normalize, then MMD^2.

    Lazily imports torch + clip inside the conda env. Runs on cuda:0 if available.
    """
    import torch
    import clip  # pip install ftfy regex git+https://github.com/openai/CLIP.git
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    def embed(frames):
        # Collectors return uint8 RGB ndarray. CLIP preprocess expects PIL.Image.
        pils = [Image.fromarray(f) if isinstance(f, np.ndarray) else f for f in frames]
        feats = []
        bs = 64
        for i in range(0, len(pils), bs):
            batch = torch.stack([preprocess(p) for p in pils[i:i + bs]]).to(device)
            with torch.no_grad():
                f = model.encode_image(batch)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
        return torch.cat(feats, dim=0).numpy().astype(np.float64)

    x = embed(pred_frames)
    y = embed(gt_frames)
    # 1000x scale factor matches the CMMD paper's convention.
    return 1000.0 * _gaussian_rbf_mmd2(x, y, sigma=10.0)


# ---------------------------------------------------------------------
# Frame loaders (model-specific). Each returns (pred_frame, gt_frame) pairs,
# already head-cropped and resized to METRIC_SIZE × METRIC_WIDTH, uint8 RGB.
# ---------------------------------------------------------------------
def _resize_canonical(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.resize(img_rgb, (METRIC_WIDTH, METRIC_SIZE),
                      interpolation=cv2.INTER_AREA)


def _motus_head_crop(full_frame_rgb: np.ndarray) -> np.ndarray:
    """Motus T-shape 384×320; head cam is rows 12:252 (see 2026-03-25:1542)."""
    return full_frame_rgb[MOTUS_HEAD_TOP:MOTUS_HEAD_BOTTOM, :FRAME_WIDTH, :]


def _vidar_head_crop(full_frame_rgb: np.ndarray) -> np.ndarray:
    """VIDAR 3-cam T-shape 736×640: head is the top 480×640."""
    return full_frame_rgb[VIDAR_HEAD_TOP:VIDAR_HEAD_BOTTOM, :VIDAR_HEAD_WIDTH, :]


# Collectors are stubs to be filled post-smoke. They mirror the contract of
# lingbot-va/compute_metrics_same_horizon.py but produce canonical-size uint8.
def collect_motus(data_dir: Path, task_filter: str, protocol: str):
    """Yield (pred_np, gt_np, meta_dict).

    data_dir = .../logs_50x10_<ts>/images containing per-task subdirs.
    Each task dir has:
      - `latents/pred_frames_<ep>_<chunk>.pt` (8, 3, 384, 320) fp32 in [0,1]
        — all 8 predicted future frames at 2-action spacing (t=2..16).
      - `latents/cond_frame_<ep>_<chunk>.pt` (not used here).
      - PNG grids `episode_<EEEE>_step_<CCCC>.png` (1600x384, cond+4 pred) —
        used only as fallback when the .pt tensor is absent.
      - `dense_data/<ep>_dense_obs.pkl` with cam_high at env-steps {3, 7, 11, 15}
        of each 16-step chunk, giving GT keyframes at t≈{4, 8, 12, 16}.

    Protocol B: chunk 0 only, pairs at t=4 and t=8 (pred[1]↔kf[0], pred[3]↔kf[1]).
    Protocol A: every chunk, pairs at t=4, 8, 12, 16 (pred[1,3,5,7] ↔ kf[0,1,2,3]).

    Using .pt instead of PNG grid lets Protocol A reach Motus's full native
    prediction horizon (t=16), matching LB-VA's t=16 coverage.
    """
    import pickle
    import torch

    data_dir = Path(data_dir)

    task_dirs = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name not in {"dense_data", "latents"}
    )
    if task_filter:
        task_dirs = [d for d in task_dirs if task_filter.lower() in d.name.lower()]

    # (pred_idx_in_8frame_tensor, kf_idx, t_action_step)
    # Protocol A: all four; Protocol B: first two only.
    FULL_PAIRS = ((1, 0, 4), (3, 1, 8), (5, 2, 12), (7, 3, 16))

    def _tensor_pred_frame(pred_t, pred_idx):
        """(8,3,384,320) fp32 [0,1] → head-cropped (240,320,3) uint8 RGB."""
        arr = pred_t[pred_idx].numpy()                    # (3, 384, 320)
        arr = (arr.clip(0.0, 1.0) * 255.0).astype(np.uint8)
        arr = arr[:, MOTUS_HEAD_TOP:MOTUS_HEAD_BOTTOM, :]  # (3, 240, 320)
        return arr.transpose(1, 2, 0)                      # (240, 320, 3)

    def _grid_pred_frame(grid_rgb, grid_idx):
        """PNG-grid fallback: extract (240, 320, 3) uint8 from the 5-cell grid."""
        cell = grid_rgb[:, grid_idx * FRAME_WIDTH:(grid_idx + 1) * FRAME_WIDTH, :]
        if cell.shape[1] != FRAME_WIDTH:
            return None
        return cell[MOTUS_HEAD_TOP:MOTUS_HEAD_BOTTOM, :, :]

    for task_dir in task_dirs:
        task = task_dir.name
        dense_dir = task_dir / "dense_data"
        latents_dir = task_dir / "latents"
        if not dense_dir.is_dir():
            continue

        ep_pkls = sorted(
            dense_dir.glob("*_dense_obs.pkl"),
            key=lambda p: int(p.name.split("_")[0]),
        )
        for pkl in ep_pkls:
            try:
                ep = int(pkl.name.split("_")[0])
            except ValueError:
                continue
            try:
                with open(pkl, "rb") as f:
                    dense_obs = pickle.load(f)
            except Exception:
                continue
            if len(dense_obs) < 16:
                continue

            n_chunks = len(dense_obs) // 16
            chunks = [0] if protocol == "controlled" else list(range(n_chunks))
            active_pairs = FULL_PAIRS if protocol == "natural" else FULL_PAIRS[:2]

            for c in chunks:
                start, end = c * 16, c * 16 + 16
                if end > len(dense_obs):
                    continue
                kfs = [
                    d["cam_high"] for d in dense_obs[start:end]
                    if isinstance(d, dict) and d.get("cam_high") is not None
                ]
                if len(kfs) < 2:
                    continue

                # Prefer .pt tensor (full 8-frame). Fall back to PNG grid.
                pt_path = latents_dir / f"pred_frames_{ep}_{c}.pt"
                pred_source = None
                if pt_path.exists():
                    try:
                        pred_t = torch.load(pt_path, map_location="cpu",
                                            weights_only=False)
                        if (pred_t.ndim == 4 and pred_t.shape[0] >= 8
                                and pred_t.shape[2:] == (384, 320)):
                            pred_source = ("pt", pred_t)
                    except Exception:
                        pass
                if pred_source is None:
                    png = task_dir / f"episode_{ep:04d}_step_{c:04d}.png"
                    if not png.exists():
                        continue
                    bgr = cv2.imread(str(png))
                    if bgr is None:
                        continue
                    pred_source = ("grid", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

                for pred_idx, kf_idx, t_step in active_pairs:
                    if kf_idx >= len(kfs):
                        continue
                    # grid fallback only has 4 pred cells (indices 1..4 in the
                    # 5-cell grid where idx 0 is cond). Map t=4→grid 2, t=8→grid 4.
                    # t=12 and t=16 are unavailable via grid path → skip.
                    if pred_source[0] == "grid":
                        grid_idx_map = {4: 2, 8: 4}
                        if t_step not in grid_idx_map:
                            continue
                        pred_cam = _grid_pred_frame(pred_source[1],
                                                    grid_idx_map[t_step])
                    else:
                        pred_cam = _tensor_pred_frame(pred_source[1], pred_idx)
                    if pred_cam is None:
                        continue
                    gt_cam = kfs[kf_idx]
                    if pred_cam.shape != gt_cam.shape:
                        gt_cam = cv2.resize(
                            gt_cam, (pred_cam.shape[1], pred_cam.shape[0]),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    pred_c = _resize_canonical(pred_cam)
                    gt_c = _resize_canonical(gt_cam)
                    yield (pred_c, gt_c,
                           {"task": task, "episode": ep, "chunk": c,
                            "pred_idx": pred_idx, "kf_idx": kf_idx,
                            "t_step": t_step, "source": pred_source[0]})


def collect_lingbot(data_dir: Path, task_filter: str, protocol: str):
    """Return list[(pred_np, gt_np, meta_dict)].

    data_dir = .../visualization_lingbot_50x10/real/<prompt>_<ts>/ one per
    (task, episode). Each dir has `decoded_frames/pred_chunk<N>_frame<F>.npy`
    (256x320 cam_high, uint8) and `gt_chunk<N>_obs<O>.npy` (240x320 uint8).

    LB-VA dir names are natural-language prompts, not task names. task_filter
    is a substring match on the dir name; pass empty for all.

    Protocol B: chunk 0, frame1 (t=4) ↔ obs0, frame2 (t=8) ↔ obs1.
    Protocol A: every chunk, frame{1,2,3,4} ↔ obs{0,1,2,3} (skip teacher-
    forced frame0 at t=0).
    """
    data_dir = Path(data_dir)

    ep_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    if task_filter:
        ep_dirs = [d for d in ep_dirs if task_filter.lower() in d.name.lower()]

    for ep_dir in ep_dirs:
        dec = ep_dir / "decoded_frames"
        if not dec.is_dir():
            continue

        if protocol == "controlled":
            mappings = [(0, 1, 0), (0, 2, 1)]
        else:
            chunks = set()
            for p in dec.glob("pred_chunk*_frame*.npy"):
                m = re.match(r"pred_chunk(\d+)_frame\d+\.npy", p.name)
                if m:
                    chunks.add(int(m.group(1)))
            mappings = []
            for c in sorted(chunks):
                mappings.extend([(c, 1, 0), (c, 2, 1), (c, 3, 2), (c, 4, 3)])

        for chunk, pred_idx, gt_idx in mappings:
            pred_path = dec / f"pred_chunk{chunk}_frame{pred_idx}.npy"
            gt_path = dec / f"gt_chunk{chunk}_obs{gt_idx}.npy"
            if not pred_path.exists() or not gt_path.exists():
                continue
            try:
                pred = np.load(pred_path)   # 256x320x3 uint8 (already cam_high)
                gt_raw = np.load(gt_path)   # 240x320x3 uint8
            except Exception:
                continue
            if pred.shape[:2] != gt_raw.shape[:2]:
                gt_raw = cv2.resize(
                    gt_raw, (pred.shape[1], pred.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            pred_c = _resize_canonical(pred)
            gt_c = _resize_canonical(gt_raw)
            yield (pred_c, gt_c,
                   {"prompt_dir": ep_dir.name, "chunk": chunk,
                    "pred_idx": pred_idx, "gt_idx": gt_idx})


def collect_vidar(server_root: Path, client_root: Path, task_filter: str, protocol: str):
    """Return list[(pred_np, gt_np, meta_dict)].

    Vidar's captures span two roots (server writes hook + pred, client writes GT):
      SERVER: {server_root}/<task>/episode<NNNN>/pred_frames/chunk<CC>/f<FFF>.png
      CLIENT: {client_root}/<task>/<task>/episode<NNNN>/gt_frames/chunk<CC>/t<SSS>_head.png
              (older runs may use a flatter {client_root}/<task>/episode<NNNN>/gt_frames/...)

    Protocol B: chunk 0 only, f[3] ↔ t004 and f[7] ↔ t008.
    Protocol A: every chunk, only pred offsets that match exact GT keyframes
                {0, 4, 8, 16, 32, 48, 60} — skips intermediate frames that
                would map to the same GT via nearest-neighbor, which would
                inflate n_pairs without adding signal.
    """
    task_dirs = sorted(d for d in server_root.iterdir() if d.is_dir())
    if task_filter:
        task_dirs = [d for d in task_dirs if task_filter.lower() in d.name.lower()]

    for task_dir in task_dirs:
        task = task_dir.name
        ep_dirs = sorted(
            d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("episode")
        )
        for ep_dir in ep_dirs:
            pred_root = ep_dir / "pred_frames"
            if not pred_root.is_dir():
                continue
            # Client GT path — try nested first (<task>/<task>/), fall back to flat.
            client_nested = client_root / task / task / ep_dir.name / "gt_frames"
            client_flat = client_root / task / ep_dir.name / "gt_frames"
            gt_root = client_nested if client_nested.is_dir() else client_flat
            if not gt_root.is_dir():
                continue

            for chunk_dir in sorted(pred_root.glob("chunk*")):
                m_c = re.search(r"chunk(\d+)", chunk_dir.name)
                if not m_c:
                    continue
                chunk_id = int(m_c.group(1))
                if protocol == "controlled" and chunk_id != 0:
                    continue
                gt_chunk_dir = gt_root / chunk_dir.name
                if not gt_chunk_dir.is_dir():
                    continue
                gt_by_step = {}
                for p in gt_chunk_dir.glob("t*_head.png"):
                    m = re.match(r"t(\d+)_head\.png", p.name)
                    if m:
                        gt_by_step[int(m.group(1))] = p
                if not gt_by_step:
                    continue

                if protocol == "controlled":
                    # f[3] ↔ t=4, f[7] ↔ t=8
                    pairs_to_emit = [(3, 4), (7, 8)]
                else:
                    # Natural: pair each pred offset with its EXACT GT step
                    # that was saved at the same action-step.
                    pairs_to_emit = [
                        (step, step) for step in (0, 4, 8, 16, 32, 48, 60)
                    ]

                for pred_off, gt_step in pairs_to_emit:
                    if gt_step not in gt_by_step:
                        continue
                    pred_path = chunk_dir / f"f{pred_off:03d}.png"
                    if not pred_path.exists():
                        continue
                    pred = cv2.imread(str(pred_path), cv2.IMREAD_COLOR)
                    gt = cv2.imread(str(gt_by_step[gt_step]), cv2.IMREAD_COLOR)
                    if pred is None or gt is None:
                        continue
                    pred = pred[..., ::-1]
                    gt = gt[..., ::-1]
                    pred_head = _resize_canonical(_vidar_head_crop(pred))
                    gt_head = _resize_canonical(gt)
                    yield (
                        pred_head, gt_head,
                        {"task": task, "chunk": chunk_id,
                         "pred_offset": pred_off, "gt_step": gt_step,
                         "episode": ep_dir.name},
                    )


# ---------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------
@dataclass
class MetricSummary:
    model: str
    protocol: str
    n_pairs: int
    mse: float
    ssim: float
    psnr: float
    cmmd: float
    fid: float

    def to_dict(self):
        return self.__dict__


def aggregate(pairs, model: str, protocol: str, cmmd_on: bool = True,
              fid_on: bool = True, batch_size: int = 64) -> MetricSummary:
    """Streaming aggregation over a pair generator.

    MSE/SSIM/PSNR: running-sum, O(1) memory.
    CMMD: CLIP ViT-L/14 embeddings [N, 768] buffered, then RBF-MMD² at end.
    FID:  clean-fid Inception-v3 pool3 features [N, 2048] buffered, then
          frechet_distance at end.

    All three use the same pair stream; images are never held as a full list.
    pairs may be a generator or a list.
    """
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model = clip_preprocess = None
    if cmmd_on:
        import clip as _clip  # noqa
        from PIL import Image  # noqa
        clip_model, clip_preprocess = _clip.load("ViT-L/14", device=device)
        clip_model.eval()

    inception_feat = None
    if fid_on:
        from cleanfid.features import build_feature_extractor
        inception_feat = build_feature_extractor("clean", device)

    n = 0
    mse_sum = ssim_sum = psnr_sum = 0.0
    pred_batch: list = []
    gt_batch: list = []
    clip_pred_embs: list = []
    clip_gt_embs: list = []
    inc_pred_embs: list = []
    inc_gt_embs: list = []

    def _clip_encode(bufs):
        from PIL import Image
        pils = [Image.fromarray(f) for f in bufs]
        batch = torch.stack([clip_preprocess(p) for p in pils]).to(device)
        with torch.no_grad():
            f = clip_model.encode_image(batch)
        f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy().astype(np.float64)

    def _inception_encode(bufs):
        # clean-fid's Inception expects [B, 3, 299, 299] in [0, 255] float.
        # Resize each frame with bicubic first (matches clean-fid's "clean" mode).
        resized = np.stack([
            cv2.resize(f, (299, 299), interpolation=cv2.INTER_CUBIC)
            for f in bufs
        ]).astype(np.float32)                                # [B, 299, 299, 3]
        t = torch.from_numpy(resized).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            f = inception_feat(t)
        return f.cpu().numpy().astype(np.float64)

    def flush():
        nonlocal pred_batch, gt_batch
        if not pred_batch:
            return
        if cmmd_on:
            clip_pred_embs.append(_clip_encode(pred_batch))
            clip_gt_embs.append(_clip_encode(gt_batch))
        if fid_on:
            inc_pred_embs.append(_inception_encode(pred_batch))
            inc_gt_embs.append(_inception_encode(gt_batch))
        pred_batch = []
        gt_batch = []

    for pred, gt, _meta in pairs:
        mse_sum += compute_mse(pred, gt)
        ssim_sum += compute_ssim(pred, gt)
        psnr_sum += compute_psnr(pred, gt)
        n += 1
        if cmmd_on or fid_on:
            pred_batch.append(pred)
            gt_batch.append(gt)
            if len(pred_batch) >= batch_size:
                flush()

    if n == 0:
        return MetricSummary(model, protocol, 0, float("nan"),
                             float("nan"), float("nan"), float("nan"),
                             float("nan"))

    flush()

    cmmd = float("nan")
    if cmmd_on and clip_pred_embs:
        X = np.concatenate(clip_pred_embs, axis=0)
        Y = np.concatenate(clip_gt_embs, axis=0)
        if len(X) >= 4:
            cmmd = 1000.0 * _gaussian_rbf_mmd2(X, Y, sigma=10.0)

    fid = float("nan")
    if fid_on and inc_pred_embs:
        from cleanfid.fid import frechet_distance
        Xi = np.concatenate(inc_pred_embs, axis=0)
        Yi = np.concatenate(inc_gt_embs, axis=0)
        if len(Xi) >= 4:
            mu1, sigma1 = Xi.mean(axis=0), np.cov(Xi, rowvar=False)
            mu2, sigma2 = Yi.mean(axis=0), np.cov(Yi, rowvar=False)
            fid = float(frechet_distance(mu1, sigma1, mu2, sigma2))

    return MetricSummary(
        model=model, protocol=protocol, n_pairs=n,
        mse=mse_sum / n, ssim=ssim_sum / n,
        psnr=psnr_sum / n, cmmd=cmmd, fid=fid,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["natural", "controlled"],
                    required=True)
    ap.add_argument("--task", default="",
                    help="Task-name substring filter (empty = all tasks)")
    ap.add_argument("--motus_dir", type=Path)
    ap.add_argument("--lbva_v25_dir", type=Path)
    ap.add_argument("--lbva_v5_dir", type=Path)
    ap.add_argument("--vidar_server_dir", type=Path,
                    help="Server-side pred/hook root (replaces --vidar_dir)")
    ap.add_argument("--vidar_client_dir", type=Path,
                    help="Client-side GT root (eval_result/ar/vidar_full)")
    ap.add_argument("--no_cmmd", action="store_true",
                    help="Skip CMMD (useful for quick smoke without CLIP weights)")
    ap.add_argument("--no_fid", action="store_true",
                    help="Skip FID (useful for quick smoke without Inception weights)")
    ap.add_argument("--out_json", type=Path, default=Path("metrics_all3.json"))
    args = ap.parse_args()

    cmmd_on = not args.no_cmmd
    fid_on = not args.no_fid
    results = {}

    if args.vidar_server_dir and args.vidar_client_dir:
        pairs = collect_vidar(args.vidar_server_dir, args.vidar_client_dir,
                              args.task, args.protocol)
        results["vidar"] = aggregate(pairs, "vidar", args.protocol, cmmd_on, fid_on).to_dict()

    if args.motus_dir:
        try:
            pairs = collect_motus(args.motus_dir, args.task, args.protocol)
            results["motus"] = aggregate(pairs, "motus", args.protocol, cmmd_on, fid_on).to_dict()
        except NotImplementedError as e:
            results["motus"] = {"error": str(e)}

    if args.lbva_v25_dir:
        try:
            pairs = collect_lingbot(args.lbva_v25_dir, args.task, args.protocol)
            results["lbva_v25"] = aggregate(pairs, "lbva_v25", args.protocol, cmmd_on, fid_on).to_dict()
        except NotImplementedError as e:
            results["lbva_v25"] = {"error": str(e)}

    if args.lbva_v5_dir:
        try:
            pairs = collect_lingbot(args.lbva_v5_dir, args.task, args.protocol)
            results["lbva_v5"] = aggregate(pairs, "lbva_v5", args.protocol, cmmd_on, fid_on).to_dict()
        except NotImplementedError as e:
            results["lbva_v5"] = {"error": str(e)}

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
