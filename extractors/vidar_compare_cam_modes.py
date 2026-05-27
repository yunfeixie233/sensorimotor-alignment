"""
Quick N=100 comparison of 3 camera modes for Vidar CKNNA feature extraction.
Loads model ONCE, runs all 3 modes sequentially, then compares feature statistics
and a proxy CKNNA using cosine-distance kNN overlap with DTW kNN.

Modes:
  single:  cam1 only (320x180), raw DROID prompt
  3cam_A:  cam2 top + cam3 wrist + cam1 exterior (bottom-right type mismatch)
  3cam_B:  cam2 top + cam3 wrist + cam3 wrist dup (slot-type matched)
"""

import os, sys, time, json, gc
import torch
import numpy as np
from PIL import Image
from collections import defaultdict

VIDAR_ROOT = "/lambda/nfs/vla/cknna_project/repos/vidar"
if VIDAR_ROOT not in sys.path:
    sys.path.insert(0, VIDAR_ROOT)

import wan
from wan.configs import WAN_CONFIGS

WAN22_BASE = "/lambda/nfs/vla/pretrained_models/Wan2.2-TI2V-5B"
VIDAR_PT = "/lambda/nfs/vla/cknna_project/checkpoints/vidar/vidar.pt"
DROID_DATA = "/lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid"

PROMPT_PREFIX_VIDAR = (
    "The whole scene is in a realistic, industrial art style with three views: "
    "a fixed external camera, a movable wrist camera, and a side external camera. "
    "The franka robot is currently performing the following task: "
)

N = 100


# ─── Hook capture (same as extract_va_features_vidar_droid.py) ───────────
class VidarHookCapture:
    VIDEO_BLOCKS = [0, 7, 14, 21, 29]

    def __init__(self):
        self._buf = {}
        self._call_idx = 0
        self._handles = []

    def reset(self):
        self._buf = {}
        self._call_idx = 0

    def _feature_hook(self, name, kind="out"):
        def hook(module, inp, out=None):
            if self._call_idx % 2 != 0:
                return
            if kind == "pre":
                t = inp[0] if isinstance(inp, tuple) else inp
            else:
                t = out[0] if isinstance(out, tuple) else out
            if not isinstance(t, torch.Tensor):
                return
            self._buf[name] = t.detach().float().cpu()
        return hook

    def _counter_hook(self):
        def hook(module, inp, out):
            self._call_idx += 1
        return hook

    def register(self, wan_model):
        self._handles.append(wan_model.blocks[0].register_forward_pre_hook(
            self._feature_hook("V-D1", kind="pre")))
        for bi in self.VIDEO_BLOCKS:
            self._handles.append(wan_model.blocks[bi].register_forward_hook(
                self._feature_hook(f"V-B{bi}")))
        self._handles.append(wan_model.head.norm.register_forward_hook(
            self._feature_hook("V-Norm")))
        self._handles.append(wan_model.text_embedding.register_forward_hook(
            self._feature_hook("T-D1")))
        self._handles.append(wan_model.head.register_forward_hook(
            self._counter_hook()))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def features(self):
        out = {}
        for name, t in self._buf.items():
            if t.dim() >= 2:
                out[name] = t[0].mean(dim=0)
            else:
                out[name] = t
        return out


# ─── Image builders ──────────────────────────────────────────────────────
def build_single(idx):
    return Image.open(f"{DROID_DATA}/images/{idx:06d}.png").convert("RGB")


def build_3cam_A(idx):
    cam1 = Image.open(f"{DROID_DATA}/images/{idx:06d}.png").convert("RGB")
    cam2 = Image.open(f"{DROID_DATA}/images_cam2/{idx:06d}.png").convert("RGB")
    cam3 = Image.open(f"{DROID_DATA}/images_cam3/{idx:06d}.png").convert("RGB")
    top = cam2.resize((640, 480), Image.LANCZOS)
    bl = cam3.resize((320, 240), Image.LANCZOS)
    br = cam1.resize((320, 240), Image.LANCZOS)  # exterior in wrist slot
    canvas = Image.new("RGB", (640, 720))
    canvas.paste(top, (0, 0))
    canvas.paste(bl, (0, 480))
    canvas.paste(br, (320, 480))
    return canvas


def build_3cam_B(idx):
    cam2 = Image.open(f"{DROID_DATA}/images_cam2/{idx:06d}.png").convert("RGB")
    cam3 = Image.open(f"{DROID_DATA}/images_cam3/{idx:06d}.png").convert("RGB")
    top = cam2.resize((640, 480), Image.LANCZOS)
    bl = cam3.resize((320, 240), Image.LANCZOS)
    br = cam3.resize((320, 240), Image.LANCZOS)  # duplicate wrist
    canvas = Image.new("RGB", (640, 720))
    canvas.paste(top, (0, 0))
    canvas.paste(bl, (0, 480))
    canvas.paste(br, (320, 480))
    return canvas


MODES = {
    "single": (build_single, lambda t: t),       # raw DROID prompt
    "3cam_A": (build_3cam_A, lambda t: PROMPT_PREFIX_VIDAR + t),
    "3cam_B": (build_3cam_B, lambda t: PROMPT_PREFIX_VIDAR + t),
}

KEYS = ["V-D1", "V-B0", "V-B7", "V-B14", "V-B21", "V-B29", "V-Norm", "T-D1"]


def extract_mode(wan_ti2v, cap, mode_name, img_fn, prompt_fn, task_descs, cfg):
    print(f"\n{'='*60}")
    print(f"  Mode: {mode_name}  N={N}")
    print(f"{'='*60}")
    accum = {k: [] for k in KEYS}
    t0 = time.time()
    for i in range(N):
        img = img_fn(i)
        prompt = prompt_fn(task_descs[i])
        cap.reset()
        with torch.inference_mode():
            _ = wan_ti2v.i2v(
                input_prompt=prompt, img=img, max_area=81920,
                frame_num=5, shift=cfg.sample_shift, sample_solver="unipc",
                sampling_steps=25, guide_scale=5.0, n_prompt="", seed=i,
                offload_model=False,
            )
        feats = cap.features()
        for k in KEYS:
            accum[k].append(feats[k])
        if (i + 1) % 25 == 0:
            print(f"    [{i+1}/{N}] elapsed={time.time()-t0:.0f}s", flush=True)
    stacked = {k: torch.stack(accum[k]) for k in KEYS}
    print(f"  Done in {time.time()-t0:.1f}s ({N/(time.time()-t0):.2f} samples/s)")
    return stacked


def compute_cosine_sim_stats(feats_dict, label):
    """Compute mean pairwise cosine similarity for each feature."""
    print(f"\n--- {label}: feature stats ---")
    print(f"  {'feat':<8} {'norm_mean':>10} {'norm_std':>10} {'cos_sim':>10} {'var_ratio':>10}")
    for k in KEYS:
        t = feats_dict[k]  # [N, D]
        norms = t.norm(dim=1)
        # Cosine similarity matrix
        t_norm = t / (t.norm(dim=1, keepdim=True) + 1e-8)
        cos_mat = t_norm @ t_norm.T
        # Mask out diagonal
        mask = ~torch.eye(N, dtype=torch.bool)
        cos_mean = cos_mat[mask].mean().item()
        # Variance ratio: sum of eigenvalues of covariance / trace
        centered = t - t.mean(dim=0, keepdim=True)
        cov = centered.T @ centered / (N - 1)
        eigs = torch.linalg.eigvalsh(cov)
        var_ratio = (eigs[-5:].sum() / eigs.sum()).item()  # top-5 components
        print(f"  {k:<8} {norms.mean().item():>10.2f} {norms.std().item():>10.2f} "
              f"{cos_mean:>10.4f} {var_ratio:>10.4f}")


def compute_knn_overlap_proxy(feats_dict, label):
    """Compute a proxy for CKNNA: kNN overlap between feature cosine distance and DTW.

    Uses DTW top-k (k=10) cache, restricted to first N samples.
    Counts how many DTW-nearest-neighbor pairs are also feature-nearest-neighbor pairs.
    """
    print(f"\n--- {label}: kNN overlap with DTW (proxy CKNNA) ---")
    # Load DTW top-k for a few horizons
    dtw_dir = "/lambda/nfs/vla/cache/cknna_data_store/record/dtw_cache/droid"
    horizons = [1, 7, 25, 75, 220]
    k = 10

    for h in horizons:
        dtw_path = os.path.join(dtw_dir, f"dtw_topk_h{h}_k{k}_sym2_cuda_nowin_nopad.npy")
        if not os.path.exists(dtw_path):
            continue
        dtw_topk = np.load(dtw_path)  # [6000, k]
        # Restrict to first N: keep only neighbors that are also in [0, N)
        dtw_sub = dtw_topk[:N]  # [N, k]

        results = {}
        for feat_key in ["V-D1", "V-Norm", "T-D1"]:
            t = feats_dict[feat_key]  # [N, D]
            # Compute feature kNN (cosine distance)
            t_norm = t / (t.norm(dim=1, keepdim=True) + 1e-8)
            cos_sim = t_norm @ t_norm.T  # [N, N]
            cos_sim.fill_diagonal_(-1)  # exclude self
            _, feat_topk = cos_sim.topk(k, dim=1)  # [N, k]
            feat_topk = feat_topk.numpy()

            # Count overlap: for each sample i, how many of its DTW k-NN
            # are also in its feature k-NN (restricted to [0,N))
            overlap = 0
            valid_pairs = 0
            for i in range(N):
                dtw_nn = set(int(j) for j in dtw_sub[i] if 0 <= j < N and j != i)
                feat_nn = set(int(j) for j in feat_topk[i])
                valid_pairs += len(dtw_nn)
                overlap += len(dtw_nn & feat_nn)

            results[feat_key] = (overlap, valid_pairs)

        print(f"  h={h:>3}: ", end="")
        for fk in ["V-D1", "V-Norm", "T-D1"]:
            ov, vp = results[fk]
            rate = ov / max(vp, 1) * 100
            print(f"{fk}={ov}/{vp}({rate:.1f}%)  ", end="")
        print()


def main():
    with open(os.path.join(DROID_DATA, "metadata.json")) as f:
        meta = json.load(f)
    task_descs = meta["task_descriptions"]

    print("Loading WanTI2V (T5 + VAE + WanModel + vidar.pt)...")
    cfg = WAN_CONFIGS["ti2v-5B"]
    t0 = time.time()
    wan_ti2v = wan.WanTI2V(
        config=cfg, checkpoint_dir=WAN22_BASE, pt_dir=VIDAR_PT,
        device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False,
        use_sp=False, t5_cpu=False, convert_model_dtype=True,
    )
    print(f"  loaded in {time.time()-t0:.1f}s")

    cap = VidarHookCapture()
    cap.register(wan_ti2v.model)

    all_feats = {}
    for mode_name, (img_fn, prompt_fn) in MODES.items():
        all_feats[mode_name] = extract_mode(
            wan_ti2v, cap, mode_name, img_fn, prompt_fn, task_descs, cfg
        )
        torch.cuda.empty_cache()
        gc.collect()

    cap.remove()

    # Compare
    for mode_name in MODES:
        compute_cosine_sim_stats(all_feats[mode_name], mode_name)

    for mode_name in MODES:
        compute_knn_overlap_proxy(all_feats[mode_name], mode_name)

    # Direct comparison table
    print(f"\n{'='*60}")
    print(f"  CROSS-MODE COMPARISON (N={N})")
    print(f"{'='*60}")
    print(f"  {'feat':<8} {'single_norm':>12} {'3camA_norm':>12} {'3camB_norm':>12} "
          f"{'single_cos':>12} {'3camA_cos':>12} {'3camB_cos':>12}")
    for k in KEYS:
        norms = {}
        cos_sims = {}
        for mode in MODES:
            t = all_feats[mode][k]
            norms[mode] = t.norm(dim=1).mean().item()
            t_n = t / (t.norm(dim=1, keepdim=True) + 1e-8)
            cs = t_n @ t_n.T
            mask = ~torch.eye(N, dtype=torch.bool)
            cos_sims[mode] = cs[mask].mean().item()
        print(f"  {k:<8} {norms['single']:>12.2f} {norms['3cam_A']:>12.2f} {norms['3cam_B']:>12.2f} "
              f"{cos_sims['single']:>12.4f} {cos_sims['3cam_A']:>12.4f} {cos_sims['3cam_B']:>12.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
