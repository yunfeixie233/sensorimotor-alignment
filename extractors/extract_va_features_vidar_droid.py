"""
Vidar feature extraction on DROID v2 (single-camera, 6000 episodes).

Wraps the official `wan.WanTI2V` pipeline (matches generate.py:351-362) and
calls `WanTI2V.i2v(prompt, img, sampling_steps=25, frame_num=5, max_area=81920)`
on each DROID image. Hooks attached to WanModel internals BEFORE i2v() and
removed AFTER. Hooks are read-only forward/forward_pre hooks that do not
alter the WanModel forward path.

Native forward path (verified 2026-04-11 via running official generate.py
with examples/robotwin_example.json — 3 mp4 outputs at 640x736 produced):
  WanTI2V.i2v():
    1. resize image via best_output_size to multiples of 32x32
    2. T5 encode prompt + negative prompt
    3. VAE encode image to latent
    4. mask first latent frame with image (i2v conditioning trick)
    5. UniPC sampling loop:
       for each step:
         model(latent, t, context=cond_context, ...)    <- our hooks fire (cond pass)
         model(latent, t, context=null_context, ...)    <- our hooks fire (uncond pass, FILTERED OUT)
         scheduler.step()
    6. VAE decode (we SKIP this in extractor, not needed)

Compute config (matches LingBot-VA va_droid_1cam_i2va_cfg for cross-comparability):
  frame_num=5             (= 2 latent frames, matches frame_chunk_size=2)
  sampling_steps=25       (matches num_inference_steps=25)
  max_area=81920          (= 256x320, matches LingBot-VA height=256 width=320)
  guide_scale=5.0         (CFG enabled, matches guidance_scale=5)
  sample_solver='unipc'   (Vidar default)
  shift=5.0               (Vidar ti2v_5B default)

Estimated compute: ~2.5 s/sample × 6000 = ~4.2 hours.

Hook positions (all on `wan_ti2v.model.*`):
  V-D1     blocks[0]  forward_pre_hook  -> post-flatten input  [1, L, 3072]
  V-B0     blocks[0]  forward_hook
  V-B7     blocks[7]  forward_hook
  V-B14    blocks[14] forward_hook
  V-B21    blocks[21] forward_hook
  V-B29    blocks[29] forward_hook
  V-Norm   head.norm  forward_hook      -> bare RMSNorm (pre-modulation)
  T-D1     text_embedding  forward_hook -> T5 -> 3072d projection

CFG handling: model() is called twice per denoising step (cond first, uncond
second, see textimage2video.py:580-585). We use a parity counter on the
outermost head module to know which pass is current. Feature hooks capture
ONLY on conditional passes (call_idx % 2 == 0) and overwrite the buffer
each time. After i2v() returns, the buffer holds the LAST denoising step's
conditional-pass features.

Output: 7 .pt files matching LingBot-VA per-feature naming convention:
  feats_A_vit.pt        V-D1     [N, 3072]
  feats_A_block0.pt     V-B0
  feats_A_block7.pt     V-B7
  feats_A_block14.pt    V-B14
  feats_A_block21.pt    V-B21
  feats_A_block29.pt    V-B29
  feats_A_norm.pt       V-Norm
  feats_A_text.pt       T-D1
"""

import argparse
import gc
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import torch
from PIL import Image

# ----------------------------------------------------------------------
# Path setup: vidar repo for `wan` module + the local checkpoint dirs.
# ----------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIDAR_ROOT = os.environ.get(
    "VIDAR_ROOT", os.path.join(_PROJECT_ROOT, "repos", "vidar")
)
if VIDAR_ROOT not in sys.path:
    sys.path.insert(0, VIDAR_ROOT)

import wan  # noqa: E402
from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS  # noqa: E402
from wan.utils.utils import save_video  # noqa: E402

# Checkpoint paths read from environment with a fallback to the dev-machine
# layout. Set CKPT_WAN22_TI2V_5B + CKPT_VIDAR in paths.env on a fresh clone.
WAN22_BASE = os.environ.get("CKPT_WAN22_TI2V_5B", "/lambda/nfs/vla/pretrained_models/Wan2.2-TI2V-5B")
VIDAR_PT = os.environ.get("CKPT_VIDAR", "/lambda/nfs/vla/cknna_project/checkpoints/vidar/vidar.pt")
_DATA_STORE = os.environ.get("DATA_STORE", "/lambda/nfs/vla/cache/cknna_data_store")
DROID_DATA = os.path.join(_DATA_STORE, "cknna_data_droid")

# Vidar-style ALOHA prompt prefix (matches examples/robotwin_example.json format,
# adapted for DROID's Franka single-arm: "fixed external + movable wrist + side
# external" rather than "rear + L_arm + R_arm" since Franka has only 1 wrist).
PROMPT_PREFIX_VIDAR = (
    "The whole scene is in a realistic, industrial art style with three views: "
    "a fixed external camera, a movable wrist camera, and a side external camera. "
    "The franka robot is currently performing the following task: "
)


def build_3cam_concat_aloha_native(idx, data_dir):
    """Native bimanual ALOHA layout from cknna_data_aloha.

    Reads:
       images/{idx}.png       = observation.images.cam_high       (real head)
       images_cam2/{idx}.png  = observation.images.cam_left_wrist (real left wrist)
       images_cam3/{idx}.png  = observation.images.cam_right_wrist(real right wrist)

    Builds Vidar's expected 640x720 layout: cam_high (640x480) on top + half-size
    left/right wrists (320x240 each) in the bottom row. Exactly matches Vidar's
    examples/images/robotwin_aloha_3_*.png — no camera duplication, both wrist
    slots are real wrists. This is the layout Vidar was trained on.
    """
    head = Image.open(f"{data_dir}/images/{idx:06d}.png").convert("RGB")
    left = Image.open(f"{data_dir}/images_cam2/{idx:06d}.png").convert("RGB")
    right = Image.open(f"{data_dir}/images_cam3/{idx:06d}.png").convert("RGB")
    top = head.resize((640, 480), Image.LANCZOS)
    bl = left.resize((320, 240), Image.LANCZOS)
    br = right.resize((320, 240), Image.LANCZOS)
    canvas = Image.new("RGB", (640, 720))
    canvas.paste(top, (0, 0))
    canvas.paste(bl, (0, 480))
    canvas.paste(br, (320, 480))
    return canvas


def build_3cam_concat_modeB(idx):
    """Build a 640x720 ALOHA-layout concat from DROID's 3 camera views in mode B.

    Mode B duplicates the wrist view (cam3) in both bottom slots to match Vidar's
    expected slot-type layout (1 rear + 2 wrist). Since DROID's Franka has only
    one wrist camera, both bottom slots are identical wrist views — equivalent
    to telling Vidar "this scene has only one robot arm".

    Layout (matches Vidar's examples/images/robotwin_aloha_3_adjust_bottle_*.png):
      +---------------------+
      |                     |
      |  cam2 (640x480)     |   <- top: fixed external
      |                     |
      +----------+----------+
      |   cam3   |   cam3   |   <- bottom: wrist DUPLICATED
      | 320x240  | 320x240  |
      +----------+----------+
    """
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


# ----------------------------------------------------------------------
# Hook capture: parity counter filters CFG conditional-only passes.
# ----------------------------------------------------------------------
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
        self._handles.append(
            wan_model.blocks[0].register_forward_pre_hook(
                self._feature_hook("V-D1", kind="pre")
            )
        )
        for bi in self.VIDEO_BLOCKS:
            self._handles.append(
                wan_model.blocks[bi].register_forward_hook(
                    self._feature_hook(f"V-B{bi}")
                )
            )
        self._handles.append(
            wan_model.head.norm.register_forward_hook(
                self._feature_hook("V-Norm")
            )
        )
        self._handles.append(
            wan_model.text_embedding.register_forward_hook(
                self._feature_hook("T-D1")
            )
        )
        self._handles.append(
            wan_model.head.register_forward_hook(self._counter_hook())
        )

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def features(self):
        out = {}
        for name, t in self._buf.items():
            if t.dim() >= 2:
                out[name] = t[0].mean(dim=0)  # [1, L, 3072] -> [3072]
            else:
                out[name] = t
        return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Vidar 7-feat extraction on DROID")
    parser.add_argument(
        "--data_dir",
        default="/lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid",
    )
    parser.add_argument(
        "--output_dir",
        default="/lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid_7feat/vidar",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--frame_num", type=int, default=5)
    parser.add_argument("--sampling_steps", type=int, default=25)
    parser.add_argument("--max_area", type=int, default=81920)
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument(
        "--input_mode", default="single", choices=["single", "3cam_B", "3cam_aloha_native"],
        help="single = 1-cam DROID PNG (cam1); "
             "3cam_B = DROID 640x720 3-cam concat with cam2 top + cam3 wrist duplicated "
             "in both bottom slots; "
             "3cam_aloha_native = ALOHA-native 640x720 3-cam concat from cknna_data_aloha "
             "with images/=cam_high (top), images_cam2/=cam_left_wrist (BL), images_cam3/=cam_right_wrist (BR).",
    )
    parser.add_argument(
        "--prompt_style", default="droid_raw",
        choices=["droid_raw", "vidar_aloha_prefix"],
        help="droid_raw = use DROID task description as-is; "
             "vidar_aloha_prefix = prepend Vidar's ALOHA-style 3-view prompt prefix",
    )
    parser.add_argument(
        "--save_videos_first_n", type=int, default=0,
        help="DEBUG: also save the first N decoded mp4s alongside features "
             "(slows down by ~1-2s/sample for the first N samples)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_videos_first_n > 0:
        videos_debug_dir = os.path.join(args.output_dir, "debug_videos")
        os.makedirs(videos_debug_dir, exist_ok=True)
        if args.input_mode == "3cam_B":
            os.makedirs(os.path.join(videos_debug_dir, "concat_inputs"), exist_ok=True)

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_total = metadata["num_samples"]
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    end_idx = args.end_idx if args.end_idx is not None else num_total
    if args.max_samples is not None:
        end_idx = min(end_idx, args.start_idx + args.max_samples)

    print(f"=== Vidar feature extraction on DROID v2 ===")
    print(f"  Samples: [{args.start_idx}, {end_idx})  total={end_idx - args.start_idx}")
    print(f"  Output: {args.output_dir}")
    print(f"  Input mode: {args.input_mode}  Prompt style: {args.prompt_style}")
    print(f"  frame_num={args.frame_num}  sampling_steps={args.sampling_steps}")
    print(f"  max_area={args.max_area}  guide_scale={args.guide_scale}")
    if args.save_videos_first_n > 0:
        print(f"  DEBUG: saving first {args.save_videos_first_n} mp4s to {videos_debug_dir}/")

    print("\nLoading WanTI2V (T5 + VAE + WanModel base + vidar.pt overlay)...")
    t_load = time.time()
    cfg = WAN_CONFIGS["ti2v-5B"]
    wan_ti2v = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=WAN22_BASE,
        pt_dir=VIDAR_PT,
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=True,
    )
    print(f"  loaded in {time.time()-t_load:.1f}s")

    print("\nRegistering hooks on wan_ti2v.model...")
    cap = VidarHookCapture()
    cap.register(wan_ti2v.model)

    keys = ["V-D1", "V-B0", "V-B7", "V-B14", "V-B21", "V-B29", "V-Norm", "T-D1"]
    accum = {k: [] for k in keys}

    print("\nExtraction starting...")
    t0 = time.time()
    for i in range(args.start_idx, end_idx):
        droid_task = task_descriptions[i]

        # Build prompt
        if args.prompt_style == "vidar_aloha_prefix":
            prompt = PROMPT_PREFIX_VIDAR + droid_task
        else:
            prompt = droid_task

        # Build input image
        if args.input_mode == "3cam_B":
            img = build_3cam_concat_modeB(i)
        elif args.input_mode == "3cam_aloha_native":
            img = build_3cam_concat_aloha_native(i, args.data_dir)
        else:
            img = Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")

        # Optionally save concat input for debug
        save_video_this_sample = (
            args.save_videos_first_n > 0
            and (i - args.start_idx) < args.save_videos_first_n
        )
        if save_video_this_sample and args.input_mode == "3cam_B":
            img.save(os.path.join(videos_debug_dir, "concat_inputs", f"droid_{i:06d}.png"))

        cap.reset()
        with torch.inference_mode():
            video = wan_ti2v.i2v(
                input_prompt=prompt,
                img=img,
                max_area=args.max_area,
                frame_num=args.frame_num,
                shift=cfg.sample_shift,
                sample_solver="unipc",
                sampling_steps=args.sampling_steps,
                guide_scale=args.guide_scale,
                n_prompt="",
                seed=i,
                offload_model=False,
            )
        feats = cap.features()

        # Optionally save the decoded mp4 for debug
        if save_video_this_sample:
            out_mp4 = os.path.join(videos_debug_dir, f"droid_{i:06d}.mp4")
            save_video(
                tensor=video[None],
                save_file=out_mp4,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )

        for k in keys:
            if k not in feats:
                raise RuntimeError(
                    f"Feature {k} missing at sample {i}. Got: {sorted(feats.keys())}"
                )
            accum[k].append(feats[k])

        if (i + 1 - args.start_idx) % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        if (i + 1 - args.start_idx) % 10 == 0 or i == args.start_idx:
            elapsed = time.time() - t0
            done = i + 1 - args.start_idx
            rate = done / elapsed
            eta = (end_idx - args.start_idx - done) / rate if rate > 0 else 0
            shapes = "  ".join(f"{k}={tuple(feats[k].shape)}" for k in keys[:3])
            print(
                f"  [{done}/{end_idx - args.start_idx}]  {shapes}  rate={rate:.2f}/s  ETA={eta/60:.1f}min",
                flush=True,
            )

    cap.remove()

    file_map = {
        "V-D1": "feats_A_vit.pt",
        "V-B0": "feats_A_block0.pt",
        "V-B7": "feats_A_block7.pt",
        "V-B14": "feats_A_block14.pt",
        "V-B21": "feats_A_block21.pt",
        "V-B29": "feats_A_block29.pt",
        "V-Norm": "feats_A_norm.pt",
        "T-D1": "feats_A_text.pt",
    }
    print(f"\nSaving {len(keys)} feature tensors:")
    for k in keys:
        tensor = torch.stack(accum[k])
        out_path = os.path.join(args.output_dir, file_map[k])
        torch.save(tensor, out_path)
        print(f"  {file_map[k]:<24} shape={tuple(tensor.shape)}")

    meta = {
        "model": "vidar (Wan2.2-TI2V-5B + vidar.pt overlay)",
        "family": "Vidar (Wan2.2 video diffusion finetune)",
        "type": "8_features_video_text",
        "start_idx": args.start_idx,
        "end_idx": end_idx,
        "num_samples": end_idx - args.start_idx,
        "frame_num": args.frame_num,
        "sampling_steps": args.sampling_steps,
        "max_area": args.max_area,
        "guide_scale": args.guide_scale,
        "sample_solver": "unipc",
        "shift": float(cfg.sample_shift),
        "input_mode": args.input_mode,
        "prompt_style": args.prompt_style,
        "captured_step": "last conditional pass (parity counter on head)",
        "feature_dims": {k: int(accum[k][0].shape[-1]) for k in keys},
        "file_map": file_map,
    }
    with open(os.path.join(args.output_dir, "extraction_metadata_vidar.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(
        f"\nDone. {elapsed/60:.1f} min total, {(end_idx - args.start_idx)/elapsed:.2f} samples/s"
    )


if __name__ == "__main__":
    main()
