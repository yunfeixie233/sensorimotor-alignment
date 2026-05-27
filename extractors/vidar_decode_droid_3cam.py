"""
Side script: generate Vidar mp4 outputs for DROID samples using a 3-CAMERA
CONCAT input matching Vidar's official ALOHA training distribution layout.

Vidar's example dataset (examples/robotwin_example.json) uses 640x720 PNGs
arranged as:
  - Top 640x480: rear/overhead camera
  - Bottom-left 320x240: left arm wrist camera
  - Bottom-right 320x240: right arm wrist camera

DROID v2 has 3 cameras already cached at /lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid/:
  - images/        cam1: exterior side view
  - images_cam2/   cam2: exterior top/front view
  - images_cam3/   cam3: wrist camera (Franka has only ONE wrist)

Layout modes (for the bottom 2 wrist slots — Vidar expects bimanual ALOHA L+R wrist):
  --mode A   Top=cam2, BL=cam3 wrist, BR=cam1 (other exterior)  ← honest about DROID's hardware
  --mode B   Top=cam2, BL=cam3 wrist, BR=cam3 wrist (DUPLICATED) ← matches Vidar slot-type
                                                                   expectation: 1 ext + 2 wrist

Each DROID image is native 320x180 (16:9). To fit Vidar's 640x720 layout we
resize each pane (with stretching to match Vidar's expected aspect):
  cam2 320x180 -> 640x480  (vertical stretch by 1.33x)
  cam3 320x180 -> 320x240  (vertical stretch by 1.33x)
  cam1 320x180 -> 320x240  (vertical stretch by 1.33x)

Prompt format also matches Vidar's training distribution: prefix that names
the 3 views and then the actual DROID task description.

Usage:
    PYTHONUNBUFFERED=1 python -u vidar_decode_droid_3cam.py \
        --output_dir /lambda/nfs/vla/cache/cknna_data_store/vidar_droid_videos_n10_3cam \
        --num_samples 10
"""

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import torch
from PIL import Image

VIDAR_ROOT = "/lambda/nfs/vla/cknna_project/repos/vidar"
if VIDAR_ROOT not in sys.path:
    sys.path.insert(0, VIDAR_ROOT)

import wan
from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.utils import save_video

WAN22_BASE = "/lambda/nfs/vla/pretrained_models/Wan2.2-TI2V-5B"
VIDAR_PT = "/lambda/nfs/vla/cknna_project/checkpoints/vidar/vidar.pt"
DROID_DATA = "/lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid"

# Vidar-style prompt prefix (adapted from examples/robotwin_example.json prompt
# format, with "aloha" -> "franka" and camera names matching DROID's 3 views).
PROMPT_PREFIX = (
    "The whole scene is in a realistic, industrial art style with three views: "
    "a fixed external camera, a movable wrist camera, and a side external camera. "
    "The franka robot is currently performing the following task: "
)


def build_3cam_concat(idx, mode="A"):
    """Build a 640x720 ALOHA-layout concat from DROID's 3 camera views.

    Mode A (honest): Top=cam2 ext, BL=cam3 wrist, BR=cam1 ext
      → bottom-right slot has WRONG type vs Vidar's expected wrist
    Mode B (slot-matched): Top=cam2 ext, BL=cam3 wrist, BR=cam3 wrist (duplicated)
      → both bottom slots are wrist views, matching Vidar's "1 rear + 2 wrist"
        slot-type expectation. Equivalent to telling Vidar "this scene has only
        one robot arm" since both wrist views are identical.

    Layout (matches Vidar's examples/images/robotwin_aloha_3_adjust_bottle_*.png):
      +---------------------+
      |                     |
      |  cam2 (640x480)     |   <- top half: fixed external
      |                     |
      +----------+----------+
      |   cam3   |  cam1/3  |   <- bottom: 2 wrist slots
      | 320x240  | 320x240  |
      +----------+----------+
    """
    cam1 = Image.open(f"{DROID_DATA}/images/{idx:06d}.png").convert("RGB")
    cam2 = Image.open(f"{DROID_DATA}/images_cam2/{idx:06d}.png").convert("RGB")
    cam3 = Image.open(f"{DROID_DATA}/images_cam3/{idx:06d}.png").convert("RGB")

    # Resize each pane to its target dimensions (Lanczos for quality).
    top = cam2.resize((640, 480), Image.LANCZOS)
    bl = cam3.resize((320, 240), Image.LANCZOS)
    if mode == "A":
        br = cam1.resize((320, 240), Image.LANCZOS)
    elif mode == "B":
        br = cam3.resize((320, 240), Image.LANCZOS)  # duplicate wrist
    else:
        raise ValueError(f"Unknown mode: {mode!r}; expected 'A' or 'B'")

    canvas = Image.new("RGB", (640, 720))
    canvas.paste(top, (0, 0))
    canvas.paste(bl, (0, 480))
    canvas.paste(br, (320, 480))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/lambda/nfs/vla/cache/cknna_data_store/vidar_droid_videos_n10_3cam")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--frame_num", type=int, default=121)
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--size", default="640*736",
                        help="Vidar SIZE_CONFIGS key for max_area lookup")
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--save_concat_inputs", action="store_true",
                        help="Also save the 640x720 3-cam concat PNGs alongside the mp4s")
    parser.add_argument("--mode", default="A", choices=["A", "B"],
                        help="3-cam layout mode: A=cam2/cam3/cam1 (honest), "
                             "B=cam2/cam3/cam3 (Vidar slot-type matched, wrist duplicated)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_concat_inputs:
        os.makedirs(os.path.join(args.output_dir, "concat_inputs"), exist_ok=True)

    with open(os.path.join(DROID_DATA, "metadata.json")) as f:
        meta = json.load(f)
    task_descriptions = meta["task_descriptions"]

    max_area = MAX_AREA_CONFIGS[args.size]
    print(f"Loading WanTI2V (T5 + VAE + WanModel base + vidar.pt overlay)...")
    cfg = WAN_CONFIGS["ti2v-5B"]
    t0 = time.time()
    wan_ti2v = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=WAN22_BASE,
        pt_dir=VIDAR_PT,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=True,
    )
    print(f"  loaded in {time.time()-t0:.1f}s")

    print(f"\nGenerating {args.num_samples} videos at "
          f"frame_num={args.frame_num}, sampling_steps={args.sampling_steps}, "
          f"size={args.size} (max_area={max_area}), guide_scale={args.guide_scale}")
    if args.mode == "A":
        print(f"Input format: 640x720 3-cam concat (mode A: cam2 top, cam3+cam1 bottom)")
    else:
        print(f"Input format: 640x720 3-cam concat (mode B: cam2 top, cam3+cam3 wrist DUPLICATED bottom)")
    print(f"Prompt prefix: '{PROMPT_PREFIX[:80]}...'")

    for i in range(args.num_samples):
        droid_task = task_descriptions[i]
        full_prompt = PROMPT_PREFIX + droid_task
        img = build_3cam_concat(i, mode=args.mode)
        if args.save_concat_inputs:
            img.save(os.path.join(args.output_dir, "concat_inputs", f"droid_{i:06d}.png"))
        print(f"\n[{i+1}/{args.num_samples}] DROID #{i:06d}  "
              f"input=640x720(3cam)  task='{droid_task[:60]}...'", flush=True)
        t0 = time.time()
        with torch.inference_mode():
            video = wan_ti2v.i2v(
                input_prompt=full_prompt,
                img=img,
                max_area=max_area,
                frame_num=args.frame_num,
                shift=cfg.sample_shift,
                sample_solver="unipc",
                sampling_steps=args.sampling_steps,
                guide_scale=args.guide_scale,
                n_prompt="",
                seed=i,
                offload_model=False,
            )
        out_path = os.path.join(args.output_dir, f"droid_3cam_{i:06d}.mp4")
        save_video(
            tensor=video[None],
            save_file=out_path,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        print(f"  saved {out_path} in {time.time()-t0:.1f}s")

    print(f"\nDone. {args.num_samples} videos saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
