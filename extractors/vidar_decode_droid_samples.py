"""
Side script: generate Vidar mp4 outputs for the FIRST 10 DROID v2 samples
using the same hyperparameters our feature extractor uses.

Purpose: visually verify what Vidar "sees + predicts" for DROID images at
frame_num=5 (= 5 output frames = ~0.21s of video). Outputs mp4 files only,
no feature extraction.

Usage:
    PYTHONUNBUFFERED=1 python -u vidar_decode_droid_samples.py \
        --output_dir /tmp/vidar_droid_videos_n10 \
        --num_samples 10
"""

import argparse
import json
import os
import sys
import warnings
import time

warnings.filterwarnings("ignore")

import torch
from PIL import Image

VIDAR_ROOT = "/lambda/nfs/vla/cknna_project/repos/vidar"
if VIDAR_ROOT not in sys.path:
    sys.path.insert(0, VIDAR_ROOT)

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video

WAN22_BASE = "/lambda/nfs/vla/pretrained_models/Wan2.2-TI2V-5B"
VIDAR_PT = "/lambda/nfs/vla/cknna_project/checkpoints/vidar/vidar.pt"
DROID_DATA = "/lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/tmp/vidar_droid_videos_n10")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--frame_num", type=int, default=5)
    parser.add_argument("--sampling_steps", type=int, default=25)
    parser.add_argument("--max_area", type=int, default=81920)
    parser.add_argument("--guide_scale", type=float, default=5.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(DROID_DATA, "metadata.json")) as f:
        meta = json.load(f)
    task_descriptions = meta["task_descriptions"]
    images_dir = os.path.join(DROID_DATA, "images")

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

    print(f"\nGenerating {args.num_samples} videos at frame_num={args.frame_num}, "
          f"sampling_steps={args.sampling_steps}, guide_scale={args.guide_scale}, "
          f"max_area={args.max_area}")

    for i in range(args.num_samples):
        prompt = task_descriptions[i]
        img_path = os.path.join(images_dir, f"{i:06d}.png")
        img = Image.open(img_path).convert("RGB")
        print(f"\n[{i+1}/{args.num_samples}] {i:06d}.png  prompt='{prompt[:80]}...'", flush=True)
        t0 = time.time()
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
        out_path = os.path.join(args.output_dir, f"droid_{i:06d}.mp4")
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
