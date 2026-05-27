"""
Phase 1 (full): Load Bridge V2 data from local RLDS TFRecord files.

Reads RLDS TFRecord shards from $BRIDGE_RLDS_DIR (set in paths.env;
default $WORK/datasets/openvla/bridge_orig/1.0.0), 124 GB,
1024 train shards, ~53K episodes. No network downloads required.

State:  7D [x, y, z, roll, pitch, yaw, gripper] (already without pad)
Action: 7D [dx, dy, dz, droll, dpitch, dyaw, dgripper]
Image:  256x256 RGB (embedded as JPEG in RLDS)

Output (identical schema to load_bridge_data_full.py):
  <output_dir>/feats_B.pt         -- (N, 7) float32 state at time t
  <output_dir>/actions.pt         -- (N, 7) float32 action at time t
  <output_dir>/feats_B_seq.pt     -- (N, H+1, 7) states [t, t+1, ..., t+H]
  <output_dir>/actions_seq.pt     -- (N, H+1, 7) actions [t, t+1, ..., t+H]
  <output_dir>/images/NNNNNN.png  -- PNG images at time t
  <output_dir>/metadata.json

Usage:
  Run with simpler_env conda env (has TensorFlow 2.15):
    conda activate simpler_env
    CUDA_VISIBLE_DEVICES="" python load_bridge_data_rlds.py \\
        --output_dir cknna_data_exp1 --max_horizon 15 --seed 42

Note: CUDA_VISIBLE_DEVICES="" prevents TF from allocating GPU memory.
"""

import argparse
import glob
import io
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from PIL import Image as PILImage

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import tensorflow as tf

RLDS_DIR = os.environ.get(
    "BRIDGE_RLDS_DIR",
    os.path.expanduser("~/vla/datasets/openvla/bridge_orig/1.0.0"),
)
STATE_DIM = 7
ACTION_DIM = 7
# Use compress_level=1 (fastest) -- trades slightly larger PNG for ~10x faster write
PNG_COMPRESS = 1
# Number of parallel image-write threads
IMAGE_WORKERS = 16


def _write_png(path, img_bytes):
    """Decode JPEG bytes and write PNG to disk (runs in thread pool)."""
    img = PILImage.open(io.BytesIO(img_bytes))
    img.save(path, format="PNG", compress_level=PNG_COMPRESS)


def read_shard(path, rng, max_horizon, skipped_counter, max_length=999999):
    """Read one TFRecord shard and sample one frame per episode."""
    H = max_horizon
    episodes = []
    for raw in tf.data.TFRecordDataset([path]):
        ex = tf.train.Example.FromString(raw.numpy())
        fts = ex.features.feature

        is_first = list(fts["steps/is_first"].int64_list.value)
        n_steps = len(is_first)

        if n_steps <= H:
            skipped_counter[0] += 1
            continue

        if n_steps > max_length:
            skipped_counter[0] += 1
            continue

        fidx = rng.randint(0, n_steps - 1 - H)

        raw_states = list(fts["steps/observation/state"].float_list.value)
        states_all = np.array(raw_states, dtype=np.float32).reshape(n_steps, STATE_DIM)
        states_seq = states_all[fidx:fidx + H + 1]

        raw_actions = list(fts["steps/action"].float_list.value)
        actions_all = np.array(raw_actions, dtype=np.float32).reshape(n_steps, ACTION_DIM)
        actions_seq = actions_all[fidx:fidx + H + 1]

        lang_bytes = fts["steps/language_instruction"].bytes_list.value
        lang = lang_bytes[fidx].decode("utf-8", errors="replace").strip()

        img_bytes = bytes(fts["steps/observation/image_0"].bytes_list.value[fidx])
        ep_id = int(fts["episode_metadata/episode_id"].int64_list.value[0])

        episodes.append({
            "episode_index": ep_id,
            "frame_index": fidx,
            "episode_length": n_steps,
            "state": states_seq[0],
            "action": actions_seq[0],
            "states_seq": states_seq,
            "actions_seq": actions_seq,
            "task": lang,
            "img_bytes": img_bytes,
        })
    return episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_horizon", type=int, default=15)
    parser.add_argument("--max_length", type=int, default=999999,
                        help="Upper bound on episode length (episodes with "
                             "n_steps > max_length are skipped)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rlds_dir", type=str, default=RLDS_DIR)
    args = parser.parse_args()

    H = args.max_horizon
    rng = random.Random(args.seed)

    print(f"=== Phase 1 (RLDS): Bridge V2, max_horizon={H}, seed={args.seed} ===")
    print(f"RLDS dir: {args.rlds_dir}")
    print(f"Output:   {args.output_dir}")

    shards = sorted(glob.glob(
        os.path.join(args.rlds_dir, "bridge_dataset-train.tfrecord-*-of-*")
    ))
    print(f"Train shards found: {len(shards)}")
    assert len(shards) > 0, f"No train TFRecords found in {args.rlds_dir}"

    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    t0 = time.time()
    sampled = []
    skipped = [0]
    img_futures = []

    print(f"\nReading {len(shards)} shards + saving images in parallel "
          f"({IMAGE_WORKERS} writer threads)...")

    with ThreadPoolExecutor(max_workers=IMAGE_WORKERS) as img_pool:
        for i, shard_path in enumerate(shards):
            shard_rng = random.Random(rng.randint(0, 2**31))
            episodes = read_shard(shard_path, shard_rng, H, skipped,
                                  max_length=args.max_length)

            for ep in episodes:
                global_idx = len(sampled)
                img_path = os.path.join(images_dir, f"{global_idx:06d}.png")
                fut = img_pool.submit(_write_png, img_path, ep["img_bytes"])
                img_futures.append(fut)
                ep.pop("img_bytes")   # free memory after submit
                sampled.append(ep)

            if (i + 1) % 64 == 0 or (i + 1) == len(shards):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(shards) - (i + 1)) / rate if rate > 0 else 0
                done_imgs = sum(1 for f in img_futures if f.done())
                print(f"  [{i+1}/{len(shards)} shards]  eps={len(sampled)}"
                      f"  imgs_written={done_imgs}"
                      f"  {rate:.1f} shards/s  ETA {eta:.0f}s")

        print("  Waiting for remaining image writes...")
        for fut in as_completed(img_futures):
            fut.result()   # propagate any write errors

    t1 = time.time()
    print(f"\nAll done in {t1-t0:.0f}s")
    print(f"  Episodes sampled: {len(sampled)}")
    print(f"  Skipped (L <= H): {skipped[0]}")
    print(f"  Images saved: {len(img_futures)}")

    # --- Save tensors ---
    print("\nBuilding tensors...")
    feats_B = torch.tensor(
        np.stack([sf["state"] for sf in sampled]), dtype=torch.float32
    )
    torch.save(feats_B, os.path.join(args.output_dir, "feats_B.pt"))
    print(f"  feats_B:     {tuple(feats_B.shape)}")

    actions = torch.tensor(
        np.stack([sf["action"] for sf in sampled]), dtype=torch.float32
    )
    torch.save(actions, os.path.join(args.output_dir, "actions.pt"))
    print(f"  actions:     {tuple(actions.shape)}")

    feats_B_seq = torch.tensor(
        np.stack([sf["states_seq"] for sf in sampled]), dtype=torch.float32
    )
    torch.save(feats_B_seq, os.path.join(args.output_dir, "feats_B_seq.pt"))
    print(f"  feats_B_seq: {tuple(feats_B_seq.shape)}")

    actions_seq = torch.tensor(
        np.stack([sf["actions_seq"] for sf in sampled]), dtype=torch.float32
    )
    torch.save(actions_seq, os.path.join(args.output_dir, "actions_seq.pt"))
    print(f"  actions_seq: {tuple(actions_seq.shape)}")

    # --- Save metadata ---
    metadata = {
        "num_samples": len(sampled),
        "max_horizon": H,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "source": "rlds",
        "rlds_dir": args.rlds_dir,
        "seed": args.seed,
        "episode_indices": [sf["episode_index"] for sf in sampled],
        "frame_indices": [sf["frame_index"] for sf in sampled],
        "episode_lengths": [sf["episode_length"] for sf in sampled],
        "task_descriptions": [sf["task"] for sf in sampled],
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    total_time = time.time() - t0
    print(f"\n=== Phase 1 (RLDS) complete in {total_time:.0f}s ({total_time/60:.1f} min) ===")
    print(f"  N = {len(sampled)} episodes")

    # Quick sanity check on group sizes
    lengths = metadata["episode_lengths"]
    tasks = metadata["task_descriptions"]
    labeled = sum(1 for t in tasks if t and t.strip())
    S = sum(1 for L, t in zip(lengths, tasks) if 16 <= L <= 25 and t and t.strip())
    M = sum(1 for L, t in zip(lengths, tasks) if 26 <= L <= 40 and t and t.strip())
    Lg = sum(1 for L, t in zip(lengths, tasks) if L > 40 and t and t.strip())
    print(f"\n  Labeled: {labeled}/{len(sampled)}")
    print(f"  Group S (16<=L<=25, labeled): {S}")
    print(f"  Group M (26<=L<=40, labeled): {M}")
    print(f"  Group L (L>40,      labeled): {Lg}")


if __name__ == "__main__":
    main()
