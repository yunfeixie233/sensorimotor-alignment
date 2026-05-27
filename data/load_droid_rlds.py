"""
Phase 1: Load DROID v1.0.1 data from local RLDS TFRecord files.

Reads RLDS TFRecord shards from $DROID_RLDS_DIR (set in paths.env;
default $WORK/datasets/droid/1.0.1), 2048 train shards, ~76-94K
episodes, Franka Panda.

State (primary):  7D [x, y, z, roll, pitch, yaw, gripper]
                  (observation/cartesian_position + observation/gripper_position)
State (secondary): 8D [joint_0..joint_6, gripper]
                  (observation/joint_position + observation/gripper_position)
Action: 7D [cartesian_target(6), gripper_target(1)]
        (steps/action = action_dict/cartesian_position + action_dict/gripper_position)
        NOTE: this is the commanded end-effector target pose, NOT joint velocities.
Image:  180x320 RGB JPEG from exterior_image_1_left (downsampled from 1280x720 native)

--- min_length design rationale ---
With max_horizon=H, valid fidx (anchor frame) range is [0, L-1-H].
Setting min_length = 2*(H+1) = 441 guarantees every eligible episode has at least
H+1 = 221 valid anchor positions, which ensures fidx can reach the middle of the
episode for even the shortest eligible episode.

Empirical distribution of fidx position (50-shard sample, min_length=441):
  First third  (0-33% of episode): 54.6%
  Middle third (33-66%):           43.2%
  Last third   (66-100%):           2.2%  (structurally rare, requires L >= 737)

Compare with min_length=222:
  First third: 84.3%, Middle: 14.9%, Last: 0.9%  -- strongly biased to early frames.

Estimated pool at min_length=441: ~9,300 eligible labeled episodes across 2048 shards.
Set max_samples <= 6,000 for a comfortable buffer (65% utilization).

Output:
  <output_dir>/feats_B.pt            -- (N, 7) float32 cartesian state at time t
  <output_dir>/feats_B_seq.pt        -- (N, H+1, 7) cartesian states [t, t+1, ..., t+H]
  <output_dir>/feats_B_joint.pt      -- (N, 8) float32 joint state at time t
  <output_dir>/feats_B_seq_joint.pt  -- (N, H+1, 8) joint states [t, t+1, ..., t+H]
  <output_dir>/actions.pt            -- (N, 7) float32 action at time t
  <output_dir>/actions_seq.pt        -- (N, H+1, 7) actions [t, t+1, ..., t+H]
  <output_dir>/images/NNNNNN.png     -- PNG images at time t (one per episode)
  <output_dir>/metadata.json         -- includes shard_paths, shard_record_indices,
                                        file_paths for full TFRecord traceability

Usage:
  conda activate simpler_env
  CUDA_VISIBLE_DEVICES="" python data/load_droid_rlds.py \\
      --output_dir /path/to/cknna_data_droid \\
      --max_horizon 220 --min_length 441 --max_samples 6000 --seed 42
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
    "DROID_RLDS_DIR",
    os.path.expanduser("~/vla/datasets/droid/1.0.1"),
)
SHARD_GLOB = "droid_101-train.tfrecord-*-of-02048"
CART_DIM = 6
JOINT_DIM = 7
GRIP_DIM = 1
STATE_DIM = CART_DIM + GRIP_DIM       # 7D primary (cartesian + gripper)
STATE_DIM_JOINT = JOINT_DIM + GRIP_DIM  # 8D secondary (joint + gripper)
ACTION_DIM = 7                         # cartesian_target(6) + gripper_target(1)
PNG_COMPRESS = 1
IMAGE_WORKERS = 16


def _write_png(path, img_bytes):
    """Decode JPEG bytes and write PNG to disk (runs in thread pool).

    Each call receives a unique path -- no concurrent writes to the same file.
    img_bytes is a plain bytes object (immutable), safe to pass across threads.
    """
    img = PILImage.open(io.BytesIO(img_bytes))
    img.save(path, format="PNG", compress_level=PNG_COMPRESS)


def read_shard(path, rng, max_horizon, min_length, skipped_counter):
    """Read one TFRecord shard and sample one frame per eligible episode.

    Returns a list of episode dicts. Each dict includes traceability fields:
      shard_path:         absolute path to this TFRecord file
      shard_record_idx:   0-based position of this episode record in the shard
                          (counts ALL records, including filtered-out ones)
      file_path:          episode_metadata/file_path (GCS path to source HDF5)
    """
    H = max_horizon
    episodes = []
    record_idx = 0
    for raw in tf.data.TFRecordDataset([path]):
        ex = tf.train.Example.FromString(raw.numpy())
        fts = ex.features.feature
        current_record_idx = record_idx
        record_idx += 1

        is_first = list(fts["steps/is_first"].int64_list.value)
        n_steps = len(is_first)

        # Single unified length guard: episode must be long enough for both
        # the minimum sampling window AND the minimum fidx diversity requirement.
        # min_length should always be >= H+2 (enforced in main()).
        if n_steps < min_length:
            skipped_counter[0] += 1
            continue

        # Skip unlabeled episodes (no language annotation at all)
        lang_bytes = fts["steps/language_instruction"].bytes_list.value
        if not lang_bytes:
            skipped_counter[0] += 1
            continue

        # Sample anchor frame from the valid range [0, L-1-H].
        # With min_length >= 2*(H+1), this range spans at least H+1 positions,
        # ensuring fidx can reach the middle of even the shortest eligible episode.
        fidx = rng.randint(0, n_steps - 1 - H)

        lang = lang_bytes[fidx].decode("utf-8", errors="replace").strip()
        if not lang:
            skipped_counter[0] += 1
            continue

        # --- State: cartesian(6) + gripper(1) = 7D primary ---
        cart = np.array(
            fts["steps/observation/cartesian_position"].float_list.value,
            dtype=np.float32,
        ).reshape(n_steps, CART_DIM)
        grip = np.array(
            fts["steps/observation/gripper_position"].float_list.value,
            dtype=np.float32,
        ).reshape(n_steps, GRIP_DIM)
        state_7d = np.concatenate([cart, grip], axis=1)  # (n_steps, 7)

        # --- State: joint(7) + gripper(1) = 8D secondary ---
        joint = np.array(
            fts["steps/observation/joint_position"].float_list.value,
            dtype=np.float32,
        ).reshape(n_steps, JOINT_DIM)
        state_8d = np.concatenate([joint, grip], axis=1)  # (n_steps, 8)

        states_7d_seq = state_7d[fidx : fidx + H + 1]
        states_8d_seq = state_8d[fidx : fidx + H + 1]

        # --- Action: steps/action = cartesian_target(6) + gripper_target(1) ---
        # steps/action stores the commanded end-effector target pose (6D) and
        # target gripper position (1D). Verified equal to action_dict/cartesian_position
        # concat action_dict/gripper_position. NOT joint velocities.
        actions_all = np.array(
            fts["steps/action"].float_list.value, dtype=np.float32
        ).reshape(n_steps, ACTION_DIM)
        actions_seq = actions_all[fidx : fidx + H + 1]

        # --- Image: exterior_image_1_left (single frame at fidx) ---
        img_bytes = bytes(
            fts["steps/observation/exterior_image_1_left"].bytes_list.value[fidx]
        )

        # --- Traceability: stable identifier back to the source TFRecord ---
        file_path = fts["episode_metadata/file_path"].bytes_list.value[0].decode(
            "utf-8", errors="replace"
        )

        episodes.append(
            {
                # Traceability fields
                "shard_path": path,
                "shard_record_idx": current_record_idx,
                "file_path": file_path,
                # Sampling fields
                "frame_index": fidx,
                "episode_length": n_steps,
                # Data fields
                "state": states_7d_seq[0],
                "action": actions_seq[0],
                "states_seq": states_7d_seq,
                "actions_seq": actions_seq,
                "state_joint": states_8d_seq[0],
                "states_seq_joint": states_8d_seq,
                "task": lang,
                "img_bytes": img_bytes,
            }
        )
    return episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_horizon", type=int, default=220)
    parser.add_argument(
        "--min_length",
        type=int,
        default=441,
        help=(
            "Minimum episode length. Must be >= max_horizon+2. "
            "Default 441 = 2*(max_horizon+1) ensures fidx can span at least "
            "half the episode even for the shortest eligible episodes, giving "
            "diverse observation frame positions (first/middle/late trajectory). "
            "See docstring for empirical distribution stats."
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=6000,
        help=(
            "Maximum number of samples to collect. "
            "With min_length=441 the full dataset has ~9,300 eligible episodes; "
            "6,000 uses 65%% of the pool with a comfortable buffer."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rlds_dir", type=str, default=RLDS_DIR)
    args = parser.parse_args()

    H = args.max_horizon

    if args.min_length < H + 2:
        raise ValueError(
            "--min_length (%d) must be >= max_horizon+2 (%d). "
            "With min_length <= H+1 the valid fidx range collapses to a single "
            "frame (always the first step), defeating the purpose of sampling."
            % (args.min_length, H + 2)
        )

    rng = random.Random(args.seed)

    print(
        f"=== Phase 1 (RLDS): DROID v1.0.1, max_horizon={H}, "
        f"min_length={args.min_length}, max_samples={args.max_samples}, "
        f"seed={args.seed} ==="
    )
    print(f"  fidx diversity: each episode's anchor spans [0, L-1-{H}]")
    print(f"  min fidx range at min_length={args.min_length}: {args.min_length - 1 - H} positions")
    print(f"RLDS dir: {args.rlds_dir}")
    print(f"Output:   {args.output_dir}")

    shards = sorted(glob.glob(os.path.join(args.rlds_dir, SHARD_GLOB)))
    print(f"Train shards found: {len(shards)}")
    assert len(shards) > 0, f"No train TFRecords found in {args.rlds_dir}"

    # Shuffle shards so --max_samples collects a representative cross-section
    # rather than the first N alphabetical shards.
    shard_rng = random.Random(args.seed)
    shard_rng.shuffle(shards)

    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    t0 = time.time()
    sampled = []
    skipped = [0]
    img_futures = []

    print(
        f"\nReading shards + saving images in parallel "
        f"({IMAGE_WORKERS} writer threads)..."
    )

    with ThreadPoolExecutor(max_workers=IMAGE_WORKERS) as img_pool:
        for i, shard_path in enumerate(shards):
            if len(sampled) >= args.max_samples:
                print(f"\n  Reached max_samples={args.max_samples}, stopping early.")
                break

            shard_rng_local = random.Random(rng.randint(0, 2**31))
            episodes = read_shard(
                shard_path, shard_rng_local, H, args.min_length, skipped
            )

            for ep in episodes:
                if len(sampled) >= args.max_samples:
                    break
                global_idx = len(sampled)
                ep["episode_index"] = global_idx
                img_path = os.path.join(images_dir, f"{global_idx:06d}.png")
                # Each submission writes to a unique path -- no write conflicts.
                fut = img_pool.submit(_write_png, img_path, ep["img_bytes"])
                img_futures.append(fut)
                ep.pop("img_bytes")
                sampled.append(ep)

            if (i + 1) % 64 == 0 or (i + 1) == len(shards):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(shards) - (i + 1)) / rate if rate > 0 else 0
                done_imgs = sum(1 for f in img_futures if f.done())
                print(
                    f"  [{i+1}/{len(shards)} shards]  eps={len(sampled)}"
                    f"  imgs_written={done_imgs}"
                    f"  {rate:.1f} shards/s  ETA {eta:.0f}s"
                )

        print("  Waiting for remaining image writes...")
        for fut in as_completed(img_futures):
            fut.result()

    t1 = time.time()
    print(f"\nAll done in {t1-t0:.0f}s")
    print(f"  Episodes sampled: {len(sampled)}")
    print(f"  Skipped (short/unlabeled): {skipped[0]}")
    print(f"  Images saved: {len(img_futures)}")

    # --- Save tensors (7D cartesian primary) ---
    print("\nBuilding tensors...")
    feats_B = torch.tensor(
        np.stack([sf["state"] for sf in sampled]), dtype=torch.float32
    )
    torch.save(feats_B, os.path.join(args.output_dir, "feats_B.pt"))
    print(f"  feats_B:            {tuple(feats_B.shape)}")

    feats_B_seq = torch.tensor(
        np.stack([sf["states_seq"] for sf in sampled]), dtype=torch.float32
    )
    torch.save(feats_B_seq, os.path.join(args.output_dir, "feats_B_seq.pt"))
    print(f"  feats_B_seq:        {tuple(feats_B_seq.shape)}")

    # --- Save tensors (8D joint secondary) ---
    feats_B_joint = torch.tensor(
        np.stack([sf["state_joint"] for sf in sampled]), dtype=torch.float32
    )
    torch.save(feats_B_joint, os.path.join(args.output_dir, "feats_B_joint.pt"))
    print(f"  feats_B_joint:      {tuple(feats_B_joint.shape)}")

    feats_B_seq_joint = torch.tensor(
        np.stack([sf["states_seq_joint"] for sf in sampled]), dtype=torch.float32
    )
    torch.save(
        feats_B_seq_joint, os.path.join(args.output_dir, "feats_B_seq_joint.pt")
    )
    print(f"  feats_B_seq_joint:  {tuple(feats_B_seq_joint.shape)}")

    # --- Save actions ---
    actions = torch.tensor(
        np.stack([sf["action"] for sf in sampled]), dtype=torch.float32
    )
    torch.save(actions, os.path.join(args.output_dir, "actions.pt"))
    print(f"  actions:            {tuple(actions.shape)}")

    actions_seq = torch.tensor(
        np.stack([sf["actions_seq"] for sf in sampled]), dtype=torch.float32
    )
    torch.save(actions_seq, os.path.join(args.output_dir, "actions_seq.pt"))
    print(f"  actions_seq:        {tuple(actions_seq.shape)}")

    # --- Save metadata (with full traceability) ---
    # To re-find episode i in its TFRecord:
    #   shard = metadata["shard_paths"][i]
    #   rec_n = metadata["shard_record_indices"][i]  (0-based record in that shard)
    #   or directly: metadata["file_paths"][i]  (GCS path to source HDF5)
    metadata = {
        "num_samples": len(sampled),
        "max_horizon": H,
        "min_length": args.min_length,
        "max_samples": args.max_samples,
        "state_dim": STATE_DIM,
        "state_dim_joint": STATE_DIM_JOINT,
        "action_dim": ACTION_DIM,
        "source": "droid_rlds",
        "rlds_dir": args.rlds_dir,
        "seed": args.seed,
        # Traceability: locate each episode in the original TFRecord
        "shard_paths": [sf["shard_path"] for sf in sampled],
        "shard_record_indices": [sf["shard_record_idx"] for sf in sampled],
        "file_paths": [sf["file_path"] for sf in sampled],
        # Sampling info
        "episode_indices": [sf["episode_index"] for sf in sampled],
        "frame_indices": [sf["frame_index"] for sf in sampled],
        "episode_lengths": [sf["episode_length"] for sf in sampled],
        "task_descriptions": [sf["task"] for sf in sampled],
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    total_time = time.time() - t0
    print(
        f"\n=== Phase 1 (RLDS) complete in {total_time:.0f}s "
        f"({total_time/60:.1f} min) ==="
    )
    print(f"  N = {len(sampled)} episodes")

    # Sanity checks
    lengths = metadata["episode_lengths"]
    tasks = metadata["task_descriptions"]
    fidxs = metadata["frame_indices"]
    labeled = sum(1 for t in tasks if t and t.strip())
    unique_tasks = len(set(t for t in tasks if t and t.strip()))
    unique_shards = len(set(metadata["shard_paths"]))
    unique_files = len(set(metadata["file_paths"]))

    print(f"\n  Labeled: {labeled}/{len(sampled)} (should be 100%)")
    print(f"  Unique tasks: {unique_tasks}")
    print(f"  Unique source shards: {unique_shards}")
    print(f"  Unique source files (file_paths): {unique_files}")
    print(f"  Min episode length: {min(lengths)}")
    print(f"  Max episode length: {max(lengths)}")
    print(f"  feats_B range: [{feats_B.min():.4f}, {feats_B.max():.4f}]")
    print(f"  feats_B_joint range: [{feats_B_joint.min():.4f}, {feats_B_joint.max():.4f}]")

    # fidx position diversity report
    N = len(sampled)
    pct_fidx = [100 * fidxs[i] / (lengths[i] - 1) for i in range(N)]
    first_third  = sum(1 for p in pct_fidx if p < 33.3)
    middle_third = sum(1 for p in pct_fidx if 33.3 <= p < 66.6)
    last_third   = sum(1 for p in pct_fidx if p >= 66.6)
    print(f"\n  fidx position diversity (fraction of episode elapsed at anchor):")
    print(f"    First third  (0-33%%):  {first_third:5d} ({100*first_third/N:.1f}%%)")
    print(f"    Middle third (33-66%%): {middle_third:5d} ({100*middle_third/N:.1f}%%)")
    print(f"    Last third   (66-100%%): {last_third:5d} ({100*last_third/N:.1f}%%)")


if __name__ == "__main__":
    main()
