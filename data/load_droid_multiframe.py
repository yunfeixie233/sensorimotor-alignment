"""
Phase 1 (Multi-Frame): Extract multi-frame observation sequences from DROID RLDS
for the multi-frame CKNNA experiment.

This script re-reads the ORIGINAL RLDS TFRecord shards and extracts m_max+1
consecutive frames [k-m_max, k-m_max+1, ..., k] ending at each anchor position k,
using the SAME anchor positions from an existing Phase 1 dataset.

Design:
  - Reads metadata.json from an existing Phase 1 output dir to get the exact
    (shard_path, shard_record_idx, frame_index) for each sample.
  - Filters to anchors with k >= m_max (enough lookback room).
  - For each valid anchor, extracts m_max+1 consecutive PNG frames.
  - Uses multiprocessing to read shards in parallel (--num_workers).
  - Copies the B-side tensors (feats_B_seq.pt, actions_seq.pt, etc.) for the
    filtered subset — these are UNCHANGED (same actions, same states).
  - The result is a new data directory with multi-frame images and the same
    B-side data, ready for multi-frame feature extraction.

Output structure:
  <output_dir>/
  ├── images/NNNNNN/           # one subdir per sample
  │   ├── 000.png              # frame at k - m_max
  │   ├── 001.png              # frame at k - m_max + 1
  │   ├── ...
  │   └── {m_max:03d}.png      # frame at k (the original anchor frame)
  ├── feats_B.pt               # (N_filtered, 7) states at anchor k
  ├── feats_B_seq.pt           # (N_filtered, H+1, 7) states [k, k+1, ..., k+H]
  ├── feats_B_joint.pt         # (N_filtered, 8) joint states at anchor k
  ├── feats_B_seq_joint.pt     # (N_filtered, H+1, 8)
  ├── actions.pt               # (N_filtered, 7) action at anchor k
  ├── actions_seq.pt           # (N_filtered, H+1, 7)
  ├── actions_history.pt       # (N_filtered, m_max+1, 7) actions [k-m_max, ..., k]
  ├── metadata.json            # filtered metadata + m_max info
  └── index_map.json           # maps new indices to original Phase 1 indices

Usage:
  CUDA_VISIBLE_DEVICES="" python data/load_droid_multiframe.py \
      --source_dir /path/to/cknna_data_droid \
      --output_dir /path/to/cknna_data_droid_multiframe \
      --m_max 64 \
      --num_workers 28 \
      --rlds_dir /lambda/nfs/vla/cache/droid/data/droid/1.0.1
"""

import argparse
import io
import json
import os
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import torch
from PIL import Image

CAM_KEYS = {
    "cam1": "steps/observation/exterior_image_1_left",
    "cam2": "steps/observation/exterior_image_2_left",
    "cam3": "steps/observation/wrist_image_left",
}
# Output dirs: images/ (cam1), images_cam2/, images_cam3/
CAM_DIRS = {
    "cam1": "images",
    "cam2": "images_cam2",
    "cam3": "images_cam3",
}


def _process_shard(args_tuple):
    """Worker function: read one shard, extract and write all multi-frame images
    for all 3 cameras.

    Each worker imports TF independently to avoid fork issues.
    Each sample writes to its own subdir — no write conflicts between workers.

    Returns: number of images written by this worker.
    """
    shard_path, items, base_images_dirs, m_max = args_tuple

    import tensorflow as tf

    # Build lookup: record_idx -> list of (new_idx, fidx)
    record_lookup = defaultdict(list)
    for new_idx, rec_idx, fidx in items:
        record_lookup[rec_idx].append((new_idx, fidx))

    n_written = 0
    record_idx = 0
    for raw in tf.data.TFRecordDataset([shard_path]):
        if record_idx in record_lookup:
            ex = tf.train.Example.FromString(raw.numpy())
            fts = ex.features.feature

            for new_idx, fidx in record_lookup[record_idx]:
                # Check if all 3 cameras are complete for this sample
                all_complete = True
                for cam_name, cam_dir_name in CAM_DIRS.items():
                    sample_dir = os.path.join(base_images_dirs[cam_name], f"{new_idx:06d}")
                    os.makedirs(sample_dir, exist_ok=True)
                    existing = len([f for f in os.listdir(sample_dir) if f.endswith('.png')])
                    if existing < m_max + 1:
                        all_complete = False
                        break
                if all_complete:
                    n_written += (m_max + 1) * len(CAM_KEYS)
                    continue

                # Extract frames for each camera
                for cam_name, cam_key in CAM_KEYS.items():
                    img_bytes_list = fts[cam_key].bytes_list.value
                    sample_dir = os.path.join(base_images_dirs[cam_name], f"{new_idx:06d}")
                    os.makedirs(sample_dir, exist_ok=True)

                    for frame_offset in range(m_max + 1):
                        frame_pos = fidx - m_max + frame_offset
                        out_path = os.path.join(sample_dir, f"{frame_offset:03d}.png")
                        if os.path.exists(out_path):
                            n_written += 1
                            continue
                        raw_bytes = bytes(img_bytes_list[frame_pos])
                        img = Image.open(io.BytesIO(raw_bytes))
                        img.save(out_path)
                        n_written += 1

        record_idx += 1

    return n_written


def _process_shard_actions(args_tuple):
    """Worker: read one shard, extract action history [k-m_max, ..., k] for all
    samples in this shard.

    Returns: list of (new_idx, actions_array) tuples where actions_array is
             np.ndarray of shape (m_max+1, 7).
    """
    shard_path, items, m_max = args_tuple

    import tensorflow as tf

    record_lookup = defaultdict(list)
    for new_idx, rec_idx, fidx in items:
        record_lookup[rec_idx].append((new_idx, fidx))

    results = []
    record_idx = 0
    for raw in tf.data.TFRecordDataset([shard_path]):
        if record_idx in record_lookup:
            ex = tf.train.Example.FromString(raw.numpy())
            fts = ex.features.feature

            n_steps = len(fts["steps/is_first"].int64_list.value)
            actions_all = np.array(
                fts["steps/action"].float_list.value, dtype=np.float32
            ).reshape(n_steps, 7)

            for new_idx, fidx in record_lookup[record_idx]:
                start = fidx - m_max
                end = fidx + 1
                assert start >= 0, f"fidx={fidx} < m_max={m_max}"
                actions_hist = actions_all[start:end]  # (m_max+1, 7)
                assert actions_hist.shape == (m_max + 1, 7), \
                    f"Sample {new_idx}: expected ({m_max+1}, 7), got {actions_hist.shape}"
                results.append((new_idx, actions_hist))
        record_idx += 1

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 (Multi-Frame): Extract multi-frame observation sequences from DROID RLDS"
    )
    parser.add_argument(
        "--source_dir", type=str, required=True,
        help="Path to existing Phase 1 output (e.g. cknna_data_droid)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for multi-frame data"
    )
    parser.add_argument(
        "--m_max", type=int, default=64,
        help="Maximum lookback frames. Each sample stores m_max+1 frames [k-m_max, ..., k]"
    )
    parser.add_argument(
        "--rlds_dir", type=str,
        default="/home/ubuntu/verl/droid/data/droid/1.0.1",
        help="Path to DROID RLDS TFRecord directory"
    )
    parser.add_argument(
        "--num_workers", type=int, default=28,
        help="Number of parallel workers for shard reading"
    )
    args = parser.parse_args()

    # ---- Load existing Phase 1 metadata ----
    meta_path = os.path.join(args.source_dir, "metadata.json")
    print(f"Loading source metadata: {meta_path}")
    with open(meta_path) as f:
        source_meta = json.load(f)

    N_source = source_meta["num_samples"]
    H = source_meta["max_horizon"]
    frame_indices = source_meta["frame_indices"]
    episode_lengths = source_meta["episode_lengths"]
    shard_paths = source_meta["shard_paths"]
    shard_record_indices = source_meta["shard_record_indices"]

    print(f"Source: N={N_source}, max_horizon={H}")

    # Validate metadata array lengths
    assert len(frame_indices) == len(episode_lengths) == len(shard_paths) == len(shard_record_indices) == N_source, \
        "Metadata arrays have mismatched lengths"

    # ---- Filter to anchors with k >= m_max AND k < episode_length ----
    valid_mask = [
        k >= args.m_max and k < ep_len
        for k, ep_len in zip(frame_indices, episode_lengths)
    ]
    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    N_filtered = len(valid_indices)

    print(f"\nFiltering to k >= {args.m_max}:")
    print(f"  {N_filtered} / {N_source} anchors survive ({100*N_filtered/N_source:.1f}%)")

    if N_filtered == 0:
        raise ValueError(f"No anchors with k >= {args.m_max}. Reduce m_max.")

    # ---- Group by shard for efficient reading ----
    # Map: shard_path -> [(new_idx, shard_record_idx, fidx)]
    shard_groups = defaultdict(list)
    for new_idx, orig_idx in enumerate(valid_indices):
        shard = shard_paths[orig_idx]
        rec_idx = shard_record_indices[orig_idx]
        fidx = frame_indices[orig_idx]
        shard_groups[shard].append((new_idx, rec_idx, fidx))

    print(f"  Samples spread across {len(shard_groups)} unique shards")

    # ---- Prepare output directories (3 cameras) ----
    os.makedirs(args.output_dir, exist_ok=True)
    base_images_dirs = {}
    for cam_name, cam_dir_name in CAM_DIRS.items():
        d = os.path.join(args.output_dir, cam_dir_name)
        os.makedirs(d, exist_ok=True)
        base_images_dirs[cam_name] = d

    # Pre-create per-sample subdirectories for all 3 cameras
    print("Creating sample directories (3 cameras)...")
    for new_idx in range(N_filtered):
        for cam_dir in base_images_dirs.values():
            os.makedirs(os.path.join(cam_dir, f"{new_idx:06d}"), exist_ok=True)

    # ---- Check resume state (all 3 cameras must be complete) ----
    cam1_dir = base_images_dirs["cam1"]
    existing_count = sum(
        1 for d in os.listdir(cam1_dir)
        if os.path.isdir(os.path.join(cam1_dir, d))
        and len([f for f in os.listdir(os.path.join(cam1_dir, d)) if f.endswith('.png')]) >= args.m_max + 1
    )
    total_imgs = N_filtered * (args.m_max + 1) * 3  # 3 cameras
    print(f"  Already complete (cam1): {existing_count} / {N_filtered} samples")
    if existing_count == N_filtered:
        print("  All images already extracted, skipping to tensor filtering.")
    else:
        # ---- Parallel shard reading ----
        print(f"\nReading {len(shard_groups)} shards with {args.num_workers} parallel workers...")
        print(f"  Target: {total_imgs} images ({N_filtered} samples × {args.m_max+1} frames × 3 cameras)")

        # Build work items: one per shard
        work_items = [
            (shard_path, items, base_images_dirs, args.m_max)
            for shard_path, items in shard_groups.items()
        ]

        t0 = time.time()
        completed_shards = 0
        total_written = 0

        with Pool(processes=args.num_workers) as pool:
            for n_written in pool.imap_unordered(_process_shard, work_items):
                completed_shards += 1
                total_written += n_written
                if completed_shards % 50 == 0 or completed_shards == len(work_items):
                    elapsed = time.time() - t0
                    rate = completed_shards / elapsed if elapsed > 0 else 0
                    eta = (len(work_items) - completed_shards) / rate if rate > 0 else 0
                    print(
                        f"  [{completed_shards}/{len(work_items)} shards]  "
                        f"imgs_written={total_written}  "
                        f"{rate:.1f} shards/s  ETA={eta:.0f}s"
                    )

        t1 = time.time()
        print(f"\nImage extraction complete in {t1-t0:.0f}s ({(t1-t0)/60:.1f} min)")
        print(f"  Total images written: {total_written}")

    # ---- Verify completeness (all 3 cameras) ----
    print("\nVerifying all images extracted (3 cameras)...")
    incomplete = []
    for new_idx in range(N_filtered):
        for cam_name, cam_dir in base_images_dirs.items():
            sample_dir = os.path.join(cam_dir, f"{new_idx:06d}")
            n_files = len([f for f in os.listdir(sample_dir) if f.endswith('.png')])
            if n_files < args.m_max + 1:
                incomplete.append((new_idx, cam_name, n_files))
    if incomplete:
        print(f"  WARNING: {len(incomplete)} sample-camera pairs incomplete!")
        for idx, cam, n in incomplete[:10]:
            print(f"    Sample {idx:06d} {cam}: {n}/{args.m_max+1} images")
        raise RuntimeError(f"{len(incomplete)} sample-camera pairs incomplete. Re-run to retry.")
    else:
        print(f"  All {N_filtered} samples × 3 cameras have {args.m_max+1} images each. OK.")

    # ---- Copy and filter B-side tensors ----
    print("\nFiltering B-side tensors...")
    idx_tensor = torch.tensor(valid_indices, dtype=torch.long)

    for tensor_name in [
        "feats_B.pt", "feats_B_seq.pt",
        "feats_B_joint.pt", "feats_B_seq_joint.pt",
        "actions.pt", "actions_seq.pt",
    ]:
        src_path = os.path.join(args.source_dir, tensor_name)
        dst_path = os.path.join(args.output_dir, tensor_name)
        if os.path.exists(dst_path):
            print(f"  {tensor_name}: already exists, skipping")
            continue
        if os.path.exists(src_path):
            t = torch.load(src_path, weights_only=True)
            t_filtered = t[idx_tensor]
            torch.save(t_filtered, dst_path)
            print(f"  {tensor_name}: {tuple(t.shape)} -> {tuple(t_filtered.shape)}")
        else:
            print(f"  {tensor_name}: not found in source, skipping")

    # ---- Extract action history from TFRecords ----
    actions_hist_path = os.path.join(args.output_dir, "actions_history.pt")
    if os.path.exists(actions_hist_path):
        print(f"\nactions_history.pt already exists, skipping action extraction")
    else:
        print(f"\nExtracting action history from TFRecords ({N_filtered} samples)...")
        actions_history = np.zeros((N_filtered, args.m_max + 1, 7), dtype=np.float32)

        action_work_items = [
            (shard_path, items, args.m_max)
            for shard_path, items in shard_groups.items()
        ]

        t_act = time.time()
        completed = 0
        with Pool(processes=args.num_workers) as pool:
            for result_list in pool.imap_unordered(_process_shard_actions, action_work_items):
                for new_idx, act_hist in result_list:
                    actions_history[new_idx] = act_hist
                completed += 1
                if completed % 200 == 0 or completed == len(action_work_items):
                    print(f"  [{completed}/{len(action_work_items)} shards]")

        actions_history_tensor = torch.from_numpy(actions_history)
        torch.save(actions_history_tensor, actions_hist_path)
        print(f"  Saved: {actions_hist_path}")
        print(f"  Shape: {tuple(actions_history_tensor.shape)}")
        print(f"  Time: {time.time() - t_act:.0f}s")

        # Verification: actions_history[:, -1, :] should match actions.pt
        actions_anchor_path = os.path.join(args.output_dir, "actions.pt")
        if os.path.exists(actions_anchor_path):
            actions_anchor = torch.load(actions_anchor_path, weights_only=True)
            diff = (actions_history_tensor[:, -1, :] - actions_anchor).abs().max().item()
            print(f"  Verification (max diff at anchor): {diff:.2e}")
            assert diff < 1e-5, f"Action history verification failed: max diff = {diff}"
            print(f"  Verification PASSED")

    # ---- Save metadata ----
    meta_path = os.path.join(args.output_dir, "metadata.json")
    if not os.path.exists(meta_path):
        new_meta = {
            "num_samples": N_filtered,
            "max_horizon": H,
            "m_max": args.m_max,
            "min_length": source_meta.get("min_length", -1),
            "max_samples": source_meta.get("max_samples", -1),
            "state_dim": source_meta.get("state_dim", 7),
            "state_dim_joint": source_meta.get("state_dim_joint", 8),
            "action_dim": source_meta.get("action_dim", 7),
            "source": "droid_rlds_multiframe",
            "source_dir": args.source_dir,
            "rlds_dir": args.rlds_dir,
            "seed": source_meta.get("seed", -1),
            "shard_paths": [shard_paths[i] for i in valid_indices],
            "shard_record_indices": [shard_record_indices[i] for i in valid_indices],
            "file_paths": [source_meta["file_paths"][i] for i in valid_indices],
            "episode_indices": list(range(N_filtered)),
            "frame_indices": [frame_indices[i] for i in valid_indices],
            "episode_lengths": [episode_lengths[i] for i in valid_indices],
            "task_descriptions": [source_meta["task_descriptions"][i] for i in valid_indices],
        }
        with open(meta_path, "w") as f:
            json.dump(new_meta, f, indent=2)
        print(f"\nMetadata saved: {meta_path}")
    else:
        print(f"\nMetadata already exists: {meta_path}")

    # ---- Save index map ----
    index_map_path = os.path.join(args.output_dir, "index_map.json")
    if not os.path.exists(index_map_path):
        index_map = {
            "description": "Maps new indices to original Phase 1 indices",
            "source_dir": args.source_dir,
            "m_max": args.m_max,
            "mapping": {str(new_idx): orig_idx for new_idx, orig_idx in enumerate(valid_indices)},
        }
        with open(index_map_path, "w") as f:
            json.dump(index_map, f, indent=2)

    # ---- Summary ----
    fidxs = [frame_indices[i] for i in valid_indices]
    print(f"\n=== Phase 1 (Multi-Frame) complete ===")
    print(f"  N = {N_filtered} samples (from {N_source} source)")
    print(f"  m_max = {args.m_max} (frames per sample: {args.m_max + 1})")
    print(f"  Total images: {N_filtered * (args.m_max + 1)}")
    print(f"  Anchor k range: [{min(fidxs)}, {max(fidxs)}]")
    print(f"  Anchor k median: {np.median(fidxs):.0f}")
    print(f"  Workers used: {args.num_workers}")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
