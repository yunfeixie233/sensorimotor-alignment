"""
Extract additional camera images from DROID RLDS for 3-camera CKNNA.

The existing load_droid_rlds.py saves only exterior_image_1_left as images/NNNNNN.png.
This script re-reads the RLDS TFRecords using the saved metadata (shard_paths,
shard_record_indices, frame_indices) to extract the other 2 cameras:
  - exterior_image_2_left → images_cam2/NNNNNN.png
  - wrist_image_left      → images_cam3/NNNNNN.png

This avoids re-running the full data pipeline. It uses the exact same episodes
and frame indices from metadata.json for perfect alignment.

Usage:
  CUDA_VISIBLE_DEVICES="" /usr/bin/python3 data/extract_droid_3cam.py \
      --data_dir /path/to/cknna_data_droid
"""

import argparse
import io
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import tensorflow as tf
from PIL import Image as PILImage

PNG_COMPRESS = 1
IMAGE_WORKERS = 16

# Camera keys to extract (cam1 = exterior_image_1_left already saved in images/)
CAM2_KEY = "steps/observation/exterior_image_2_left"
CAM3_KEY = "steps/observation/wrist_image_left"


def _write_png(path, img_bytes):
    """Decode JPEG bytes and write PNG to disk."""
    img = PILImage.open(io.BytesIO(img_bytes))
    img.save(path, format="PNG", compress_level=PNG_COMPRESS)


def main():
    parser = argparse.ArgumentParser(
        description="Extract cam2 + cam3 images from DROID RLDS for 3-camera CKNNA"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="DROID CKNNA data dir (with metadata.json and images/)")
    args = parser.parse_args()

    # Load metadata
    meta_path = os.path.join(args.data_dir, "metadata.json")
    with open(meta_path) as f:
        metadata = json.load(f)

    N = metadata["num_samples"]
    shard_paths = metadata["shard_paths"]
    shard_record_indices = metadata["shard_record_indices"]
    frame_indices = metadata["frame_indices"]

    print(f"=== Extract 3-camera images from DROID RLDS ===")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Samples: {N}")
    print(f"  Unique shards: {len(set(shard_paths))}")

    # Create output directories
    cam2_dir = os.path.join(args.data_dir, "images_cam2")
    cam3_dir = os.path.join(args.data_dir, "images_cam3")
    os.makedirs(cam2_dir, exist_ok=True)
    os.makedirs(cam3_dir, exist_ok=True)

    # Check which samples already exist (for resume support)
    existing_cam2 = set()
    existing_cam3 = set()
    for i in range(N):
        if os.path.exists(os.path.join(cam2_dir, f"{i:06d}.png")):
            existing_cam2.add(i)
        if os.path.exists(os.path.join(cam3_dir, f"{i:06d}.png")):
            existing_cam3.add(i)

    already_done = existing_cam2 & existing_cam3
    if len(already_done) == N:
        print("All camera images already extracted. Nothing to do.")
        return
    print(f"  Already extracted: {len(already_done)}/{N}")

    # Group samples by shard for efficient reading
    # shard_path → [(global_idx, record_idx, frame_idx), ...]
    shard_groups = defaultdict(list)
    for i in range(N):
        if i in already_done:
            continue
        shard_groups[shard_paths[i]].append(
            (i, shard_record_indices[i], frame_indices[i])
        )

    print(f"  Shards to read: {len(shard_groups)}")

    t0 = time.time()
    extracted = 0
    futures = []

    with ThreadPoolExecutor(max_workers=IMAGE_WORKERS) as pool:
        for shard_idx, (shard_path, samples) in enumerate(shard_groups.items()):
            # Build lookup: record_idx → [(global_idx, frame_idx), ...]
            record_lookup = defaultdict(list)
            for global_idx, rec_idx, fidx in samples:
                record_lookup[rec_idx].append((global_idx, fidx))

            # Read shard
            record_counter = 0
            for raw in tf.data.TFRecordDataset([shard_path]):
                if record_counter not in record_lookup:
                    record_counter += 1
                    continue

                ex = tf.train.Example.FromString(raw.numpy())
                fts = ex.features.feature

                for global_idx, fidx in record_lookup[record_counter]:
                    # Extract cam2
                    if global_idx not in existing_cam2:
                        cam2_bytes = bytes(fts[CAM2_KEY].bytes_list.value[fidx])
                        cam2_path = os.path.join(cam2_dir, f"{global_idx:06d}.png")
                        futures.append(pool.submit(_write_png, cam2_path, cam2_bytes))

                    # Extract cam3
                    if global_idx not in existing_cam3:
                        cam3_bytes = bytes(fts[CAM3_KEY].bytes_list.value[fidx])
                        cam3_path = os.path.join(cam3_dir, f"{global_idx:06d}.png")
                        futures.append(pool.submit(_write_png, cam3_path, cam3_bytes))

                    extracted += 1

                record_counter += 1

            if (shard_idx + 1) % 50 == 0 or (shard_idx + 1) == len(shard_groups):
                elapsed = time.time() - t0
                done_imgs = sum(1 for f in futures if f.done())
                print(f"  [{shard_idx+1}/{len(shard_groups)} shards]"
                      f"  samples={extracted}  imgs_written={done_imgs}"
                      f"  {elapsed:.0f}s")

        print("  Waiting for remaining image writes...")
        for fut in as_completed(futures):
            fut.result()

    elapsed = time.time() - t0
    print(f"\n=== Done in {elapsed:.0f}s ({elapsed/60:.1f} min) ===")
    print(f"  Extracted {extracted} samples × 2 cameras = {len(futures)} images")
    print(f"  cam2: {cam2_dir}")
    print(f"  cam3: {cam3_dir}")

    # Verify counts
    cam2_count = len([f for f in os.listdir(cam2_dir) if f.endswith('.png')])
    cam3_count = len([f for f in os.listdir(cam3_dir) if f.endswith('.png')])
    print(f"\n  Verification: cam2={cam2_count} cam3={cam3_count} (expected {N} each)")
    if cam2_count != N or cam3_count != N:
        print("  WARNING: counts don't match! Some images may be missing.")


if __name__ == "__main__":
    main()
