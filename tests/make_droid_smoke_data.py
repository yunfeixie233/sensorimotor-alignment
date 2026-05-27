"""
make_droid_smoke_data.py

Create a tiny 5-sample smoke-test dataset from the full DROID data directory.
Copies the first N images + slices the corresponding tensor rows + rewrites metadata.json.

Usage:
    python tests/make_droid_smoke_data.py [--n 5] [--src <droid_data_dir>] [--dst <smoke_dir>]
"""

import argparse
import json
import os
import shutil

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5,
                        help="Number of samples to include in smoke dataset")
    parser.add_argument("--src", type=str,
                        default="/home/ubuntu/verl/starVLA/cknna/cknna_data_droid",
                        help="Source DROID data directory (full N=7762)")
    parser.add_argument("--dst", type=str,
                        default="/tmp/cknna_droid_smoke",
                        help="Destination smoke-test data directory")
    args = parser.parse_args()

    N = args.n
    src = args.src
    dst = args.dst
    os.makedirs(dst, exist_ok=True)
    os.makedirs(os.path.join(dst, "images"), exist_ok=True)

    print(f"Creating {N}-sample smoke dataset")
    print(f"  src: {src}")
    print(f"  dst: {dst}")

    # Load and slice metadata
    with open(os.path.join(src, "metadata.json")) as f:
        meta = json.load(f)

    smoke_meta = {
        "num_samples": N,
        "max_horizon": meta["max_horizon"],
        "state_dim": meta["state_dim"],
        "state_dim_joint": meta["state_dim_joint"],
        "action_dim": meta["action_dim"],
        "source": "droid_rlds_smoke",
        "rlds_dir": meta["rlds_dir"],
        "seed": meta["seed"],
        "min_length": meta["min_length"],
        "max_samples": N,
        "episode_indices": meta["episode_indices"][:N],
        "frame_indices": meta["frame_indices"][:N],
        "episode_lengths": meta["episode_lengths"][:N],
        "task_descriptions": meta["task_descriptions"][:N],
    }
    with open(os.path.join(dst, "metadata.json"), "w") as f:
        json.dump(smoke_meta, f, indent=2)

    # Slice tensors
    tensor_files = [
        "feats_B.pt", "feats_B_seq.pt",
        "feats_B_joint.pt", "feats_B_seq_joint.pt",
        "actions.pt", "actions_seq.pt",
    ]
    for fname in tensor_files:
        src_path = os.path.join(src, fname)
        dst_path = os.path.join(dst, fname)
        if not os.path.exists(src_path):
            print(f"  WARNING: {fname} not found, skipping")
            continue
        t = torch.load(src_path, weights_only=True)
        torch.save(t[:N], dst_path)
        print(f"  {fname}: {tuple(t[:N].shape)}")

    # Copy images
    for i in range(N):
        src_img = os.path.join(src, "images", f"{i:06d}.png")
        dst_img = os.path.join(dst, "images", f"{i:06d}.png")
        if not os.path.exists(src_img):
            print(f"  WARNING: image {i:06d}.png not found")
            continue
        shutil.copy2(src_img, dst_img)
    print(f"  Copied {N} images")

    print(f"\nSmoke dataset ready at: {dst}")
    print(f"  Tasks: {smoke_meta['task_descriptions']}")


if __name__ == "__main__":
    main()
