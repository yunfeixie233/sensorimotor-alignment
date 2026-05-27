"""make_tiny_subset.py -- Stratified 1000-per-group subset of cknna_data_exp1.

Groups (based on episode_length, labeled episodes only):
  S: 16 <= L <= 25
  M: 26 <= L <= 40
  L: L > 40

Usage:
    python make_tiny_subset.py \
        --full_dir cknna_data_exp1 \
        --tiny_dir cknna_data_exp1_tiny \
        --n_per_group 1000 \
        --seed 42
"""
import argparse
import json
import os
import random
import shutil

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_dir", required=True)
    parser.add_argument("--tiny_dir", required=True)
    parser.add_argument("--n_per_group", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    with open(os.path.join(args.full_dir, "metadata.json")) as f:
        meta = json.load(f)

    N = meta["num_samples"]
    task_descriptions = meta["task_descriptions"]
    episode_lengths = meta["episode_lengths"]

    group_indices = {"S": [], "M": [], "L": []}
    for i in range(N):
        if not task_descriptions[i] or not task_descriptions[i].strip():
            continue
        L = episode_lengths[i]
        if 16 <= L <= 25:
            group_indices["S"].append(i)
        elif 26 <= L <= 40:
            group_indices["M"].append(i)
        elif L > 40:
            group_indices["L"].append(i)

    print(f"Full dataset: {N} total samples")
    for g, idxs in group_indices.items():
        print(f"  Group {g}: {len(idxs)} eligible labeled samples")

    selected = []
    for g, idxs in group_indices.items():
        sample = rng.sample(idxs, min(args.n_per_group, len(idxs)))
        print(f"  Group {g}: sampled {len(sample)}")
        selected.extend(sample)
    selected = sorted(selected)
    print(f"Total selected: {len(selected)}")

    os.makedirs(args.tiny_dir, exist_ok=True)

    for fname in ["feats_B.pt", "feats_B_seq.pt", "actions.pt", "actions_seq.pt"]:
        src = os.path.join(args.full_dir, fname)
        t = torch.load(src, weights_only=True)
        tiny = t[selected]
        torch.save(tiny, os.path.join(args.tiny_dir, fname))
        print(f"Saved {fname}: {tuple(tiny.shape)}")

    img_src = os.path.join(args.full_dir, "images")
    img_dst = os.path.join(args.tiny_dir, "images")
    os.makedirs(img_dst, exist_ok=True)
    for new_idx, old_idx in enumerate(selected):
        shutil.copy2(
            os.path.join(img_src, f"{old_idx:06d}.png"),
            os.path.join(img_dst, f"{new_idx:06d}.png"),
        )
    print(f"Copied {len(selected)} images to {img_dst}/")

    tiny_meta = dict(meta)
    tiny_meta["num_samples"] = len(selected)
    for key in ["task_descriptions", "episode_indices", "frame_indices", "episode_lengths"]:
        tiny_meta[key] = [meta[key][i] for i in selected]
    with open(os.path.join(args.tiny_dir, "metadata.json"), "w") as f:
        json.dump(tiny_meta, f, indent=2)

    counts = {
        "S": sum(1 for L in tiny_meta["episode_lengths"] if 16 <= L <= 25),
        "M": sum(1 for L in tiny_meta["episode_lengths"] if 26 <= L <= 40),
        "L": sum(1 for L in tiny_meta["episode_lengths"] if L > 40),
    }
    print(f"Written {args.tiny_dir}/metadata.json: {len(selected)} total")
    print(f"  Group S={counts['S']}, M={counts['M']}, L={counts['L']}")


if __name__ == "__main__":
    main()
