"""
Build a filtered Bridge V2 subset for the ViT-levels CKNNA experiment.

Filter criteria:
  1. Non-empty task description
  2. max_valid_horizon >= 40 (no padding at h=40)
  3. Per-task cap of 3 occurrences (reduce dominance of "sweep into pile" etc.)

Outputs a new data directory with:
  - metadata.json (filtered, with index_map to original)
  - feats_B_seq.pt (subset)
  - feats_B.pt (subset, single-step)
  - images/ symlink (images are indexed by original idx stored in metadata)
  - For each model: feats_A.pt, feats_A_vit.pt, feats_A_pre_llm.pt, feats_A_img.pt, feats_A_txt.pt
"""

import argparse
import json
import os
import shutil
from collections import Counter

import torch
import numpy as np


def compute_max_valid_horizon(feats_B_seq):
    """Detect padding: max horizon where consecutive frames differ."""
    diff = (feats_B_seq[:, 1:] != feats_B_seq[:, :-1]).any(dim=-1)
    max_h = torch.zeros(feats_B_seq.shape[0], dtype=torch.long)
    for i in range(feats_B_seq.shape[0]):
        nz = diff[i].nonzero()
        max_h[i] = nz[-1].item() + 1 if len(nz) > 0 else 0
    return max_h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True,
                        help="Raw Bridge cache produced by data/load_bridge_rlds.py (canonically $DATA_STORE/cknna_data_bridge_raw)")
    parser.add_argument("--dst_dir", required=True,
                        help="Output filtered Bridge cache (canonically $DATA_STORE/cknna_data_bridge)")
    parser.add_argument("--min_horizon", type=int, default=40)
    parser.add_argument("--task_cap", type=int, default=3)
    parser.add_argument("--models", nargs="+", default=[
        "qwen25vl-3b-raw", "qwen25vl-7b-raw", "qwen3vl-2b-raw",
        "qwen3vl-4b-raw", "qwen3vl-8b-raw", "paligemma1-raw",
        "paligemma2-raw", "kosmos2-raw",
    ])
    args = parser.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)

    # Load source
    with open(os.path.join(args.src_dir, "metadata.json")) as f:
        meta = json.load(f)
    tasks = meta["task_descriptions"]
    N = meta["num_samples"]

    feats_B_seq = torch.load(
        os.path.join(args.src_dir, "feats_B_seq.pt"), map_location="cpu"
    )
    print(f"Source: N={N}, feats_B_seq={tuple(feats_B_seq.shape)}")

    # Step 1: padding filter
    max_valid_h = compute_max_valid_horizon(feats_B_seq)
    valid_h = max_valid_h >= args.min_horizon

    # Step 2: non-empty task filter
    nonempty = torch.tensor([bool(t and t.strip()) for t in tasks])

    # Step 3: combined + task cap
    candidates = []
    for i in range(N):
        if nonempty[i] and valid_h[i]:
            candidates.append((i, tasks[i]))

    task_count = Counter()
    selected = []
    for i, t in candidates:
        if task_count[t] < args.task_cap:
            selected.append(i)
            task_count[t] += 1
    selected = sorted(selected)
    N_new = len(selected)

    unique_tasks = len(set(tasks[i] for i in selected))
    print(f"\nFilter: non-empty + h>={args.min_horizon} + cap={args.task_cap}")
    print(f"  Result: N={N_new}, unique_tasks={unique_tasks}, ratio={unique_tasks/N_new*100:.1f}%")
    print(f"  Top task: {Counter(tasks[i] for i in selected).most_common(1)}")

    # Save index map
    idx_map = torch.tensor(selected, dtype=torch.long)

    # Save filtered metadata
    new_meta = {
        "num_samples": N_new,
        "max_horizon": args.min_horizon,
        "task_descriptions": [tasks[i] for i in selected],
        "source_dir": args.src_dir,
        "filter": {
            "non_empty_task": True,
            "min_valid_horizon": args.min_horizon,
            "task_cap": args.task_cap,
        },
        "index_map": selected,  # index into original dataset
    }
    if "episode_lengths" in meta:
        new_meta["episode_lengths"] = [meta["episode_lengths"][i] for i in selected]

    with open(os.path.join(args.dst_dir, "metadata.json"), "w") as f:
        json.dump(new_meta, f, indent=2)
    print(f"  Saved metadata.json")

    # Save filtered feats_B_seq
    new_B_seq = feats_B_seq[idx_map]
    torch.save(new_B_seq, os.path.join(args.dst_dir, "feats_B_seq.pt"))
    print(f"  Saved feats_B_seq.pt: {tuple(new_B_seq.shape)}")

    # Save filtered feats_B (single-step)
    B_path = os.path.join(args.src_dir, "feats_B.pt")
    if os.path.exists(B_path):
        feats_B = torch.load(B_path, map_location="cpu")
        torch.save(feats_B[idx_map], os.path.join(args.dst_dir, "feats_B.pt"))
        print(f"  Saved feats_B.pt")

    # Symlink images directory
    src_images = os.path.join(args.src_dir, "images")
    dst_images = os.path.join(args.dst_dir, "images")
    if os.path.exists(src_images) and not os.path.exists(dst_images):
        os.symlink(src_images, dst_images)
        print(f"  Symlinked images/")

    # Subset model features
    for model in args.models:
        src_model = os.path.join(args.src_dir, model)
        dst_model = os.path.join(args.dst_dir, model)
        if not os.path.exists(src_model):
            print(f"  [SKIP] {model}: not found in source")
            continue
        os.makedirs(dst_model, exist_ok=True)

        for fname in ["feats_A.pt", "feats_A_img.pt", "feats_A_txt.pt",
                       "feats_A_vit.pt", "feats_A_pre_llm.pt"]:
            src_f = os.path.join(src_model, fname)
            if not os.path.exists(src_f):
                continue
            feats = torch.load(src_f, map_location="cpu")
            torch.save(feats[idx_map], os.path.join(dst_model, fname))

        print(f"  Saved {model}/ features (subset)")

    print(f"\nDone. New dataset: {args.dst_dir} (N={N_new})")


if __name__ == "__main__":
    main()
