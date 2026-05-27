"""
eda_droid.py — EDA for DROID v1.0.1 RLDS dataset.

Scans the first N_SHARDS shards and reports:
  - Episode length distribution (mean, P10, P25, P50, P75, P90, P95, max)
  - Language instruction coverage
  - Top 20 task descriptions by frequency
  - N eligible episodes at several min_length thresholds

Run with simpler_env conda env (has TensorFlow):
    conda activate simpler_env
    CUDA_VISIBLE_DEVICES="" python eda_droid.py [--n_shards 100]
"""

import argparse
import collections
import glob
import os
import time

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import tensorflow as tf

RLDS_DIR = os.environ.get(
    "DROID_RLDS_DIR", "/home/ubuntu/verl/droid/data/droid/1.0.1"
)
SHARD_GLOB = "droid_101-train.tfrecord-*-of-02048"


def scan_shard(path):
    """Read one shard, return list of (episode_length, language_instruction)."""
    results = []
    try:
        for raw in tf.data.TFRecordDataset([path]):
            ex = tf.train.Example.FromString(raw.numpy())
            fts = ex.features.feature

            is_first = list(fts["steps/is_first"].int64_list.value)
            n_steps = len(is_first)

            lang_bytes = fts["steps/language_instruction"].bytes_list.value
            lang = lang_bytes[0].decode("utf-8", errors="replace").strip() if lang_bytes else ""

            results.append((n_steps, lang))
    except Exception as e:
        print(f"  WARNING: failed to read {path}: {e}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_shards", type=int, default=100,
                        help="Number of shards to scan (default: 100)")
    parser.add_argument("--rlds_dir", type=str, default=RLDS_DIR)
    args = parser.parse_args()

    shards = sorted(glob.glob(os.path.join(args.rlds_dir, SHARD_GLOB)))
    print(f"Total shards found: {len(shards)}")
    print(f"Scanning first {args.n_shards} shards...\n")

    shards = shards[:args.n_shards]

    t0 = time.time()
    all_lengths = []
    all_langs = []

    for i, shard in enumerate(shards):
        episodes = scan_shard(shard)
        for length, lang in episodes:
            all_lengths.append(length)
            all_langs.append(lang)
        if (i + 1) % 10 == 0 or (i + 1) == len(shards):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(shards) - (i + 1)) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(shards)} shards]  eps={len(all_lengths)}"
                  f"  {rate:.1f} shards/s  ETA {eta:.0f}s")

    print(f"\nDone in {time.time()-t0:.0f}s")
    print(f"Total episodes scanned: {len(all_lengths)}")

    # --- Episode length distribution ---
    lengths = np.array(all_lengths)
    print("\n" + "="*60)
    print("EPISODE LENGTH DISTRIBUTION")
    print("="*60)
    print(f"  Count:  {len(lengths)}")
    print(f"  Mean:   {lengths.mean():.1f}")
    print(f"  Std:    {lengths.std():.1f}")
    print(f"  Min:    {lengths.min()}")
    print(f"  P10:    {np.percentile(lengths, 10):.0f}")
    print(f"  P25:    {np.percentile(lengths, 25):.0f}")
    print(f"  P50:    {np.percentile(lengths, 50):.0f}")
    print(f"  P75:    {np.percentile(lengths, 75):.0f}")
    print(f"  P90:    {np.percentile(lengths, 90):.0f}")
    print(f"  P95:    {np.percentile(lengths, 95):.0f}")
    print(f"  Max:    {lengths.max()}")

    # --- Language instruction coverage ---
    labeled = sum(1 for l in all_langs if l.strip())
    print("\n" + "="*60)
    print("LANGUAGE INSTRUCTION COVERAGE")
    print("="*60)
    print(f"  Episodes with language: {labeled}/{len(all_langs)} ({100*labeled/len(all_langs):.1f}%)")

    # --- Task distribution ---
    task_counts = collections.Counter(l for l in all_langs if l.strip())
    print("\n" + "="*60)
    print("TOP 30 TASK DESCRIPTIONS")
    print("="*60)
    for task, count in task_counts.most_common(30):
        pct = 100 * count / labeled if labeled > 0 else 0
        print(f"  {count:5d} ({pct:5.1f}%)  {task[:80]}")
    print(f"\n  Total unique tasks: {len(task_counts)}")

    # --- H_max eligibility ---
    print("\n" + "="*60)
    print("EPISODE ELIGIBILITY BY H_max (episodes with length > H_max+1)")
    print("="*60)
    for h_max in [15, 25, 40, 75, 100, 150, 200]:
        min_len = h_max + 2  # need at least H_max+1 steps after frame t
        n_eligible = (lengths > h_max).sum()
        pct = 100 * n_eligible / len(lengths)
        print(f"  H_max={h_max:4d}  (min_length={min_len}):  "
              f"{n_eligible:6d}/{len(lengths)}  ({pct:.1f}%)")

    # Extrapolate to full dataset
    scale = len(shards) / len(shards)  # = 1.0 for scanned shards
    total_shards = len(sorted(glob.glob(os.path.join(args.rlds_dir, SHARD_GLOB))))
    scale_full = total_shards / len(shards)
    print(f"\n  (Extrapolated to all {total_shards} shards, scale={scale_full:.1f}x:)")
    for h_max in [40, 75, 100]:
        n_eligible = (lengths > h_max).sum()
        print(f"  H_max={h_max:4d}:  ~{int(n_eligible * scale_full):6d} episodes in full dataset")

    # --- Histogram buckets for step counts ---
    print("\n" + "="*60)
    print("LENGTH HISTOGRAM (bucket width=50)")
    print("="*60)
    max_len = min(int(np.percentile(lengths, 99)), 2000)
    bins = list(range(0, max_len + 50, 50))
    hist, edges = np.histogram(lengths, bins=bins)
    for i, (lo, hi, cnt) in enumerate(zip(edges[:-1], edges[1:], hist)):
        bar = "#" * min(int(cnt / max(hist) * 40), 40)
        print(f"  [{lo:4d}-{hi:4d}):  {cnt:6d}  {bar}")


if __name__ == "__main__":
    main()
