"""
Extract small example data (N=20) from the DROID experiment for testing.

Run this once to create example_feats.pt and example_trajs.pt.
Also saves the precomputed DTW top-k cache for full N=6000 reference check.

Usage:
    python prepare_example_data.py
"""

import os
import numpy as np
import torch

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_ROOT = os.environ.get("DATA_STORE", "/lambda/nfs/vla/cache/cknna_data_store")
DATA_DIR   = os.path.join(CACHE_ROOT, "cknna_data_droid")
FEAT_DIR   = os.path.join(CACHE_ROOT, "cknna_data_droid_7feat")
DTW_CACHE  = os.path.join(CACHE_ROOT, "record/dtw_cache/droid")

MODEL   = "cogact-base-bridge"
HORIZON = 40
N_SMALL = 20


def main():
    print("Loading full DROID data...")

    # Load trajectories [6000, 221, 7]
    feats_B_seq = torch.load(
        os.path.join(DATA_DIR, "feats_B_seq.pt"), weights_only=True
    ).float()
    print(f"  feats_B_seq: {feats_B_seq.shape}")

    # Load features [6000, 4096]
    feats_A = torch.load(
        os.path.join(FEAT_DIR, MODEL, "feats_A_img.pt"), weights_only=True
    ).float()
    print(f"  feats_A: {feats_A.shape}")

    # Small example: first 20 samples (.clone() to avoid saving full storage)
    trajs_small = feats_B_seq[:N_SMALL, 1:HORIZON+1, :].clone()  # [20, 40, 7]
    feats_small = feats_A[:N_SMALL].clone()                        # [20, 4096]

    torch.save(trajs_small, os.path.join(EXAMPLE_DIR, "example_trajs.pt"))
    torch.save(feats_small, os.path.join(EXAMPLE_DIR, "example_feats.pt"))
    print(f"  Saved example_trajs.pt {trajs_small.shape}")
    print(f"  Saved example_feats.pt {feats_small.shape}")

    # Full N=6000 data for reference check
    trajs_full = feats_B_seq[:, 1:HORIZON+1, :]  # [6000, 40, 7]
    torch.save(trajs_full, os.path.join(EXAMPLE_DIR, "full_trajs_h40.pt"))
    torch.save(feats_A, os.path.join(EXAMPLE_DIR, "full_feats_img.pt"))
    print(f"  Saved full_trajs_h40.pt {trajs_full.shape}")
    print(f"  Saved full_feats_img.pt {feats_A.shape}")

    # Copy DTW cache
    dtw_src = os.path.join(DTW_CACHE, f"dtw_topk_h{HORIZON}_k10_sym2_cuda_nowin_nopad.npy")
    dtw_topk = np.load(dtw_src)
    np.save(os.path.join(EXAMPLE_DIR, "dtw_topk_h40_k10.npy"), dtw_topk)
    print(f"  Saved dtw_topk_h40_k10.npy {dtw_topk.shape}")

    print("\nDone. Now run: python run_example.py")


if __name__ == "__main__":
    main()
