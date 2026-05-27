"""
merge_exp1v2_tiny_groups.py

Merge the 3 separate group directories (S, M, L) into a single combined
dataset directory (N=3000) for computing CKNNA on all samples pooled together.

Since the 3 groups have different max_horizon (S:15, M:25, L:40), feats_B_seq
shapes differ (1000x16x7, 1000x26x7, 1000x41x7). We truncate all to the
minimum shared length (16 columns = max_horizon=15) so that shared horizons
{1, 3, 7, 15} can be evaluated on the combined data.

Usage:
    cd /home/ubuntu/verl/starVLA/cknna
    python merge_exp1v2_tiny_groups.py

After running, use run_dtw_cknna.py on the merged directory:
    python record/scripts/run_dtw_cknna.py \
        --data_dir cknna_data_exp1v2_tiny_ALL \
        --dtw_version sym2_cuda_nowin \
        --horizons 1 3 7 15 \
        --device cuda
"""

import json
import os
import shutil

import torch

DATA_STORE = os.environ.get(
    "DATA_STORE", "/home/ubuntu/verl/starVLA/cknna"
)

GROUP_DIRS = {
    "S": os.path.join(DATA_STORE, "cknna_data_exp1v2_tiny_S"),
    "M": os.path.join(DATA_STORE, "cknna_data_exp1v2_tiny_M"),
    "L": os.path.join(DATA_STORE, "cknna_data_exp1v2_tiny_L"),
}

OUT_DIR = os.path.join(DATA_STORE, "cknna_data_exp1v2_tiny_ALL")
MIN_HMAX = 15
MIN_SEQ_COLS = MIN_HMAX + 1  # 16 columns: [t, t+1, ..., t+15]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_feats_B = []
    all_feats_B_seq = []
    all_actions = []
    all_actions_seq = []
    all_task_descriptions = []
    all_episode_lengths = []
    all_episode_indices = []
    all_frame_indices = []

    for group_name in ["S", "M", "L"]:
        d = GROUP_DIRS[group_name]
        print(f"\nLoading Group {group_name} from {d}")

        with open(os.path.join(d, "metadata.json")) as f:
            meta = json.load(f)

        n = meta["num_samples"]
        hmax = meta["max_horizon"]
        print(f"  N={n}, max_horizon={hmax}")

        all_task_descriptions.extend(meta["task_descriptions"])
        all_episode_lengths.extend(meta["episode_lengths"])
        all_episode_indices.extend(meta["episode_indices"])
        all_frame_indices.extend(meta["frame_indices"])

        fb = torch.load(os.path.join(d, "feats_B.pt"), weights_only=True)
        all_feats_B.append(fb)
        print(f"  feats_B: {tuple(fb.shape)}")

        fb_seq = torch.load(os.path.join(d, "feats_B_seq.pt"), weights_only=True)
        fb_seq_trunc = fb_seq[:, :MIN_SEQ_COLS, :]
        all_feats_B_seq.append(fb_seq_trunc)
        print(f"  feats_B_seq: {tuple(fb_seq.shape)} -> truncated to {tuple(fb_seq_trunc.shape)}")

        ac = torch.load(os.path.join(d, "actions.pt"), weights_only=True)
        all_actions.append(ac)

        ac_seq = torch.load(os.path.join(d, "actions_seq.pt"), weights_only=True)
        ac_seq_trunc = ac_seq[:, :MIN_SEQ_COLS, :]
        all_actions_seq.append(ac_seq_trunc)

    merged_feats_B = torch.cat(all_feats_B, dim=0)
    merged_feats_B_seq = torch.cat(all_feats_B_seq, dim=0)
    merged_actions = torch.cat(all_actions, dim=0)
    merged_actions_seq = torch.cat(all_actions_seq, dim=0)

    N_total = merged_feats_B.shape[0]
    print(f"\nMerged N={N_total}")
    print(f"  feats_B: {tuple(merged_feats_B.shape)}")
    print(f"  feats_B_seq: {tuple(merged_feats_B_seq.shape)}")
    print(f"  actions: {tuple(merged_actions.shape)}")
    print(f"  actions_seq: {tuple(merged_actions_seq.shape)}")

    torch.save(merged_feats_B, os.path.join(OUT_DIR, "feats_B.pt"))
    torch.save(merged_feats_B_seq, os.path.join(OUT_DIR, "feats_B_seq.pt"))
    torch.save(merged_actions, os.path.join(OUT_DIR, "actions.pt"))
    torch.save(merged_actions_seq, os.path.join(OUT_DIR, "actions_seq.pt"))

    merged_meta = {
        "num_samples": N_total,
        "max_horizon": MIN_HMAX,
        "task_descriptions": all_task_descriptions,
        "episode_lengths": all_episode_lengths,
        "episode_indices": all_episode_indices,
        "frame_indices": all_frame_indices,
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(merged_meta, f, indent=2)
    print(f"  metadata.json: max_horizon={MIN_HMAX}")

    # Copy and concatenate images (re-index 0..N_total-1)
    img_dir = os.path.join(OUT_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)
    idx = 0
    for group_name in ["S", "M", "L"]:
        src_img = os.path.join(GROUP_DIRS[group_name], "images")
        with open(os.path.join(GROUP_DIRS[group_name], "metadata.json")) as f:
            n_g = json.load(f)["num_samples"]
        print(f"  Copying {n_g} images from Group {group_name}...")
        for i in range(n_g):
            src = os.path.join(src_img, f"{i:06d}.png")
            dst = os.path.join(img_dir, f"{idx:06d}.png")
            if os.path.exists(src):
                shutil.copy2(src, dst)
            idx += 1
    print(f"  Total images: {idx}")

    # Merge per-model feature files
    model_dirs_S = [
        x for x in os.listdir(GROUP_DIRS["S"])
        if os.path.isdir(os.path.join(GROUP_DIRS["S"], x)) and x != "images"
    ]
    print(f"\nMerging features for {len(model_dirs_S)} models...")

    for model_name in sorted(model_dirs_S):
        out_model = os.path.join(OUT_DIR, model_name)
        os.makedirs(out_model, exist_ok=True)

        feat_files = ["feats_A.pt", "feats_A_img.pt", "feats_A_txt.pt", "feats_action.pt"]
        for fname in feat_files:
            parts = []
            for group_name in ["S", "M", "L"]:
                p = os.path.join(GROUP_DIRS[group_name], model_name, fname)
                if os.path.exists(p):
                    parts.append(torch.load(p, weights_only=True))

            if len(parts) == 3:
                merged = torch.cat(parts, dim=0)
                torch.save(merged, os.path.join(out_model, fname))
            elif len(parts) == 0:
                pass
            else:
                print(f"  WARNING: {model_name}/{fname} found in {len(parts)}/3 groups, skipping")

        # Check what was saved
        saved = [f for f in feat_files if os.path.exists(os.path.join(out_model, f))]
        print(f"  {model_name}: {saved}")

    print(f"\nMerged dataset saved to: {OUT_DIR}")
    print(f"Next: run DTW CKNNA on the merged data:")
    print(f"  python record/scripts/run_dtw_cknna.py \\")
    print(f"      --data_dir {OUT_DIR} \\")
    print(f"      --dtw_version sym2_cuda_nowin \\")
    print(f"      --horizons 1 3 7 15 \\")
    print(f"      --device cuda")


if __name__ == "__main__":
    main()
