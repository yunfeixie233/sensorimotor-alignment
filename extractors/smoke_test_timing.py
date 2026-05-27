"""Smoke test: compare 2-feature vs 7-feature extraction timing on N=10 samples.

Also verifies that the 7-feature script produces outputs matching existing extracted features.
"""
import json
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../Qwen3-VL/qwen-vl-utils/src/"))

DATA_DIR = "/lambda/nfs/vla/cache/cknna_data_store/cknna_data_exp1v2_full_L"
N = 10


def load_model_qwen25(model_path, device):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, attn_implementation="sdpa", torch_dtype="auto"
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"
    return model, processor


def time_2feat(model, processor, samples, device):
    """Existing 2-feature extraction (P1-V + P2-V only)."""
    from extract_vit_levels_qwen import extract_vit_levels, build_inputs_qwen25

    feats_vit, feats_pre = [], []
    t0 = time.time()
    for img, instruction in samples:
        batch = build_inputs_qwen25(processor, img, instruction)
        f_vit, f_pre = extract_vit_levels(model, batch, device)
        feats_vit.append(f_vit)
        feats_pre.append(f_pre)
    elapsed = time.time() - t0
    return elapsed, feats_vit, feats_pre


def time_7feat(model, processor, samples, device):
    """New 7-feature extraction."""
    from extract_all_features import (
        extract_all_7, get_hook_targets, build_inputs_qwen25, build_task_mask
    )

    p1_mod, p2_mod, layer0, _, get_image_mask = get_hook_targets(model, "qwen2.5")
    tokenizer = processor.tokenizer
    all_feats = {k: [] for k in ["p1_v", "p2_v", "p2_t", "p2_vt", "p3_v", "p3_t", "p3_vt"]}

    t0 = time.time()
    for img, instruction in samples:
        batch = build_inputs_qwen25(processor, img, instruction)
        feats = extract_all_7(
            model, batch, tokenizer, instruction, device, "qwen2.5",
            p1_mod, p2_mod, layer0, get_image_mask,
        )
        for k in all_feats:
            all_feats[k].append(feats[k])
    elapsed = time.time() - t0
    return elapsed, all_feats


def main():
    device = "cuda"
    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"

    with open(os.path.join(DATA_DIR, "metadata.json")) as f:
        metadata = json.load(f)
    images_dir = os.path.join(DATA_DIR, "images")

    # Load samples
    samples = []
    for i in range(N):
        img = Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
        samples.append((img, metadata["task_descriptions"][i]))

    print(f"Loading model: {model_path}")
    model, processor = load_model_qwen25(model_path, device)

    # Warm-up (1 forward pass)
    print("Warm-up...")
    from extract_all_features import (
        extract_all_7, get_hook_targets, build_inputs_qwen25
    )
    batch = build_inputs_qwen25(processor, samples[0][0], samples[0][1])
    p1_mod, p2_mod, layer0, _, get_image_mask = get_hook_targets(model, "qwen2.5")
    extract_all_7(model, batch, processor.tokenizer, samples[0][1], device, "qwen2.5",
                  p1_mod, p2_mod, layer0, get_image_mask)

    # --- Time 2-feature ---
    print(f"\n=== 2-feature extraction (N={N}) ===")
    t2, vit_2, pre_2 = time_2feat(model, processor, samples, device)
    print(f"  Time: {t2:.2f}s ({t2/N*1000:.0f}ms/sample)")

    # --- Time 7-feature ---
    print(f"\n=== 7-feature extraction (N={N}) ===")
    t7, feats_7 = time_7feat(model, processor, samples, device)
    print(f"  Time: {t7:.2f}s ({t7/N*1000:.0f}ms/sample)")

    # --- Timing comparison ---
    overhead = (t7 - t2) / t2 * 100
    print(f"\n=== Comparison ===")
    print(f"  2-feature: {t2:.2f}s")
    print(f"  7-feature: {t7:.2f}s")
    print(f"  Overhead:  {overhead:+.1f}%")
    print(f"  Extra time per sample: {(t7-t2)/N*1000:.1f}ms")

    # --- Verify P1-V and P2-V match between old and new ---
    print(f"\n=== Verification: old vs new features ===")
    for i in range(N):
        # P1-V
        diff_p1 = (vit_2[i] - feats_7["p1_v"][i]).abs().max().item()
        # P2-V
        diff_p2 = (pre_2[i] - feats_7["p2_v"][i]).abs().max().item()
        if i < 3 or diff_p1 > 1e-4 or diff_p2 > 1e-4:
            print(f"  Sample {i}: P1-V diff={diff_p1:.2e}, P2-V diff={diff_p2:.2e}")

    max_p1 = max((vit_2[i] - feats_7["p1_v"][i]).abs().max().item() for i in range(N))
    max_p2 = max((pre_2[i] - feats_7["p2_v"][i]).abs().max().item() for i in range(N))
    print(f"  Max P1-V diff across {N} samples: {max_p1:.2e}")
    print(f"  Max P2-V diff across {N} samples: {max_p2:.2e}")

    # --- Verify against existing saved features ---
    existing_dir = os.path.join(DATA_DIR, "qwen25vl-3b-raw")
    print(f"\n=== Verification: new vs saved features ({existing_dir}) ===")
    checks = [
        ("feats_A_vit.pt", "p1_v"),
        ("feats_A_pre_llm.pt", "p2_v"),
        ("feats_A.pt", "p3_vt"),
        ("feats_A_img.pt", "p3_v"),
        ("feats_A_txt.pt", "p3_t"),
    ]
    for fname, key in checks:
        fpath = os.path.join(existing_dir, fname)
        if os.path.exists(fpath):
            saved = torch.load(fpath, map_location="cpu", weights_only=True)
            diffs = [(saved[i] - feats_7[key][i]).abs().max().item() for i in range(N)]
            print(f"  {fname:<28} vs {key:<6}: max_diff={max(diffs):.2e}, mean_diff={sum(diffs)/N:.2e}")
        else:
            print(f"  {fname:<28}: NOT FOUND")

    # --- Show shapes of all 7 features ---
    print(f"\n=== Feature shapes (sample 0) ===")
    for k in ["p1_v", "p2_v", "p2_t", "p2_vt", "p3_v", "p3_t", "p3_vt"]:
        print(f"  {k:<6}: {tuple(feats_7[k][0].shape)}")


if __name__ == "__main__":
    main()
