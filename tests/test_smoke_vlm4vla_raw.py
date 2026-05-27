"""test_smoke_vlm4vla_raw.py -- Smoke test for 0325_vlm4vla_raw experiment.

Verifies that each extractor (L1+L2 ViT-levels and L3 raw) can:
1. Load the model
2. Process N=10 samples from Bridge or DROID
3. Produce output tensors with correct shapes

Requires: transformers, torch, PIL, scipy
Models: Qwen2.5-VL-3B, Qwen3-VL-2B, PaliGemma-1, KosMos-2 (smallest per family)

Usage:
    python tests/test_smoke_vlm4vla_raw.py [--data_dir /path/to/cknna_data] [--device cuda]
"""

import argparse
import json
import os
import sys
import time

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# Add qwen-vl-utils to path
QWEN_VL_UTILS = os.path.join(PROJECT_ROOT, "Qwen3-VL", "qwen-vl-utils", "src")
if os.path.isdir(QWEN_VL_UTILS):
    sys.path.insert(0, QWEN_VL_UTILS)

DEFAULT_DATA_DIRS = [
    "/lambda/nfs/vla/cache/cknna_data_store/cknna_data_exp1v2_full_L",
    "/lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid",
]


def find_data_dir():
    for d in DEFAULT_DATA_DIRS:
        if os.path.isfile(os.path.join(d, "metadata.json")):
            return d
    return None


def run_extractor(label, cmd_args, expected_files, data_dir, n=10):
    """Run an extractor as subprocess with --max_samples=N and check outputs."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        full_cmd = [sys.executable, "-u"] + cmd_args + [
            "--data_dir", data_dir,
            "--output_dir", tmpdir,
            "--device", "cuda" if torch.cuda.is_available() else "cpu",
        ]
        # Add --max_samples if the extractor supports it
        if "--max_samples" not in " ".join(cmd_args):
            full_cmd += ["--max_samples", str(n)]

        print(f"  Running: {label} (N={n})...", end="", flush=True)
        t0 = time.time()
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f" FAIL ({time.time()-t0:.1f}s)")
            print(f"    stderr: {result.stderr[-500:]}")
            return False

        # Check output files exist and have correct N
        for fname in expected_files:
            fpath = os.path.join(tmpdir, fname)
            if not os.path.exists(fpath):
                print(f" FAIL — missing {fname}")
                return False
            t = torch.load(fpath, map_location="cpu")
            if t.shape[0] != n:
                print(f" FAIL — {fname} has {t.shape[0]} rows, expected {n}")
                return False

        print(f" OK ({time.time()-t0:.1f}s)")
        return True


def test_l12_qwen25(data_dir):
    """Test L1+L2 extraction for Qwen2.5-VL-3B (smallest Qwen2.5)."""
    return run_extractor(
        "L1+L2 Qwen2.5VL-3B",
        [os.path.join(PROJECT_ROOT, "extractors", "extract_vit_levels_qwen.py"),
         "--model_path", "Qwen/Qwen2.5-VL-3B-Instruct",
         "--model_family", "qwen2.5"],
        ["feats_A_vit.pt", "feats_A_pre_llm.pt"],
        data_dir,
    )


def test_l12_qwen3(data_dir):
    """Test L1+L2 extraction for Qwen3-VL-2B (smallest Qwen3)."""
    return run_extractor(
        "L1+L2 Qwen3VL-2B",
        [os.path.join(PROJECT_ROOT, "extractors", "extract_vit_levels_qwen.py"),
         "--model_path", "Qwen/Qwen3-VL-2B-Instruct",
         "--model_family", "qwen3"],
        ["feats_A_vit.pt", "feats_A_pre_llm.pt"],
        data_dir,
    )


def test_l12_paligemma(data_dir):
    """Test L1+L2 extraction for PaliGemma-1 (gated — needs HF token)."""
    return run_extractor(
        "L1+L2 PaliGemma-1",
        [os.path.join(PROJECT_ROOT, "extractors", "extract_vit_levels_paligemma.py"),
         "--model_path", "google/paligemma-3b-pt-224"],
        ["feats_A_vit.pt", "feats_A_pre_llm.pt"],
        data_dir,
    )


def test_l12_kosmos2(data_dir):
    """Test L1+L2 extraction for KosMos-2."""
    return run_extractor(
        "L1+L2 KosMos-2",
        [os.path.join(PROJECT_ROOT, "extractors", "extract_vit_levels_kosmos2.py"),
         "--model_path", "microsoft/kosmos-2-patch14-224"],
        ["feats_A_vit.pt", "feats_A_pre_llm.pt"],
        data_dir,
    )


def test_l3_qwen25(data_dir):
    """Test L3 extraction for Qwen2.5-VL-3B."""
    return run_extractor(
        "L3 Qwen2.5VL-3B",
        [os.path.join(PROJECT_ROOT, "extractors", "starvla", "extract_raw_qwen.py"),
         "--model_path", "Qwen/Qwen2.5-VL-3B-Instruct",
         "--model_family", "qwen2.5"],
        ["feats_A.pt"],
        data_dir,
    )


def test_l3_kosmos2(data_dir):
    """Test L3 extraction for KosMos-2."""
    return run_extractor(
        "L3 KosMos-2",
        [os.path.join(PROJECT_ROOT, "extractors", "kosmos2", "extract_raw_kosmos2.py"),
         "--model_path", "microsoft/kosmos-2-patch14-224"],
        ["feats_A.pt"],
        data_dir,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    data_dir = args.data_dir or find_data_dir()
    if data_dir is None:
        print("SKIP: No CKNNA data directory found")
        return

    print(f"Data: {data_dir}")
    print(f"Device: {args.device}\n")

    tests = [
        ("L1+L2 Qwen2.5VL-3B", test_l12_qwen25),
        ("L1+L2 Qwen3VL-2B", test_l12_qwen3),
        ("L1+L2 PaliGemma-1", test_l12_paligemma),
        ("L1+L2 KosMos-2", test_l12_kosmos2),
        ("L3 Qwen2.5VL-3B", test_l3_qwen25),
        ("L3 KosMos-2", test_l3_kosmos2),
    ]

    results = {}
    for name, fn in tests:
        try:
            results[name] = fn(data_dir)
        except Exception as e:
            print(f"  {name}: ERROR — {e}")
            results[name] = False

    print(f"\n{'='*50}")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
