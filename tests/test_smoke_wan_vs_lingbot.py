"""test_smoke_wan_vs_lingbot.py -- Smoke test for 0324_wan_vs_lingbot experiment.

Verifies WAN 2.2 raw transformer feature extraction:
1. Loads WanTransformer3DModel from diffusers
2. Processes N=10 samples from DROID
3. Produces correct output shape

Requires: diffusers, torch, PIL, repos/lingbot-va in PYTHONPATH
Model: Wan-AI/Wan2.2-TI2V-5B-Diffusers (transformer subfolder)

Usage:
    python tests/test_smoke_wan_vs_lingbot.py [--transformer_path /path/to/transformer]
"""

import argparse
import json
import os
import subprocess
import sys
import time

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_DROID_DIR = "/lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid"
DEFAULT_TRANSFORMER_PATH = os.path.join(
    os.environ.get("DATA_STORE", "/lambda/nfs/vla/cache"),
    "Wan2.2-TI2V-5B-Diffusers", "transformer"
)


def test_wan_import():
    """Test that diffusers WanTransformer3DModel can be imported."""
    print("  Checking diffusers import...", end="", flush=True)
    try:
        from diffusers import WanTransformer3DModel
        print(" OK")
        return True
    except ImportError as e:
        print(f" FAIL — {e}")
        return False


def test_wan_extraction(data_dir, transformer_path):
    """Run WAN raw extraction on N=10 samples."""
    extractor = os.path.join(PROJECT_ROOT, "extractors", "wan_raw", "extract_features.py")
    if not os.path.isfile(extractor):
        print(f"  SKIP: {extractor} not found")
        return False

    if not os.path.isdir(transformer_path):
        print(f"  SKIP: transformer not found at {transformer_path}")
        print(f"    Download with: hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers")
        return False

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        env = os.environ.copy()
        lingbot_path = os.path.join(PROJECT_ROOT, "repos", "lingbot-va")
        env["PYTHONPATH"] = lingbot_path + ":" + env.get("PYTHONPATH", "")

        cmd = [
            sys.executable, "-u", extractor,
            "--transformer_path", transformer_path,
            "--data_dir", data_dir,
            "--output_dir", tmpdir,
            "--max_samples", "10",
        ]
        print(f"  Running WAN extraction (N=10)...", end="", flush=True)
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)

        if result.returncode != 0:
            print(f" FAIL ({time.time()-t0:.1f}s)")
            print(f"    stderr: {result.stderr[-500:]}")
            return False

        feats_path = os.path.join(tmpdir, "feats_A.pt")
        if not os.path.exists(feats_path):
            print(f" FAIL — feats_A.pt not created")
            return False

        t = torch.load(feats_path, map_location="cpu")
        if t.shape[0] != 10:
            print(f" FAIL — shape {tuple(t.shape)}, expected (10, ...)")
            return False

        print(f" OK — shape {tuple(t.shape)} ({time.time()-t0:.1f}s)")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DEFAULT_DROID_DIR)
    parser.add_argument("--transformer_path", default=DEFAULT_TRANSFORMER_PATH)
    args = parser.parse_args()

    print(f"Data: {args.data_dir}")
    print(f"Transformer: {args.transformer_path}\n")

    results = {}

    results["wan_import"] = test_wan_import()
    results["wan_extraction"] = test_wan_extraction(args.data_dir, args.transformer_path)

    print(f"\n{'='*50}")
    passed = sum(1 for v in results.values() if v)
    print(f"Results: {passed}/{len(results)} passed")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL/SKIP'}  {name}")

    if passed < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
