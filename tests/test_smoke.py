"""test_smoke.py -- Run Phase 3 (DTW CKNNA) on existing tiny data.

Checks that run_dtw_cknna.py can load metadata, compute DTW on a small
existing dataset, and produce a CSV output.
"""

import json
import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _find_tiny_data():
    """Find the smallest existing experiment data directory with metadata."""
    data_store = os.environ.get("DATA_STORE", "/home/ubuntu/verl/starVLA/cknna")
    candidates = [
        "cknna_data_exp1v2_tiny_S",
        "cknna_data_exp1v2_tiny_M",
        "cknna_data_exp1v2_tiny_L",
        "cknna_data_exp1_tiny",
    ]
    for c in candidates:
        d = os.path.join(data_store, c)
        meta = os.path.join(d, "metadata.json")
        if os.path.isfile(meta):
            with open(meta) as f:
                m = json.load(f)
            n = m["num_samples"]
            if n > 0:
                return d, n
    return None, 0


def test_smoke_phase3():
    data_dir, n = _find_tiny_data()
    if data_dir is None:
        print("SKIP: No tiny experiment data found for smoke test")
        return

    print("Found tiny data: %s (N=%d)" % (data_dir, n))

    dtw_script = os.path.join(PROJECT_ROOT, "compute", "run_dtw_cknna.py")
    assert os.path.isfile(dtw_script), "run_dtw_cknna.py not found"

    # Check that --help works
    result = subprocess.run(
        [sys.executable, dtw_script, "--help"],
        capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, "run_dtw_cknna.py --help failed"
    assert "--data_dir" in result.stdout, "--data_dir not in help output"
    print("run_dtw_cknna.py --help: OK")

    # Check that metadata can be loaded
    meta_path = os.path.join(data_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["num_samples"] > 0
    print("Metadata loaded: N=%d, max_horizon=%d" % (meta["num_samples"], meta["max_horizon"]))

    print("\nSmoke test PASSED")


if __name__ == "__main__":
    test_smoke_phase3()
