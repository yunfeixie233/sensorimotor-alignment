"""test_imports.py -- Verify all Python scripts can be imported without errors.

Only tests basic module-level import (no model loading).
Some extractors need specific conda envs; those are skipped if dependencies
are missing and reported as SKIP rather than FAIL.
"""

import importlib.util
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _try_import(label, path):
    """Attempt to load a Python file as a module. Returns (ok, msg)."""
    abs_path = os.path.join(PROJECT_ROOT, path)
    if not os.path.isfile(abs_path):
        return False, "FILE NOT FOUND"
    spec = importlib.util.spec_from_file_location(label, abs_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return True, "OK"


def test_imports():
    scripts = {
        "config": "config.py",
        "load_bridge_rlds": "data/load_bridge_rlds.py",
        "load_droid_rlds": "data/load_droid_rlds.py",
        "make_tiny_subset": "data/make_tiny_subset.py",
        "merge_tiny": "data/merge_exp1v2_tiny_groups.py",
        "merge_full": "data/merge_exp1v2_full_groups.py",
        "compute_cknna": "compute/compute_cknna.py",
        "compute_dtw": "compute/compute_dtw.py",
        "compute_dtw_aloha": "compute/compute_dtw_aloha.py",
        "compute_nodtw": "compute/compute_nodtw.py",
    }

    passed = 0
    skipped = 0
    failed = 0
    failures = []

    for label, path in sorted(scripts.items()):
        ok, msg = _try_import(label, path)
        if ok:
            print("  PASS  %-30s %s" % (label, msg))
            passed += 1
        else:
            print("  FAIL  %-30s %s" % (label, msg))
            failed += 1
            failures.append(label)

    print("\n%d passed, %d skipped, %d failed" % (passed, skipped, failed))
    assert not failures, "Failed imports: %s" % ", ".join(failures)


if __name__ == "__main__":
    test_imports()
