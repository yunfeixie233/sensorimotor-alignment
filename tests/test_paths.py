"""test_paths.py -- Verify all paths in config resolve to existing locations."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def test_all_paths():
    local_repo_checks = [
        ("STARVLA_ROOT (local)", config.STARVLA_ROOT),
        ("MOTUS_ROOT (local)", config.MOTUS_ROOT),
        ("XVLA_ROOT (local)", config.XVLA_ROOT),
        ("LINGBOT_VA_ROOT (local)", config.LINGBOT_VA_ROOT),
        ("SOFTDTW_DIR (local)", config.SOFTDTW_DIR),
        ("INTERNVLA_ROOT (local)", config.INTERNVLA_ROOT),
        ("ISAAC_GROOT_DIR (local)", config.ISAAC_GROOT_DIR),
        ("ISAAC_GROOT_N16_DIR (local)", config.ISAAC_GROOT_N16_DIR),
        ("COGACT_ROOT (local)", config.COGACT_ROOT),
        ("OCTO_ROOT (local)", config.OCTO_ROOT),
    ]

    external_checks = [
        ("WORK", config.WORK),
        ("CONDA_ROOT", config.CONDA_ROOT),
        ("BRIDGE_RLDS_DIR", config.BRIDGE_RLDS_DIR),
        ("DROID_RLDS_DIR", config.DROID_RLDS_DIR),
        ("DATA_STORE", config.DATA_STORE),
        ("RECORD_DIR", config.RECORD_DIR),
        ("META_CSV", config.META_CSV),
    ]

    # Optional checkpoints: not required at Setup time. The GR00T-N1.5/N1.6
    # and StarVLA checkpoints are downloaded inline by the VLA-SR Feature
    # extraction loop (`hf download` to checkpoints/<HF_ID>/); the OpenVLA /
    # RT-1X / Prismatic checkpoints are not used by any paper experiment.
    # Missing -> warn, do not fail.
    optional_checks = [
        ("CKPT_STARVLA_DIR", config.CKPT_STARVLA_DIR),
        ("CKPT_GROOT_N15", config.CKPT_GROOT_N15),
        ("CKPT_GROOT_N16", config.CKPT_GROOT_N16),
        ("CKPT_OPENVLA_7B", config.CKPT_OPENVLA_7B),
        ("CKPT_OPENVLA_FT", config.CKPT_OPENVLA_FT),
        ("CKPT_RT1X", config.CKPT_RT1X),
        ("CKPT_PRISMATIC_RAW", config.CKPT_PRISMATIC_RAW),
    ]

    print("=== Local repo paths (bundled code) ===")
    failed = []
    for name, path in local_repo_checks:
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print("  %-30s %s  [%s]" % (name, path, status))
        if not exists:
            failed.append(name)

    assert not failed, "Missing LOCAL repo paths: %s" % ", ".join(failed)

    print("\n=== External paths (checkpoints, datasets, data store) ===")
    ext_failed = []
    for name, path in external_checks:
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print("  %-30s %s  [%s]" % (name, path, status))
        if not exists:
            ext_failed.append(name)

    assert not ext_failed, "Missing external paths: %s" % ", ".join(ext_failed)

    print("\n=== Optional checkpoints (not used by paper experiments) ===")
    opt_missing = []
    for name, path in optional_checks:
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING (optional)"
        print("  %-30s %s  [%s]" % (name, path, status))
        if not exists:
            opt_missing.append(name)
    if opt_missing:
        print("  WARN: %s missing -- harmless, not needed to reproduce the paper."
              % ", ".join(opt_missing))

    total = len(local_repo_checks) + len(external_checks) + len(optional_checks)
    print("\nAll %d required paths verified (%d optional missing)."
          % (len(local_repo_checks) + len(external_checks), len(opt_missing)))


if __name__ == "__main__":
    test_all_paths()
