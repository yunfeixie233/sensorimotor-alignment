"""test_wiring.py -- Verify shell scripts reference Python files that exist."""

import os
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")


def _extract_python_calls(sh_path):
    """Extract Python script paths from shell script lines like:
       python "${PROJECT_ROOT}/extractors/foo/bar.py"
    """
    refs = []
    with open(sh_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            matches = re.findall(
                r'python[3]?\s+"?\$\{PROJECT_ROOT\}/([^"]+\.py)"?', line
            )
            for m in matches:
                refs.append(m)
    return refs


def test_wiring():
    sh_files = [
        f for f in os.listdir(SCRIPTS_DIR)
        if f.endswith(".sh") and os.path.isfile(os.path.join(SCRIPTS_DIR, f))
    ]

    missing = []
    total_refs = 0

    for sh in sorted(sh_files):
        sh_path = os.path.join(SCRIPTS_DIR, sh)
        refs = _extract_python_calls(sh_path)
        for ref in refs:
            total_refs += 1
            full = os.path.join(PROJECT_ROOT, ref)
            exists = os.path.isfile(full)
            if not exists:
                missing.append("%s -> %s" % (sh, ref))
                print("  MISSING  %-40s (in %s)" % (ref, sh))

    print("\nChecked %d Python references in %d shell scripts." % (total_refs, len(sh_files)))

    if missing:
        print("MISSING files:")
        for m in missing:
            print("  %s" % m)
    else:
        print("All references resolved.")

    assert not missing, "%d broken references" % len(missing)


if __name__ == "__main__":
    test_wiring()
