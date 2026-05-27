#!/usr/bin/env bash
# Quick-check that each created conda env can `import torch` and report
# torch.cuda.is_available().
#
# CONDA_ROOT resolution order:
#   1. command-line arg: ./verify_envs.sh /my/conda
#   2. $CONDA_ROOT env var (typically set by sourcing ../paths.env)
#   3. auto-detect from `which conda`
#   4. fallback to $HOME/miniconda3
set -u

CONDA_ROOT="${1:-${CONDA_ROOT:-}}"
if [[ -z "$CONDA_ROOT" ]]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_ROOT="$(dirname "$(dirname "$(command -v conda)")")"
    else
        CONDA_ROOT="$HOME/miniconda3"
    fi
fi

if [[ ! -d "$CONDA_ROOT/envs" ]]; then
    echo "CONDA_ROOT=$CONDA_ROOT/envs does not exist."
    echo "Usage: $0 [path-to-conda-root]   (or export CONDA_ROOT first)"
    exit 1
fi
echo "Using CONDA_ROOT=$CONDA_ROOT"
echo

ENVS=(
    starVLA            # Tiers 1/2/3 — required by all paper figures
    cogact             # CogACT + OpenVLA + raw Prismatic
    groot_libero       # GR00T-N1.5 / N1.6
    pi0fast_env        # Pi0-Bridge
    spatialvla_env     # SpatialVLA-Bridge
    openvla_env        # OpenVLA (alt)
    simpler_env        # RT-1-X, Octo (TF/JAX)
    motus              # Motus WAM + RoboTwin
    lingbot_va         # LingBot-VA WAM + RoboTwin
    vidar              # Vidar WAM + RoboTwin
    robotwin           # SAPIEN simulation
)

ok=0
miss=0
fail=0
for e in "${ENVS[@]}"; do
    PY="$CONDA_ROOT/envs/$e/bin/python"
    if [[ ! -x "$PY" ]]; then
        printf "  MISSING %-18s (env not installed)\n" "$e"
        miss=$((miss+1))
        continue
    fi
    OUT=$("$PY" - <<'PY' 2>&1
import sys
try:
    import torch
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
except Exception as exc:
    print(f"FAIL {type(exc).__name__}: {exc}")
    sys.exit(1)
PY
)
    if [[ $? -eq 0 ]]; then
        printf "  OK      %-18s %s\n" "$e" "$OUT"
        ok=$((ok+1))
    else
        printf "  FAIL    %-18s %s\n" "$e" "$OUT"
        fail=$((fail+1))
    fi
done

echo "----"
echo "Result: $ok OK, $miss missing, $fail failed (of ${#ENVS[@]} total)."
[[ $fail -eq 0 ]]
