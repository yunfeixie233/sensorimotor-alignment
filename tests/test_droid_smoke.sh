#!/bin/bash
# =============================================================================
# test_droid_smoke.sh
# =============================================================================
# Smoke test for all DROID Phase 2 extractors on a 5-sample subset.
# Runs each model sequentially (single GPU), checks output shapes and values.
#
# Usage (from cknna_project root):
#   bash tests/test_droid_smoke.sh [--gpu 0] [--n 5] [--skip-slow]
#
# --gpu N       : GPU index to use (default: 0)
# --n N         : number of samples in smoke dataset (default: 5)
# --skip-slow   : skip models that take >3 min to load (OpenVLA, SpatialVLA, Prismatic)
# =============================================================================
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${PROJECT_ROOT}/paths.env"

# ---- Defaults ----
GPU=0
N_SAMPLES=5
SKIP_SLOW=0
SMOKE_DIR="/tmp/cknna_droid_smoke"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)    GPU="$2";       shift 2 ;;
        --n)      N_SAMPLES="$2"; shift 2 ;;
        --skip-slow) SKIP_SLOW=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="${GPU}"
CONDA="${CONDA_ROOT}"
EXT="${PROJECT_ROOT}/extractors"
LOG_DIR="${PROJECT_ROOT}/logs/droid_smoke"
mkdir -p "${LOG_DIR}"

PASS=0; FAIL=0; SKIP=0
declare -a RESULTS=()
RESULT_FILE="/tmp/droid_smoke_results_$$.txt"   # temp file for subshell results
: > "${RESULT_FILE}"

log()   { echo "[$(date '+%H:%M:%S')] $*"; }
pass()  {
    PASS=$((PASS+1))
    RESULTS+=("  PASS  $1")
    echo "PASS $1" >> "${RESULT_FILE}"
    log "PASS  $1"
}
fail()  {
    FAIL=$((FAIL+1))
    RESULTS+=("  FAIL  $1: $2")
    echo "FAIL $1: $2" >> "${RESULT_FILE}"
    log "FAIL  $1: $2"
}
skip_()  {
    SKIP=$((SKIP+1))
    RESULTS+=("  SKIP  $1 (--skip-slow)")
    echo "SKIP $1" >> "${RESULT_FILE}"
    log "SKIP  $1"
}

# Called inside subshells: writes result to temp file, also prints for live log
subshell_pass() { echo "PASS $1" >> "${RESULT_FILE}"; log "PASS  $1"; }
subshell_fail() { echo "FAIL $1: $2" >> "${RESULT_FILE}"; log "FAIL  $1: $2"; }

# Drain results from subshell into parent counters+array
drain_results() {
    while IFS= read -r line; do
        status="${line%% *}"
        rest="${line#* }"
        case "${status}" in
            PASS) PASS=$((PASS+1)); RESULTS+=("  PASS  ${rest}") ;;
            FAIL) FAIL=$((FAIL+1)); RESULTS+=("  FAIL  ${rest}") ;;
        esac
    done < "${RESULT_FILE}"
    : > "${RESULT_FILE}"
}

# ---- Python checker: verify output files and tensor properties ----
check_outputs() {
    local label="$1"; local out_dir="$2"; shift 2
    local files=("$@")
    python3 - "${out_dir}" "${label}" "${files[@]}" << 'PYEOF'
import sys, os, torch, math
out_dir, label = sys.argv[1], sys.argv[2]
files = sys.argv[3:]
errors = []
for fname in files:
    p = os.path.join(out_dir, fname)
    if not os.path.exists(p):
        errors.append(f"{fname} MISSING")
        continue
    t = torch.load(p, weights_only=True)
    if t.numel() == 0:
        errors.append(f"{fname} empty tensor")
        continue
    if not torch.isfinite(t).all():
        errors.append(f"{fname} contains NaN/Inf")
        continue
    if t.abs().max() == 0:
        errors.append(f"{fname} all zeros")
        continue
    print(f"    {fname}: {tuple(t.shape)}  range=[{t.min():.4f},{t.max():.4f}]")
if errors:
    for e in errors:
        print(f"  ERROR: {e}")
    sys.exit(1)
PYEOF
}

# =============================================================================
# Step 0: Create smoke dataset
# =============================================================================
log "========================================"
log "Creating ${N_SAMPLES}-sample smoke dataset"
log "========================================"
python3 "${PROJECT_ROOT}/tests/make_droid_smoke_data.py" \
    --n "${N_SAMPLES}" \
    --dst "${SMOKE_DIR}" 2>&1 | tee "${LOG_DIR}/make_smoke.log"
if [ $? -ne 0 ]; then
    echo "FATAL: Could not create smoke dataset"
    exit 1
fi
log "Smoke data ready at ${SMOKE_DIR}"
echo ""

# =============================================================================
# Helper: run extractor, check outputs, report pass/fail
# =============================================================================
run_model() {
    local label="$1"; local env="$2"; local out_name="$3"
    local out_dir="${SMOKE_DIR}/${out_name}"
    shift 3
    local cmd=("$@")
    local log_file="${LOG_DIR}/${label// /_}.log"

    # files to verify: always feats_A* triplet; feats_action if cmd contains extract_all
    local check_files=("feats_A.pt" "feats_A_img.pt" "feats_A_txt.pt")
    local cmd_str="${cmd[*]}"
    if [[ "${cmd_str}" == *"extract_all"* ]] || [[ "${cmd_str}" == *"extract_action_repr"* ]]; then
        check_files+=("feats_action.pt")
    fi

    mkdir -p "${out_dir}"
    log "--- ${label} ---"

    (
        source "${CONDA}/bin/activate" "${env}" 2>/dev/null
        export PYTHONNOUSERSITE=1
        PYTHONUNBUFFERED=1 "${cmd[@]}"
    ) > "${log_file}" 2>&1

    local rc=$?
    if [ $rc -ne 0 ]; then
        fail "${label}" "exit code ${rc} (see ${log_file})"
        tail -5 "${log_file}" | sed 's/^/    /'
        return
    fi

    if check_outputs "${label}" "${out_dir}" "${check_files[@]}" 2>&1; then
        pass "${label}"
    else
        fail "${label}" "output check failed (see ${log_file})"
    fi
}

run_model_slow() {
    local label="$1"
    if [ "${SKIP_SLOW}" -eq 1 ]; then
        skip_ "${label}"
    else
        shift
        run_model "${label}" "$@"
    fi
}

# =============================================================================
# STREAM 1: StarVLA x5 (starVLA env) -- feats_A + feats_action (two scripts)
# =============================================================================
log "========================================"
log "STREAM 1: StarVLA x5"
log "========================================"

declare -A STARVLA_CKPTS=(
    ["Qwen-GR00T-Bridge"]="${CKPT_STARVLA_DIR}/Qwen-GR00T-Bridge/checkpoints/steps_45000_pytorch_model.pt"
    ["Qwen-GR00T-Bridge-RT-1"]="${CKPT_STARVLA_DIR}/Qwen-GR00T-Bridge-RT-1/checkpoints/steps_30000_pytorch_model.pt"
    ["Qwen-OFT-Bridge-RT-1"]="${CKPT_STARVLA_DIR}/Qwen-OFT-Bridge-RT-1/checkpoints/steps_10000_pytorch_model.pt"
    ["Qwen3VL-GR00T-Bridge-RT-1"]="${CKPT_STARVLA_DIR}/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt"
    ["Qwen3VL-OFT-Bridge-RT-1"]="${CKPT_STARVLA_DIR}/Qwen3VL-OFT-Bridge-RT-1/checkpoints/steps_5000_pytorch_model.pt"
)

for NAME in "Qwen-GR00T-Bridge" "Qwen-GR00T-Bridge-RT-1" "Qwen-OFT-Bridge-RT-1" \
            "Qwen3VL-GR00T-Bridge-RT-1" "Qwen3VL-OFT-Bridge-RT-1"; do
    CKPT="${STARVLA_CKPTS[$NAME]}"
    OUT="${SMOKE_DIR}/${NAME}"
    mkdir -p "${OUT}"
    LOG_FA="${LOG_DIR}/starvla_${NAME}_featsA.log"
    LOG_FA2="${LOG_DIR}/starvla_${NAME}_featsact.log"

    log "--- StarVLA: ${NAME} feats_A ---"
    (
        source "${CONDA}/bin/activate" starVLA 2>/dev/null
        export PYTHONNOUSERSITE=1
        PYTHONUNBUFFERED=1 python "${EXT}/starvla/extract_features.py" \
            --ckpt_path "${CKPT}" \
            --data_dir "${SMOKE_DIR}" \
            --output_dir "${OUT}"
    ) > "${LOG_FA}" 2>&1
    RC_A=$?

    log "--- StarVLA: ${NAME} feats_action ---"
    (
        source "${CONDA}/bin/activate" starVLA 2>/dev/null
        export PYTHONNOUSERSITE=1
        PYTHONUNBUFFERED=1 python "${EXT}/starvla/extract_action_repr.py" \
            --ckpt_path "${CKPT}" \
            --data_dir "${SMOKE_DIR}" \
            --output_dir "${OUT}"
    ) > "${LOG_FA2}" 2>&1
    RC_B=$?

    if [ $RC_A -ne 0 ] || [ $RC_B -ne 0 ]; then
        fail "StarVLA ${NAME}" "feats_A exit=${RC_A} feats_action exit=${RC_B}"
        tail -3 "${LOG_FA}" | sed 's/^/    /'
    elif check_outputs "StarVLA ${NAME}" "${OUT}" feats_A.pt feats_A_img.pt feats_A_txt.pt feats_action.pt 2>&1; then
        pass "StarVLA ${NAME}"
    else
        fail "StarVLA ${NAME}" "output check failed"
    fi
done

# =============================================================================
# STREAM 2: CogACT x3 (cogact env) -- feats_A + feats_action (two scripts)
# =============================================================================
log "========================================"
log "STREAM 2: CogACT x3"
log "========================================"
for SIZE in "Small" "Base" "Large"; do
    size_lower=$(echo "${SIZE}" | tr '[:upper:]' '[:lower:]')
    case "${SIZE}" in Small) DIT="DiT-S" ;; Base) DIT="DiT-B" ;; Large) DIT="DiT-L" ;; esac
    NAME="cogact-${size_lower}-bridge"
    OUT="${SMOKE_DIR}/${NAME}"
    mkdir -p "${OUT}"
    LOG_FA="${LOG_DIR}/cogact_${size_lower}_featsA.log"
    LOG_FA2="${LOG_DIR}/cogact_${size_lower}_featsact.log"

    log "--- CogACT-${SIZE} feats_A ---"
    (
        source "${CONDA}/bin/activate" cogact 2>/dev/null
        export PYTHONNOUSERSITE=1
        PYTHONUNBUFFERED=1 python "${EXT}/cogact/extract_features.py" \
            --ckpt "CogACT/CogACT-${SIZE}" \
            --action_model_type "${DIT}" \
            --data_dir "${SMOKE_DIR}" \
            --output_dir "${OUT}"
    ) > "${LOG_FA}" 2>&1
    RC_A=$?

    log "--- CogACT-${SIZE} feats_action ---"
    (
        source "${CONDA}/bin/activate" cogact 2>/dev/null
        export PYTHONNOUSERSITE=1
        PYTHONUNBUFFERED=1 python "${EXT}/cogact/extract_action_repr.py" \
            --ckpt "CogACT/CogACT-${SIZE}" \
            --action_model_type "${DIT}" \
            --data_dir "${SMOKE_DIR}" \
            --output_dir "${OUT}"
    ) > "${LOG_FA2}" 2>&1
    RC_B=$?

    if [ $RC_A -ne 0 ] || [ $RC_B -ne 0 ]; then
        fail "CogACT-${SIZE}" "feats_A exit=${RC_A} feats_action exit=${RC_B}"
        tail -3 "${LOG_FA}" | sed 's/^/    /'
    elif check_outputs "CogACT-${SIZE}" "${OUT}" feats_A.pt feats_A_img.pt feats_A_txt.pt feats_action.pt 2>&1; then
        pass "CogACT-${SIZE}"
    else
        fail "CogACT-${SIZE}" "output check failed"
    fi
done

# =============================================================================
# STREAM 3: SpatialVLA (spatialvla_env) -- extract_all.py
# =============================================================================
log "========================================"
log "STREAM 3: SpatialVLA"
log "========================================"
run_model_slow "SpatialVLA" spatialvla_env "spatialvla-sft-bridge" \
    python "${EXT}/spatialvla/extract_all.py" \
        --ckpt IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge \
        --data_dir "${SMOKE_DIR}" \
        --output_dir "${SMOKE_DIR}/spatialvla-sft-bridge"

# =============================================================================
# STREAM 4: Raw Qwen x2 (starVLA env) -- feats_A only
# =============================================================================
log "========================================"
log "STREAM 4: Raw Qwen VLMs"
log "========================================"
run_model "Raw Qwen2.5-3B" starVLA "qwen25vl-3b-raw" \
    python "${EXT}/starvla/extract_raw_qwen.py" \
        --model_path "${CKPT_STARVLA_DIR}/Qwen2.5-VL-3B-Instruct" \
        --model_family qwen2.5 \
        --data_dir "${SMOKE_DIR}" \
        --output_dir "${SMOKE_DIR}/qwen25vl-3b-raw"

run_model "Raw Qwen3-4B" starVLA "qwen3vl-4b-raw" \
    python "${EXT}/starvla/extract_raw_qwen.py" \
        --model_path "${CKPT_STARVLA_DIR}/Qwen3-VL-4B-Instruct" \
        --model_family qwen3 \
        --data_dir "${SMOKE_DIR}" \
        --output_dir "${SMOKE_DIR}/qwen3vl-4b-raw"

# =============================================================================
# STREAM 5: OpenVLA x2 (openvla_env) -- extract_all.py
# =============================================================================
log "========================================"
log "STREAM 5: OpenVLA x2"
log "========================================"
run_model_slow "OpenVLA-7B" openvla_env "openvla-7b-bridge" \
    python "${EXT}/openvla/extract_all.py" \
        --ckpt "${CKPT_OPENVLA_7B}" \
        --data_dir "${SMOKE_DIR}" \
        --output_dir "${SMOKE_DIR}/openvla-7b-bridge"

run_model_slow "OpenVLA-7B-FT" openvla_env "openvla-7b-bridge-ft-200k" \
    python "${EXT}/openvla/extract_all.py" \
        --ckpt "${CKPT_OPENVLA_FT}" \
        --data_dir "${SMOKE_DIR}" \
        --output_dir "${SMOKE_DIR}/openvla-7b-bridge-ft-200k"

# =============================================================================
# STREAM 6: Pi0 (pi0fast_env) -- extract_all.py
# =============================================================================
log "========================================"
log "STREAM 6: Pi0"
log "========================================"
run_model "Pi0" pi0fast_env "pi0-lerobot-bridge" \
    python "${EXT}/pi0/extract_all.py" \
        --ckpt_path HaomingSong/lerobot-pi0-bridge \
        --data_dir "${SMOKE_DIR}" \
        --output_dir "${SMOKE_DIR}/pi0-lerobot-bridge"

# =============================================================================
# STREAM 7: GR00T N1.5 + N1.6 (groot_libero) -- extract_all_n15/n16.py
# extract_all_n15 reads feats_B.pt for real state (not dummy zeros).
# =============================================================================
log "========================================"
log "STREAM 7: GR00T N1.5 + N1.6"
log "========================================"
(
    source "${CONDA}/bin/activate" groot_libero 2>/dev/null
    export PYTHONNOUSERSITE=1
    OUT15="${SMOKE_DIR}/groot-n15-bridge"
    mkdir -p "${OUT15}"
    LOG_N15="${LOG_DIR}/groot_n15.log"
    log "--- GR00T N1.5 ---"
    PYTHONPATH="${ISAAC_GROOT_DIR}" PYTHONUNBUFFERED=1 python "${EXT}/groot/extract_all_n15.py" \
        --ckpt "${CKPT_GROOT_N15}" \
        --data_dir "${SMOKE_DIR}" \
        --output_dir "${OUT15}" > "${LOG_N15}" 2>&1
    RC=$?
    if [ $RC -ne 0 ]; then
        subshell_fail "GR00T N1.5" "exit ${RC} (see ${LOG_N15})"
        tail -3 "${LOG_N15}" | sed 's/^/    /'
    elif check_outputs "GR00T N1.5" "${OUT15}" feats_A.pt feats_A_img.pt feats_A_txt.pt feats_action.pt 2>&1; then
        subshell_pass "GR00T N1.5"
    else
        subshell_fail "GR00T N1.5" "output check failed"
    fi

    OUT16="${SMOKE_DIR}/groot-n16-bridge"
    mkdir -p "${OUT16}"
    LOG_N16="${LOG_DIR}/groot_n16.log"
    log "--- GR00T N1.6 ---"
    PYTHONPATH="${ISAAC_GROOT_N16_DIR}" PYTHONUNBUFFERED=1 python "${EXT}/groot/extract_all_n16.py" \
        --ckpt "${CKPT_GROOT_N16}" \
        --data_dir "${SMOKE_DIR}" \
        --output_dir "${OUT16}" > "${LOG_N16}" 2>&1
    RC=$?
    if [ $RC -ne 0 ]; then
        subshell_fail "GR00T N1.6" "exit ${RC} (see ${LOG_N16})"
        tail -3 "${LOG_N16}" | sed 's/^/    /'
    elif check_outputs "GR00T N1.6" "${OUT16}" feats_A.pt feats_A_img.pt feats_A_txt.pt feats_action.pt 2>&1; then
        subshell_pass "GR00T N1.6"
    else
        subshell_fail "GR00T N1.6" "output check failed"
    fi
)
drain_results

# =============================================================================
# STREAM 8: RT-1-X + Octo (simpler_env) -- feats_A only
# =============================================================================
log "========================================"
log "STREAM 8: RT-1-X + Octo"
log "========================================"
(
    source "${CONDA}/bin/activate" simpler_env 2>/dev/null
    export PYTHONNOUSERSITE=1
    export TF_FORCE_GPU_ALLOW_GROWTH=true
    export XLA_PYTHON_CLIENT_PREALLOCATE=false

    RT1X_OUT="${SMOKE_DIR}/rt1x-bridge"; mkdir -p "${RT1X_OUT}"
    LOG_RT="${LOG_DIR}/rt1x.log"
    log "--- RT-1-X ---"
    PYTHONUNBUFFERED=1 python "${EXT}/simpler/extract_rt1x.py" \
        --ckpt "${CKPT_RT1X}" \
        --data_dir "${SMOKE_DIR}" \
        --output_dir "${RT1X_OUT}" > "${LOG_RT}" 2>&1
    RC=$?
    if [ $RC -ne 0 ]; then
        subshell_fail "RT-1-X" "exit ${RC} (see ${LOG_RT})"
        tail -3 "${LOG_RT}" | sed 's/^/    /'
    elif check_outputs "RT-1-X" "${RT1X_OUT}" feats_A.pt 2>&1; then
        subshell_pass "RT-1-X"
    else
        subshell_fail "RT-1-X" "output check failed"
    fi

    OCTO_OUT="${SMOKE_DIR}/octo-base-bridge"; mkdir -p "${OCTO_OUT}"
    LOG_OCT="${LOG_DIR}/octo.log"
    log "--- Octo ---"
    PYTHONUNBUFFERED=1 python "${EXT}/simpler/extract_octo.py" \
        --model_type "hf://rail-berkeley/octo-base" \
        --data_dir "${SMOKE_DIR}" \
        --output_dir "${OCTO_OUT}" > "${LOG_OCT}" 2>&1
    RC=$?
    if [ $RC -ne 0 ]; then
        subshell_fail "Octo" "exit ${RC} (see ${LOG_OCT})"
        tail -3 "${LOG_OCT}" | sed 's/^/    /'
    elif check_outputs "Octo" "${OCTO_OUT}" feats_A.pt 2>&1; then
        subshell_pass "Octo"
    else
        subshell_fail "Octo" "output check failed"
    fi
)
drain_results

# =============================================================================
# STREAM 9: Prismatic raw (cogact env) -- feats_A only
# =============================================================================
log "========================================"
log "STREAM 9: Prismatic raw"
log "========================================"
run_model_slow "Prismatic-raw" cogact "prismatic-raw" \
    python "${EXT}/prismatic/extract_raw.py" \
        --model_path "${CKPT_PRISMATIC_RAW}" \
        --data_dir "${SMOKE_DIR}" \
        --output_dir "${SMOKE_DIR}/prismatic-raw"

# =============================================================================
# Final summary
# =============================================================================
rm -f "${RESULT_FILE}"
TOTAL=$((PASS + FAIL + SKIP))
echo ""
echo "========================================"
echo "DROID Smoke Test Results  (${TOTAL} models)"
echo "========================================"
for r in "${RESULTS[@]}"; do echo "${r}"; done
echo ""
echo "  PASS=${PASS}  FAIL=${FAIL}  SKIP=${SKIP}"
echo ""
if [ "${FAIL}" -gt 0 ]; then
    echo "LOGS: ${LOG_DIR}/"
    echo "FAIL -- ${FAIL} model(s) did not pass"
    exit 1
else
    echo "PASS -- all tested models succeeded"
fi
