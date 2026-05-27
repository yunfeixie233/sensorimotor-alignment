#!/usr/bin/env bash
# Sidecar: watches a run's ckpt dir; on each new DeepSpeed stage-2 save,
# converts to FP32 single-file, rsyncs to archive on /workspace,
# then prunes old raw DS dirs keeping only the newest N for fault-tolerant resume.
#
# Usage: ckpt_convert_sidecar.sh <ckpt_dir> <archive_dir> [keep_ds=2] [log_path]

# NOTE: intentionally no `set -e` — ls/grep with no matches return nonzero and
# would otherwise kill this long-running watcher. We handle errors manually.
set -uo pipefail

CKPT_DIR="${1:?ckpt dir required}"
ARCHIVE_DIR="${2:?archive dir required}"
KEEP_DS="${3:-2}"
LOG="${4:-/dev/shm/vlm4vla/tmp/ckpt_convert_sidecar.log}"

mkdir -p "$ARCHIVE_DIR" "$(dirname "$LOG")"
echo "[$(date -Is)] sidecar starting, watching=$CKPT_DIR archive=$ARCHIVE_DIR keep=$KEEP_DS" | tee -a "$LOG"

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate vlm4vla
cd /workspace/VLM4VLA

declare -A DONE

while true; do
    if [ -d "$CKPT_DIR" ]; then
        # enumerate *.ckpt dirs; avoid glob literal when no match
        shopt -s nullglob
        ckpt_dirs=("$CKPT_DIR"/*.ckpt)
        shopt -u nullglob

        for d in "${ckpt_dirs[@]}"; do
            [ -d "$d" ] || continue
            name=$(basename "$d")
            # skip the symlink-like "last.ckpt" alias
            [ "$name" = "last.ckpt" ] && continue

            fp32_target="$CKPT_DIR/${name%.ckpt}.fp32.pt"
            if [ -f "$fp32_target" ]; then
                DONE[$name]=done
                continue
            fi
            [ "${DONE[$name]:-}" = "done" ] && continue

            # wait for deepspeed to finish writing all 8 rank shards
            # (match both legacy zero_pp_rank_* and bf16 prefixed variants)
            shards=0
            if [ -d "$d/checkpoint" ]; then
                shards=$(find "$d/checkpoint" -maxdepth 1 -name '*zero_pp_rank_*_optim_states.pt' 2>/dev/null | wc -l)
            fi
            if [ "${shards:-0}" -lt 8 ]; then
                continue
            fi

            # stability pause
            sleep 5

            echo "[$(date -Is)] convert START $d -> $fp32_target" | tee -a "$LOG"
            if python scripts/convert_ckpt_standalone.py "$d" "$fp32_target" 2>&1 | tee -a "$LOG"; then
                if [ -f "$fp32_target" ]; then
                    size=$(du -sh "$fp32_target" | awk '{print $1}')
                    echo "[$(date -Is)] convert OK, fp32 size=$size" | tee -a "$LOG"
                    # PRUNE BEFORE RSYNC: overlay is too small to hold two FP32 files at once.
                    # Brief moment with no archive copy, but unavoidable here.
                    to_prune=$(ls -t "$ARCHIVE_DIR"/*.fp32.pt 2>/dev/null)
                    if [ -n "$to_prune" ]; then
                        echo "[$(date -Is)] archive prune (pre-rsync): $to_prune" | tee -a "$LOG"
                        echo "$to_prune" | xargs -r rm -f
                    fi
                    rsync -a "$fp32_target" "$ARCHIVE_DIR/" 2>&1 | tee -a "$LOG"
                    DONE[$name]=done
                fi
            else
                echo "[$(date -Is)] convert FAILED for $d" | tee -a "$LOG"
                DONE[$name]=failed
            fi

            # rotate: keep only the KEEP_DS newest raw DS dirs
            shopt -s nullglob
            all_dirs=("$CKPT_DIR"/*.ckpt)
            shopt -u nullglob
            # sort by mtime desc, drop last.ckpt
            mapfile -t all_dirs_sorted < <(for x in "${all_dirs[@]}"; do
                    [ "$(basename "$x")" = "last.ckpt" ] && continue
                    printf '%d\t%s\n' "$(stat -c %Y "$x")" "$x"
                done | sort -nr | cut -f2-)
            if [ ${#all_dirs_sorted[@]} -gt $KEEP_DS ]; then
                for old in "${all_dirs_sorted[@]:$KEEP_DS}"; do
                    if [ -d "$old" ]; then
                        echo "[$(date -Is)] prune old DS dir: $old" | tee -a "$LOG"
                        rm -rf "$old"
                    fi
                done
            fi
        done
    fi
    sleep 30
done
