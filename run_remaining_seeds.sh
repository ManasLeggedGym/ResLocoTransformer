#!/bin/bash
# Finishes the 5-arm x 3-seed ablation matrix.
#
# State as of 2026-08-18:
#   baseline   seeds 0,1,2  COMPLETE  -> nothing to do
#   stateonly  seed  0      complete
#   heads1     seed  0      complete;  seed 1 stopped at epoch 250/500
#   heads2     seed  0      complete
#   heads8     seed  0      complete
#
# Priority (as requested): baseline arm first (already done), then stateonly,
# then the head-count arms. Within the head arms we go breadth-first on seed 1
# across all three before seed 2, so a crash leaves the matrix balanced.
#
# Launch:  ./run_remaining_seeds.sh
# Watch:   tail -f log_ablation/remaining.out
# Stop:    kill -- -$(cat log_ablation/remaining.pgid)
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PY=/home/manas/miniforge3/envs/resloco/bin/python3
LOG_DIR="$ROOT_DIR/log_ablation"
OUT="$LOG_DIR/remaining.out"

# CUDA_VISIBLE_DEVICES uses FASTEST_FIRST ordering by default, so 0 is the
# RTX 3070 (sm_86). The two GTX 1080s are sm_61 and torch 2.11+cu130 ships no
# sm_61 kernels, so this job is single-GPU by necessity.
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl

# "<arm> <seed> <fresh|resume>"
QUEUE=(
    "ablation_stateonly 1 fresh"
    "ablation_stateonly 2 fresh"
    "ablation_heads1    1 resume"
    "ablation_heads2    1 fresh"
    "ablation_heads8    1 fresh"
    "ablation_heads1    2 fresh"
    "ablation_heads2    2 fresh"
    "ablation_heads8    2 fresh"
)

run_queue() {
    local t_all
    t_all=$(date +%s)
    for entry in "${QUEUE[@]}"; do
        read -r arm seed mode <<<"$entry"
        local t0 rc flag
        t0=$(date +%s)
        flag="--overwrite"
        [ "$mode" = "resume" ] && flag="--resume"

        echo ""
        echo "============================================================"
        echo "  $arm  seed=$seed  ($mode)  started $(date '+%F %T')"
        echo "============================================================"

        "$PY" scripts/train.py \
            --config "$ROOT_DIR/configs/$arm.json" \
            --id "$arm" \
            --log_dir "$LOG_DIR" \
            --seed "$seed" \
            --vec_env_nums 16 \
            --proc_nums 16 \
            "$flag"
        rc=$?

        echo "  $arm seed=$seed -> $([ $rc -eq 0 ] && echo DONE || echo "FAILED rc=$rc")"\
             "in $(( ($(date +%s) - t0) / 60 )) min"
    done
    echo ""
    echo "ALL RUNS FINISHED after $(( ($(date +%s) - t_all) / 3600 )) h at $(date '+%F %T')"
}

if [ "${_ABLATION_CHILD:-0}" = "1" ]; then
    run_queue
    exit 0
fi

mkdir -p "$LOG_DIR"
_ABLATION_CHILD=1 setsid nohup "$0" >"$OUT" 2>&1 </dev/null &
echo $! > "$LOG_DIR/remaining.pgid"
echo "launched pid $! (own process group) -> $OUT"
