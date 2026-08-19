#!/bin/bash
# Adds the Visual Pool arm to the ablation matrix (report.tex E002):
#   ablation_visual_pool -- encoder tokens mean-pooled, transformer_params: [],
#   no AttnRes layers (288k params) -- vs the already-trained ablation_baseline
#   (2x [4,128] AttnRes layers over the same tokens, 422k params).
# Isolates "attention over visual tokens" from "visual tokens at all", which the
# stateonly arm cannot separate.
#
# Same format as run_remaining_seeds.sh: one arm at a time on GPU 0, detached so
# it survives closing the terminal. ~10 h per seed, ~30 h for all three.
#
# IMPORTANT: run_remaining_seeds.sh must have finished first -- a second
# concurrent arm at --vec_env_nums 16 OOMs the 8 GB RTX 3070.
#
# Launch:  ./run_visualpool_seeds.sh
# Watch:   tail -f log_ablation/visualpool.out
# Stop:    kill -- -$(cat log_ablation/visualpool.pgid)
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PY=/home/manas/miniforge3/envs/resloco/bin/python3
LOG_DIR="$ROOT_DIR/log_ablation"
OUT="$LOG_DIR/visualpool.out"

# GPU 1/2 are GTX 1080s (sm_61) and torch 2.11+cu130 ships no sm_61 kernels,
# so this job is single-GPU by necessity.
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl

# "<arm> <seed> <fresh|resume>"
QUEUE=(
    "ablation_visual_pool 0 fresh"
    "ablation_visual_pool 1 fresh"
    "ablation_visual_pool 2 fresh"
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
echo $! > "$LOG_DIR/visualpool.pgid"
echo "launched pid $! (own process group) -> $OUT"
