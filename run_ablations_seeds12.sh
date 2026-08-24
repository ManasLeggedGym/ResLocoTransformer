#!/bin/bash
# Completes the 5 arms x 3 seeds ablation matrix: seed 0 is already done,
# this runs seeds 1 and 2 for all five arms (10 arm-seeds, ~60 h).
#
# Launched detached via setsid+nohup so it survives closing the terminal:
#   ./run_ablations_seeds12.sh
# Watch it with:
#   tail -f log_ablation/seeds12.out
# Stop it with:
#   kill -- -$(cat log_ablation/seeds12.pgid)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PY=/home/manas/miniforge3/envs/resloco/bin/python3
OUT="$ROOT_DIR/log_ablation/seeds12.out"

# GPU 1 is a GTX 1080 (sm_61); torch 2.11+cu130 has no sm_61 kernels, so pin GPU 0.
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl

setsid nohup "$PY" scripts/run_ablations.py \
    --seeds 1 2 \
    --vec-env-nums 16 \
    --proc-nums 16 \
    >"$OUT" 2>&1 < /dev/null &

echo $! > "$ROOT_DIR/log_ablation/seeds12.pgid"
echo "launched pid $! (own process group), logging to $OUT"
