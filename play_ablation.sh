#!/bin/bash
# Watch a trained ablation in the MuJoCo viewer.
# Usage: ./play_ablation.sh <baseline|heads1|heads2|heads8|stateonly|visual_pool> [checkpoint] [episodes]
# Pick a training seed other than 0 with SEED=1 ./play_ablation.sh baseline
set -e

NAME="${1:?Usage: play_ablation.sh <baseline|heads1|heads2|heads8|stateonly|visual_pool> [checkpoint] [episodes]}"
CKPT="${2:-best}"
EPISODES="${3:-20}"
SEED="${SEED:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$ROOT_DIR/log_ablation/ablation_$NAME/UnitreeMujocoGymEnv/$SEED"

if [ ! -f "$RUN_DIR/params.json" ]; then
    echo "No such ablation/seed: $NAME seed=$SEED (looked in $RUN_DIR)" >&2
    exit 1
fi

# stateonly is the MLP policy; the rest are transformers
EXTRA=()
if [ "$NAME" = "stateonly" ]; then EXTRA+=(--use_mlp); fi

export MUJOCO_GL="${MUJOCO_GL:-glfw}"

cd "$ROOT_DIR"
python3 scripts/play.py \
    --config "$RUN_DIR/params.json" \
    --log_dir "$RUN_DIR" \
    --checkpoint "$CKPT" \
    --episodes "$EPISODES" \
    --seed 0 \
    --render \
    "${EXTRA[@]}" \
    "${@:4}"
