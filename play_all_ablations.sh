#!/bin/bash
# Play every trained run under log_ablation/ in the MuJoCo viewer, one after another.
#
# Usage:
#   ./play_all_ablations.sh                       # all 6 arms x seeds 0,1,2, 20 episodes each
#   ./play_all_ablations.sh -e 5                  # 5 episodes per run
#   ./play_all_ablations.sh -a baseline,heads8    # only these arms
#   ./play_all_ablations.sh -s 0                  # only seed 0
#   ./play_all_ablations.sh -c 1000               # a specific epoch checkpoint instead of best
#   ./play_all_ablations.sh -n                    # headless (no viewer), just print returns
#
# Between runs it waits for Enter unless -y / NONSTOP=1 is given.
set -u

ARMS="baseline,heads1,heads2,heads8,stateonly,visual_pool"
SEEDS="0,1,2"
EPISODES=20
CKPT="best"
RENDER=1
PAUSE="${NONSTOP:+0}"; PAUSE="${PAUSE:-1}"

while getopts "a:s:e:c:nyh" opt; do
    case "$opt" in
        a) ARMS="$OPTARG" ;;
        s) SEEDS="$OPTARG" ;;
        e) EPISODES="$OPTARG" ;;
        c) CKPT="$OPTARG" ;;
        n) RENDER=0 ;;
        y) PAUSE=0 ;;
        h) sed -n '2,15p' "$0"; exit 0 ;;
        *) exit 1 ;;
    esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"

# Interpreter: $PYTHON wins; otherwise plain python3 if it has the deps,
# else the conda env the runs were trained in.
PYBIN="${PYTHON:-}"
if [ -z "$PYBIN" ]; then
    if python3 -c "import gymnasium" >/dev/null 2>&1; then
        PYBIN=python3
    elif [ -x "$HOME/miniforge3/envs/resloco/bin/python" ]; then
        PYBIN="$HOME/miniforge3/envs/resloco/bin/python"
    else
        echo "No python with gymnasium found; set PYTHON=/path/to/python" >&2; exit 1
    fi
fi
echo "python: $PYBIN"

IFS=',' read -r -a ARM_LIST  <<< "$ARMS"
IFS=',' read -r -a SEED_LIST <<< "$SEEDS"

declare -a OK=() SKIPPED=() FAILED=()

for NAME in "${ARM_LIST[@]}"; do
    for SEED in "${SEED_LIST[@]}"; do
        RUN_DIR="$ROOT_DIR/log_ablation/ablation_$NAME/UnitreeMujocoGymEnv/$SEED"
        TAG="$NAME/seed$SEED"

        if [ ! -f "$RUN_DIR/params.json" ] || [ ! -f "$RUN_DIR/model/model_pf_$CKPT.pth" ]; then
            echo "[skip] $TAG -- no params.json or model_pf_$CKPT.pth in $RUN_DIR"
            SKIPPED+=("$TAG"); continue
        fi

        # stateonly is the MLP policy; the rest are transformers
        EXTRA=()
        [ "$NAME" = "stateonly" ] && EXTRA+=(--use_mlp)
        [ "$RENDER" = "1" ]      && EXTRA+=(--render)

        echo
        echo "=============================================================="
        echo "  $TAG  (checkpoint=$CKPT, episodes=$EPISODES)"
        echo "=============================================================="
        if "$PYBIN" scripts/play.py \
                --config "$RUN_DIR/params.json" \
                --log_dir "$RUN_DIR" \
                --checkpoint "$CKPT" \
                --episodes "$EPISODES" \
                --seed 0 \
                "${EXTRA[@]}"; then
            OK+=("$TAG")
        else
            echo "[fail] $TAG exited $?" >&2
            FAILED+=("$TAG")
        fi

        if [ "$PAUSE" = "1" ] && [ "$RENDER" = "1" ]; then
            read -r -p "Enter for next run (Ctrl-C to stop)... " _ </dev/tty || exit 0
        fi
    done
done

echo
echo "=== summary ==="
echo "played:  ${#OK[@]}   ${OK[*]:-}"
echo "skipped: ${#SKIPPED[@]}   ${SKIPPED[*]:-}"
echo "failed:  ${#FAILED[@]}   ${FAILED[*]:-}"
[ "${#FAILED[@]}" -eq 0 ]
