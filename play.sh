export MUJOCO_GL=glfw
python3 scripts/play.py \
    --config configs/reward_experiments/01_action_rate.json \
    --checkpoint best \
    --episodes 10 \
    --log_dir ./sweep-logs \
    --render
