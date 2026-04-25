export MUJOCO_GL=egl
python3 scripts/reward_sweep.py --log-dir ./sweep-logs --seed 0 --vec-env-nums 4 --proc-nums 4 --configs-dir configs/reward_experiments/
