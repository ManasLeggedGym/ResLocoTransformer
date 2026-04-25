export MUJOCO_GL=egl

# Fresh training run (overwrites any existing log for this config/seed):
python3 scripts/train.py --config configs/reward_experiments/00_baseline.json --overwrite

# Resume training from the latest checkpoint (uncomment to use):
# python3 scripts/train.py --config configs/go2_attnres_mujoco.json --resume

# Resume from a specific checkpoint epoch (uncomment to use):
# python3 scripts/train.py --config configs/go2_attnres_mujoco.json --resume --checkpoint 1000