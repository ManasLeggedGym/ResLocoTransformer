export MUJOCO_GL=egl

# Fresh training run (overwrites any existing log for this config/seed):
python3 scripts/train.py --config configs/go2_natural_gait_tf.json --overwrite --vec_env_nums 16 --proc_nums 16

# Resume training from the latest checkpoint (uncomment to use):
# python3 scripts/train.py --config configs/go2_attnres_mujoco.json --resume

# Resume from a specific checkpoint epoch (uncomment to use):
# python3 scripts/train.py --config configs/go2_attnres_mujoco.json --resume --checkpoint 1000

# Architecture ablations, 3 arms x 3 seeds (uncomment to use):
# python3 scripts/run_ablations.py --seeds 0 1 2 --vec-env-nums 16 --proc-nums 16
# python3 scripts/run_ablations.py --seeds 0 1 2 --resume
# python3 scripts/run_ablations.py --filter mlp --seeds 0
# python3 scripts/compare_experiments.py --log-dir ./log_ablation --no-plot
