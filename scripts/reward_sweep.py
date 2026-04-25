"""
scripts/reward_sweep.py — Run all reward experiments sequentially and compare results.

Usage:
    python3 scripts/reward_sweep.py
    python3 scripts/reward_sweep.py --log-dir ./log --seed 0 --vec-env-nums 4 --proc-nums 4
    python3 scripts/reward_sweep.py --configs-dir configs/reward_experiments --dry-run
"""

import argparse
import csv
import glob
import os
import os.path as osp
import subprocess
import sys
import time

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
REPO_ROOT = osp.dirname(SCRIPT_DIR)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--configs-dir", default=osp.join(REPO_ROOT, "configs", "reward_experiments"))
    p.add_argument("--log-dir", default=osp.join(REPO_ROOT, "log"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vec-env-nums", type=int, default=4)
    p.add_argument("--proc-nums", type=int, default=4)
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    return p.parse_args()


def read_csv_metric(csv_path, column):
    """Return (final_value, peak_value) for *column* from log.csv, or (None, None)."""
    if not osp.exists(csv_path):
        return None, None
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows or column not in rows[0]:
            return None, None
        vals = [float(r[column]) for r in rows if r.get(column, "").strip()]
        if not vals:
            return None, None
        return vals[-1], max(vals)
    except Exception as e:
        print(f"  [warn] Could not read {csv_path}: {e}")
        return None, None


def find_csv(log_dir, experiment_id, env_name, seed):
    return osp.join(log_dir, experiment_id, env_name, str(seed), "log.csv")


def run_experiment(config_path, experiment_id, args):
    cmd = [
        sys.executable,
        osp.join(SCRIPT_DIR, "train.py"),
        "--config", config_path,
        "--id", experiment_id,
        "--log_dir", args.log_dir,
        "--seed", str(args.seed),
        "--vec_env_nums", str(args.vec_env_nums),
        "--proc_nums", str(args.proc_nums),
        "--overwrite",
    ]
    if args.no_cuda:
        cmd.append("--no_cuda")

    print(f"\n{'='*60}")
    print(f"  Experiment: {experiment_id}")
    print(f"  Config:     {osp.relpath(config_path, REPO_ROOT)}")
    print(f"  Command:    {' '.join(cmd)}")
    print(f"{'='*60}")

    if args.dry_run:
        print("  [dry-run] Skipping execution.")
        return True

    start = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = time.time() - start
    success = result.returncode == 0
    status = "DONE" if success else f"FAILED (exit {result.returncode})"
    print(f"\n  {status} in {elapsed/60:.1f} min")
    return success


def print_summary(results):
    print(f"\n{'='*70}")
    print("  SWEEP SUMMARY")
    print(f"{'='*70}")
    header = f"{'Experiment':<35} {'Final Reward':>13} {'Peak Reward':>12} {'Status':>8}"
    print(header)
    print("-" * 70)
    for exp_id, final, peak, ok in results:
        final_s = f"{final:.4f}" if final is not None else "n/a"
        peak_s  = f"{peak:.4f}"  if peak  is not None else "n/a"
        status  = "ok" if ok else "FAILED"
        print(f"{exp_id:<35} {final_s:>13} {peak_s:>12} {status:>8}")
    print(f"{'='*70}")

    # Rank by peak reward
    ranked = [(e, p) for e, _, p, ok in results if p is not None and ok]
    ranked.sort(key=lambda x: x[1], reverse=True)
    if ranked:
        print("\n  Ranking by peak Running_Average_Rewards:")
        for i, (exp_id, peak) in enumerate(ranked, 1):
            print(f"    {i}. {exp_id:<35} {peak:.4f}")


def main():
    args = parse_args()

    configs = sorted(glob.glob(osp.join(args.configs_dir, "*.json")))
    if not configs:
        print(f"No configs found in {args.configs_dir}")
        sys.exit(1)

    print(f"Found {len(configs)} experiment configs in {osp.relpath(args.configs_dir, REPO_ROOT)}")
    for c in configs:
        print(f"  {osp.basename(c)}")

    results = []
    env_name = "UnitreeMujocoGymEnv"

    for config_path in configs:
        experiment_id = osp.splitext(osp.basename(config_path))[0]
        ok = run_experiment(config_path, experiment_id, args)
        csv_path = find_csv(args.log_dir, experiment_id, env_name, args.seed)
        final, peak = read_csv_metric(csv_path, "Running_Average_Rewards")
        results.append((experiment_id, final, peak, ok))

        # Print one-line summary after each run
        final_s = f"{final:.4f}" if final is not None else "n/a"
        peak_s  = f"{peak:.4f}"  if peak  is not None else "n/a"
        print(f"  Summary → final={final_s}  peak={peak_s}")

    print_summary(results)
    print(f"\nTensorBoard: tensorboard --logdir {args.log_dir}")
    print(f"Plots:       python3 scripts/compare_experiments.py --log-dir {args.log_dir}")


if __name__ == "__main__":
    main()
