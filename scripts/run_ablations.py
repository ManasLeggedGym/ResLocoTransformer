"""
scripts/run_ablations.py — Run the architecture ablations (MLP / visual pool /
AttnRes) across seeds and print a per-arm summary.

Usage:
    python3 scripts/run_ablations.py                       # 3 arms x seeds 0,1,2
    python3 scripts/run_ablations.py --seeds 0             # single seed
    python3 scripts/run_ablations.py --filter attnres      # one arm only
    python3 scripts/run_ablations.py --resume              # continue from checkpoints
    python3 scripts/run_ablations.py --dry-run
"""

import argparse
import csv
import glob
import os.path as osp
import subprocess
import sys
import time

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
REPO_ROOT  = osp.dirname(SCRIPT_DIR)

ENV_NAME = "UnitreeMujocoGymEnv"

PRIMARY_METRIC = "Running_Average_Rewards"
GAIT_COLUMNS = [
    ("reward/foot_slip",      "slip"),
    ("reward/foot_clearance", "clear"),
    ("reward/feet_air_time",  "air"),
    ("diag/base_height",      "height"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--configs-dir",  default=osp.join(REPO_ROOT, "configs"))
    p.add_argument("--log-dir",      default=osp.join(REPO_ROOT, "log_ablation"))
    p.add_argument("--seeds",        type=int, nargs="+", default=[0, 1, 2],
                   help="Seeds to run per arm (default: 0 1 2)")
    p.add_argument("--vec-env-nums", type=int, default=16)
    p.add_argument("--proc-nums",    type=int, default=16)
    p.add_argument("--no-cuda",      action="store_true")
    p.add_argument("--filter",       default="",
                   help="Only run ablation configs whose filename contains this string")
    p.add_argument("--resume",       action="store_true",
                   help="Resume each run from its latest checkpoint instead of overwriting")
    p.add_argument("--dry-run",      action="store_true")
    return p.parse_args()


def load_column(log_dir, exp_id, seed, column):
    csv_path = osp.join(log_dir, exp_id, ENV_NAME, str(seed), "log.csv")
    if not osp.exists(csv_path):
        return []
    try:
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        return [float(r[column]) for r in rows if r.get(column, "").strip()]
    except Exception as e:
        print(f"  [warn] {csv_path}: {e}")
        return []


def run_one(config_path, exp_id, seed, args):
    cmd = [
        sys.executable, osp.join(SCRIPT_DIR, "train.py"),
        "--config", config_path,
        "--id", exp_id,
        "--log_dir", args.log_dir,
        "--seed", str(seed),
        "--vec_env_nums", str(args.vec_env_nums),
        "--proc_nums", str(args.proc_nums),
        "--resume" if args.resume else "--overwrite",
    ]
    if args.no_cuda:
        cmd.append("--no_cuda")

    print(f"\n{'='*60}\n  {exp_id}  seed={seed}\n{'='*60}")
    if args.dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return True

    t0 = time.time()
    ok = subprocess.run(cmd, cwd=REPO_ROOT).returncode == 0
    print(f"  {'DONE' if ok else 'FAILED'} in {(time.time()-t0)/60:.1f} min")
    return ok


def summarize(results, args):
    print(f"\n{'='*104}\n  ARCHITECTURE ABLATION — primary={PRIMARY_METRIC}\n{'='*104}")
    hdr = f"{'Arm':<24} {'Seed':>5} {'Final':>10} {'Peak':>10}"
    for _, short in GAIT_COLUMNS:
        hdr += f" {short:>9}"
    hdr += f" {'Status':>8}"
    print(hdr)
    print("-" * 104)

    finals = {}
    for exp_id, seed, ok in results:
        series = load_column(args.log_dir, exp_id, seed, PRIMARY_METRIC)
        final = series[-1] if series else None
        peak  = max(series) if series else None
        if final is not None:
            finals.setdefault(exp_id, []).append(final)

        row = (f"{exp_id:<24} {seed:>5} "
               f"{(f'{final:.4f}' if final is not None else 'n/a'):>10} "
               f"{(f'{peak:.4f}' if peak is not None else 'n/a'):>10}")
        for col, _ in GAIT_COLUMNS:
            s = load_column(args.log_dir, exp_id, seed, col)
            tail = s[-max(1, len(s) // 10):] if s else []
            row += f" {(f'{sum(tail)/len(tail):.3f}' if tail else 'n/a'):>9}"
        row += f" {'ok' if ok else 'FAIL':>8}"
        print(row)

    print("-" * 104)
    print(f"{'Arm':<24} {'Seeds':>5} {'Mean final':>12} {'Spread':>10}")
    for exp_id, vals in finals.items():
        spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
        print(f"{exp_id:<24} {len(vals):>5} {sum(vals)/len(vals):>12.4f} {spread:>10.4f}")
    print(f"{'='*104}")
    print("\nReward is not the verdict — replay the best checkpoint of each arm before "
          "concluding:\n  python3 scripts/play.py --config <log_dir>/<arm>/"
          f"{ENV_NAME}/<seed>/params.json \\\n"
          "      --log_dir <log_dir>/<arm>/" + ENV_NAME + "/<seed> --checkpoint best --render")
    print(f"\nTensorBoard: tensorboard --logdir {args.log_dir}")


def main():
    args = parse_args()
    configs = sorted(glob.glob(osp.join(args.configs_dir, "ablation_*.json")))
    if args.filter:
        configs = [c for c in configs if args.filter in osp.basename(c)]
    if not configs:
        print(f"No ablation_*.json configs found in {args.configs_dir}" +
              (f" matching '{args.filter}'" if args.filter else ""))
        sys.exit(1)

    print(f"Running {len(configs)} arm(s) x {len(args.seeds)} seed(s):")
    for c in configs:
        print(f"  {osp.basename(c)}")

    results = []
    for cfg in configs:
        exp_id = osp.splitext(osp.basename(cfg))[0]
        for seed in args.seeds:
            results.append((exp_id, seed, run_one(cfg, exp_id, seed, args)))

    if not args.dry_run:
        summarize(results, args)


if __name__ == "__main__":
    main()
