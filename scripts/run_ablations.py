import argparse
import csv
import glob
import json
import os.path as osp
import subprocess
import sys
import time

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
REPO_ROOT  = osp.dirname(SCRIPT_DIR)

ENV_NAME = "UnitreeMujocoGymEnv"
BASELINE = "ablation_baseline"

PRIMARY_METRIC = "Running_Average_Rewards"
GAIT_COLUMNS = [
    ("reward/foot_slip",      "slip"),
    ("reward/foot_clearance", "clear"),
    ("reward/feet_air_time",  "air"),
    ("diag/base_height",      "height"),
]

TODO_GROUPS = [
    ("Baseline training session (Tier 2/3)",
     [BASELINE]),
    ("Different `attn_res_heads` in LocoAttnResTransformer",
     ["ablation_heads1", "ablation_heads2", BASELINE, "ablation_heads8"]),
    ("Effect of the depth camera (state-only policy)",
     [BASELINE, "ablation_stateonly"]),
]

ALLOWED_DIFFS = {
    "ablation_heads1":   {"net.attn_res_heads"},
    "ablation_heads2":   {"net.attn_res_heads"},
    "ablation_heads8":   {"net.attn_res_heads"},
    "ablation_stateonly": {
        "env.env_build.get_image", "policy_type",
        "net.attn_res_heads", "net.transformer_params",
        "net.token_norm", "net.max_pool", "net.hidden_shapes",
        "encoder.hidden_shapes", "encoder.token_dim", "encoder.two_by_two",
    },
}

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

def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, prefix + k + "."))
        else:
            out[prefix + k] = v
    return out

def check_interpreter():
    probe = ("import sys; sys.path.insert(0, %r); import gymnasium, mujoco, "
             "torch, envs, torchrl" % REPO_ROOT)
    r = subprocess.run([sys.executable, "-c", probe],
                       capture_output=True, text=True, cwd=REPO_ROOT)
    if r.returncode != 0:
        last = (r.stderr.strip().splitlines() or ["<no output>"])[-1]
        print(f"[ERROR] {sys.executable} cannot import the training stack:\n"
              f"  {last}\n"
              f"Run this script with the project environment, e.g.\n"
              f"  conda run -n resloco python3 scripts/run_ablations.py ...")
        sys.exit(1)

def check_arms(configs, configs_dir):
    base_path = osp.join(configs_dir, BASELINE + ".json")
    if not osp.exists(base_path):
        print(f"[warn] {base_path} missing — skipping one-variable check")
        return

    with open(base_path) as f:
        base = flatten(json.load(f))

    problems = []
    for cfg in configs:
        exp_id = osp.splitext(osp.basename(cfg))[0]
        if exp_id == BASELINE:
            continue
        with open(cfg) as f:
            arm = flatten(json.load(f))
        diffs = {k for k in set(base) | set(arm)
                 if base.get(k, "<absent>") != arm.get(k, "<absent>")}
        unexpected = diffs - ALLOWED_DIFFS.get(exp_id, set())
        if unexpected:
            problems.append((exp_id, sorted(unexpected)))

    if problems:
        print("\n[ERROR] arms differ from the baseline on unintended keys:")
        for exp_id, keys in problems:
            for k in keys:
                print(f"  {exp_id}: {k} "
                      f"(baseline={base.get(k, '<absent>')!r})")
        print("Fix the configs or extend ALLOWED_DIFFS before trusting a comparison.")
        sys.exit(1)
    print("One-variable check: OK — every arm differs from the baseline only "
          "on its intended keys.")

def arm_label(exp_id, configs_dir):
    path = osp.join(configs_dir, exp_id + ".json")
    if not osp.exists(path):
        return ""
    with open(path) as f:
        cfg = json.load(f)
    if cfg.get("policy_type") == "mlp":
        return "mlp, no camera"
    heads = cfg.get("net", {}).get("attn_res_heads")
    return f"heads={heads}" if heads is not None else ""

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

def tail_mean(series, frac=10):
    """Mean over the final 1/frac of a series. Single eval points on this env
    swing by more than the between-arm effect we are trying to measure, so the
    tail mean -- not Final or Peak -- is the comparison column."""
    if not series:
        return None
    tail = series[-max(1, len(series) // frac):]
    return sum(tail) / len(tail)

def arm_rows(exp_id, args, status):
    rows, tails = [], []
    for seed in args.seeds:
        series = load_column(args.log_dir, exp_id, seed, PRIMARY_METRIC)
        final = series[-1] if series else None
        peak  = max(series) if series else None
        tail  = tail_mean(series)
        if tail is not None:
            tails.append(tail)

        label = f"{exp_id} ({arm_label(exp_id, args.configs_dir)})"
        row = (f"{label:<40} {seed:>5} "
               f"{(f'{final:.4f}' if final is not None else 'n/a'):>10} "
               f"{(f'{peak:.4f}' if peak is not None else 'n/a'):>10} "
               f"{(f'{tail:.4f}' if tail is not None else 'n/a'):>10}")
        for col, _ in GAIT_COLUMNS:
            s = load_column(args.log_dir, exp_id, seed, col)
            tail = s[-max(1, len(s) // 10):] if s else []
            row += f" {(f'{sum(tail)/len(tail):.3f}' if tail else 'n/a'):>9}"
        ok = status.get((exp_id, seed))
        row += f" {('ok' if ok else 'FAIL') if ok is not None else 'skip':>8}"
        rows.append(row)
    return rows, tails

def summarize(status, ran, args):
    width = 120

    def available(exp_id):
        return exp_id in ran or any(
            load_column(args.log_dir, exp_id, s, PRIMARY_METRIC)
            for s in args.seeds)

    for title, arms in TODO_GROUPS:
        arms = [a for a in arms if available(a)]
        if not arms:
            continue

        print(f"\n{'='*width}\n  TODO: {title}\n"
              f"  primary={PRIMARY_METRIC}\n{'='*width}")
        hdr = f"{'Arm':<40} {'Seed':>5} {'Final':>10} {'Peak':>10} {'Tail':>10}"
        for _, short in GAIT_COLUMNS:
            hdr += f" {short:>9}"
        hdr += f" {'Status':>8}"
        print(hdr)
        print("-" * width)

        means = {}
        for exp_id in arms:
            rows, tails = arm_rows(exp_id, args, status)
            for r in rows:
                print(r)
            if tails:
                means[exp_id] = (sum(tails) / len(tails),
                                 max(tails) - min(tails) if len(tails) > 1 else 0.0,
                                 len(tails))

        if means:
            print("-" * width)
            ref = means.get(BASELINE, (None,))[0]
            print(f"{'Arm':<40} {'Seeds':>5} {'Mean tail':>12} {'Spread':>10} "
                  f"{'vs baseline':>13}")
            for exp_id, (mean, spread, n) in means.items():
                delta = ("--" if ref is None or exp_id == BASELINE
                         else f"{mean - ref:+.4f}")
                print(f"{exp_id:<40} {n:>5} {mean:>12.4f} {spread:>10.4f} "
                      f"{delta:>13}")
            if any(n < 2 for _, _, n in means.values()):
                print("\n  [warn] single seed per arm. Between-seed spread on this "
                      "env is unmeasured, and eval swings by ~1000 reward between\n"
                      "         adjacent points, so a between-arm delta below that "
                      "is not evidence. Prefer >=3 seeds before concluding.")

    print(f"\n{'='*width}")
    print("\nReward is not the verdict — replay the best checkpoint of each arm before "
          "concluding:\n  python3 scripts/play.py --config <log_dir>/<arm>/"
          f"{ENV_NAME}/<seed>/params.json \\\n"
          "      --log_dir <log_dir>/<arm>/" + ENV_NAME + "/<seed> --checkpoint best --render")
    print(f"\nTensorBoard: tensorboard --logdir {args.log_dir}")

def main():
    args = parse_args()
    configs = sorted(glob.glob(osp.join(args.configs_dir, "ablation_*.json")))
    if not configs:
        print(f"No ablation_*.json configs found in {args.configs_dir}")
        sys.exit(1)

    check_interpreter()
    check_arms(configs, args.configs_dir)

    if args.filter:
        configs = [c for c in configs if args.filter in osp.basename(c)]
        if not configs:
            print(f"No ablation configs matching '{args.filter}'")
            sys.exit(1)

    print(f"\nRunning {len(configs)} arm(s) x {len(args.seeds)} seed(s):")
    for c in configs:
        exp_id = osp.splitext(osp.basename(c))[0]
        print(f"  {exp_id:<24} {arm_label(exp_id, args.configs_dir)}")

    status, ran = {}, []
    for cfg in configs:
        exp_id = osp.splitext(osp.basename(cfg))[0]
        ran.append(exp_id)
        for seed in args.seeds:
            status[(exp_id, seed)] = run_one(cfg, exp_id, seed, args)

    if not args.dry_run:
        summarize(status, ran, args)

if __name__ == "__main__":
    main()
