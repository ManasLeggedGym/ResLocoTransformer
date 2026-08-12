"""
scripts/compare_experiments.py — Compare reward experiments from CSV logs.

Usage:
    python3 scripts/compare_experiments.py
    python3 scripts/compare_experiments.py --log-dir ./log --metric Running_Average_Rewards
    python3 scripts/compare_experiments.py --filter 01_action --no-plot
"""

import argparse
import csv
import glob
import os
import os.path as osp

import numpy as np

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
REPO_ROOT   = osp.dirname(SCRIPT_DIR)

GAIT_COLUMNS = [
    ("reward/foot_slip",      "slip"),
    ("reward/foot_clearance", "clear"),
    ("reward/feet_air_time",  "air"),
    ("diag/base_height",      "height"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir",     default=osp.join(REPO_ROOT, "log"))
    p.add_argument("--metric",      default="Running_Average_Rewards")
    p.add_argument("--reward-term", default="reward/forward_vel",
                   help="Secondary series to plot (e.g. reward/action_rate)")
    p.add_argument("--filter",      default="", help="Only include experiment IDs containing this string")
    p.add_argument("--no-plot",     action="store_true")
    p.add_argument("--output",      default=osp.join(REPO_ROOT, "reward_comparison.png"))
    return p.parse_args()


def load_series(csv_path, column):
    try:
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        return np.array([float(r[column]) for r in rows
                         if r.get(column, "").strip()])
    except Exception:
        return np.array([])


def converged_mean(series, frac=0.1):
    if not len(series):
        return None
    tail = series[-max(1, int(len(series) * frac)):]
    return float(tail.mean())


def convergence_epoch(series, threshold=0.95):
    if not len(series):
        return None
    idxs = np.where(series >= threshold * series.max())[0]
    return int(idxs[0]) if len(idxs) else None


def find_logs(log_dir, filter_str):
    for path in sorted(glob.glob(osp.join(log_dir, "*", "*", "*", "log.csv"))):
        parts = osp.relpath(path, log_dir).split(os.sep)
        exp_id = parts[0]
        if filter_str and filter_str not in exp_id:
            continue
        yield exp_id, path


def main():
    args = parse_args()
    logs = list(find_logs(args.log_dir, args.filter))
    if not logs:
        print(f"No log.csv files found under {args.log_dir}" +
              (f" matching '{args.filter}'" if args.filter else ""))
        return

    rows, series_map = [], {}
    for exp_id, csv_path in logs:
        m  = load_series(csv_path, args.metric)
        t  = load_series(csv_path, args.reward_term)
        ar = load_series(csv_path, "reward/action_rate")
        row = {
            "exp_id":    exp_id,
            "final":     float(m[-1])    if len(m)  else None,
            "peak":      float(m.max())  if len(m)  else None,
            "conv":      convergence_epoch(m),
            "ar_mean":   float(ar.mean()) if len(ar) else None,
        }
        for col, short in GAIT_COLUMNS:
            row[short] = converged_mean(load_series(csv_path, col))
        rows.append(row)
        series_map[exp_id] = {"metric": m, "term": t, "action_rate": ar}

    # baseline first, then sort by peak desc
    rows.sort(key=lambda r: (0 if "baseline" in r["exp_id"] else 1, -(r["peak"] or -1e9)))

    baseline_peak = next((r["peak"] for r in rows if "baseline" in r["exp_id"] and r["peak"]), None)

    # ---- table ----
    width = 85 + 9 * len(GAIT_COLUMNS)
    print(f"\n{'='*width}\n  EXPERIMENT COMPARISON   primary={args.metric}\n{'='*width}")
    hdr = f"{'Experiment':<38} {'Final':>8} {'Peak':>8} {'vs Base':>8} {'Conv':>6} {'AR mean':>8}"
    for _, short in GAIT_COLUMNS:
        hdr += f" {short:>8}"
    print(hdr)
    print("-" * width)
    for r in rows:
        fs   = f"{r['final']:.4f}"  if r["final"] is not None else "n/a"
        ps   = f"{r['peak']:.4f}"   if r["peak"]  is not None else "n/a"
        conv = str(r["conv"])        if r["conv"]  is not None else "n/a"
        arm  = f"{r['ar_mean']:.4f}" if r["ar_mean"] is not None else "n/a"
        delta = (f"{r['peak']-baseline_peak:+.4f}" if baseline_peak and r["peak"] else "n/a")
        line = f"{r['exp_id']:<38} {fs:>8} {ps:>8} {delta:>8} {conv:>6} {arm:>8}"
        for _, short in GAIT_COLUMNS:
            line += f" {(f'{r[short]:.3f}' if r[short] is not None else 'n/a'):>8}"
        print(line)
    print(f"{'='*width}")
    print("  Gait columns are converged means (last 10% of epochs); reward alone "
          "does not decide an arm.\n")

    if args.no_plot:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available, skipping plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Reward Experiment Comparison", fontsize=13)
    titles = [args.metric, args.reward_term, "reward/action_rate (raw penalty)"]
    keys   = ["metric", "term", "action_rate"]

    for ax, key, title in zip(axes, keys, titles):
        for exp_id, data in sorted(series_map.items()):
            s = data[key]
            if len(s):
                lw = 2.0 if "baseline" in exp_id else 1.5
                ls = "-"  if "baseline" in exp_id else "--"
                ax.plot(s, label=exp_id, linewidth=lw, linestyle=ls)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Eval epoch")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Plot saved → {args.output}")


if __name__ == "__main__":
    main()
