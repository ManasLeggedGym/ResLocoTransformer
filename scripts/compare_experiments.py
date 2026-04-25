"""
scripts/compare_experiments.py — Compare reward experiments from CSV logs.

Usage:
    python3 scripts/compare_experiments.py
    python3 scripts/compare_experiments.py --log-dir ./log --metric Running_Average_Rewards
    python3 scripts/compare_experiments.py --log-dir ./log --no-plot
"""

import argparse
import csv
import glob
import os
import os.path as osp

import numpy as np

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
REPO_ROOT = osp.dirname(SCRIPT_DIR)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default=osp.join(REPO_ROOT, "log"))
    p.add_argument("--metric", default="Running_Average_Rewards",
                   help="CSV column to use as the primary comparison metric")
    p.add_argument("--reward-term", default="reward/forward_vel",
                   help="Secondary metric (reward term) to compare across experiments")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--output", default="reward_comparison.png")
    return p.parse_args()


def load_csv(path):
    """Return list of dicts (one per row) from a log.csv."""
    with open(path) as f:
        return list(csv.DictReader(f))


def extract_series(rows, column):
    """Extract a float series for *column*, skipping missing/empty values."""
    vals = []
    for r in rows:
        v = r.get(column, "").strip()
        if v:
            try:
                vals.append(float(v))
            except ValueError:
                pass
    return np.array(vals)


def convergence_epoch(series, threshold=0.95):
    """First epoch index where value >= threshold * peak."""
    if len(series) == 0:
        return None
    peak = series.max()
    idxs = np.where(series >= threshold * peak)[0]
    return int(idxs[0]) if len(idxs) > 0 else None


def find_logs(log_dir):
    """Yield (experiment_id, csv_path) pairs for all log.csv files found."""
    pattern = osp.join(log_dir, "*", "*", "*", "log.csv")
    for path in sorted(glob.glob(pattern)):
        parts = osp.relpath(path, log_dir).split(os.sep)
        exp_id = parts[0]
        yield exp_id, path


def print_table(rows):
    if not rows:
        print("No experiments found.")
        return

    col_w = [35, 13, 13, 13, 13, 12]
    headers = ["Experiment", "Final", "Peak", "Baseline Δ", "Conv. Epoch", "Term Mean"]
    sep = "  ".join("-" * w for w in col_w)
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)

    print(f"\n{'='*95}")
    print("  EXPERIMENT COMPARISON")
    print(f"{'='*95}")
    print(fmt.format(*headers))
    print(sep)

    baseline_peak = None
    for r in rows:
        if r["exp_id"].startswith("00_baseline"):
            baseline_peak = r["peak"]
            break

    for r in rows:
        final = f"{r['final']:.4f}" if r["final"] is not None else "n/a"
        peak  = f"{r['peak']:.4f}"  if r["peak"]  is not None else "n/a"
        conv  = str(r["conv"])       if r["conv"]  is not None else "n/a"
        term  = f"{r['term_mean']:.4f}" if r["term_mean"] is not None else "n/a"

        if baseline_peak is not None and r["peak"] is not None:
            delta = r["peak"] - baseline_peak
            delta_s = f"{delta:+.4f}"
        else:
            delta_s = "n/a"

        print(fmt.format(r["exp_id"][:35], final, peak, delta_s, conv, term))

    print(f"{'='*95}")


def plot_comparison(series_map, metric, reward_term, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available, skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Reward Experiment Comparison", fontsize=13)

    for exp_id, data in sorted(series_map.items()):
        if len(data["metric"]) > 0:
            axes[0].plot(data["metric"], label=exp_id, linewidth=1.5)
        if len(data["term"]) > 0:
            axes[1].plot(data["term"], label=exp_id, linewidth=1.5, linestyle="--")

    axes[0].set_title(metric.replace("_", " "))
    axes[0].set_xlabel("Eval epoch")
    axes[0].set_ylabel("Value")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    axes[1].set_title(reward_term)
    axes[1].set_xlabel("Eval epoch")
    axes[1].set_ylabel("Mean per-step value")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")


def main():
    args = parse_args()

    logs = list(find_logs(args.log_dir))
    if not logs:
        print(f"No log.csv files found under {args.log_dir}")
        return

    table_rows = []
    series_map = {}

    for exp_id, csv_path in logs:
        try:
            rows = load_csv(csv_path)
        except Exception as e:
            print(f"[warn] Cannot read {csv_path}: {e}")
            continue

        metric_series = extract_series(rows, args.metric)
        term_series   = extract_series(rows, args.reward_term)

        final     = float(metric_series[-1]) if len(metric_series) > 0 else None
        peak      = float(metric_series.max()) if len(metric_series) > 0 else None
        conv      = convergence_epoch(metric_series) if len(metric_series) > 0 else None
        term_mean = float(term_series.mean()) if len(term_series) > 0 else None

        table_rows.append({
            "exp_id": exp_id,
            "final": final,
            "peak": peak,
            "conv": conv,
            "term_mean": term_mean,
        })
        series_map[exp_id] = {"metric": metric_series, "term": term_series}

    # Sort: baseline first, then by peak descending
    table_rows.sort(key=lambda r: (0 if r["exp_id"].startswith("00") else 1,
                                   -(r["peak"] or -1e9)))
    print_table(table_rows)
    print(f"\nPrimary metric : {args.metric}")
    print(f"Secondary term : {args.reward_term}")

    if not args.no_plot:
        plot_comparison(series_map, args.metric, args.reward_term, args.output)


if __name__ == "__main__":
    main()
