from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_label(run_dir: Path) -> str:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return run_dir.name
    with config_path.open("r", encoding="utf-8") as fp:
        cfg = json.load(fp)
    lwe = cfg.get("lwe", {})
    name = cfg.get("run_name", run_dir.name)
    return f"{name}\\nh={lwe.get('h')} n={lwe.get('n')} sigma={lwe.get('sigma_e')}"


def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    train = [row for row in rows if row.get("split") == "train"]
    eval_rows = [row for row in rows if row.get("split") == "eval"]
    return train, eval_rows


def plot_train_loss(run_dirs: list[Path], out_dir: Path) -> None:
    plt.figure(figsize=(9, 5))
    for run_dir in run_dirs:
        rows, _ = split_rows(read_jsonl(run_dir / "metrics.jsonl"))
        if not rows:
            continue
        xs = [row["step"] for row in rows]
        ys = [row.get("loss_support", row.get("loss")) for row in rows]
        plt.plot(xs, ys, label=run_label(run_dir))
    plt.xlabel("step")
    plt.ylabel("support loss")
    plt.yscale("log")
    plt.title("Training loss")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "train_loss.png", dpi=170)
    plt.close()


def plot_eval_metric(run_dirs: list[Path], out_dir: Path, metric: str) -> None:
    plt.figure(figsize=(9, 5))
    plotted = False
    for run_dir in run_dirs:
        _, rows = split_rows(read_jsonl(run_dir / "metrics.jsonl"))
        rows = [row for row in rows if metric in row]
        if not rows:
            continue
        xs = [row["step"] for row in rows]
        ys = [row[metric] for row in rows]
        plt.plot(xs, ys, marker="o", label=run_label(run_dir))
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel("step")
    plt.ylabel(metric)
    plt.ylim(bottom=0.0)
    if metric not in ("residual_gap", "candidate_count", "reduction_factor"):
        plt.ylim(0.0, 1.05)
    plt.title(metric)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"eval_{metric}.png", dpi=170)
    plt.close()


def final_eval_rows(run_dirs: list[Path]) -> list[dict]:
    rows = []
    for run_dir in run_dirs:
        _, eval_rows = split_rows(read_jsonl(run_dir / "metrics.jsonl"))
        if not eval_rows:
            continue
        row = dict(eval_rows[-1])
        row["label"] = run_label(run_dir)
        row["run_dir"] = str(run_dir)
        rows.append(row)
    return rows


def plot_final_bars(rows: list[dict], out_dir: Path, metrics: list[str]) -> None:
    labels = [row["label"] for row in rows]
    x = list(range(len(rows)))
    width = 0.8 / max(1, len(metrics))
    plt.figure(figsize=(max(8, len(rows) * 2.5), 5))
    for i, metric in enumerate(metrics):
        ys = [row.get(metric, 0.0) for row in rows]
        offset = (i - (len(metrics) - 1) / 2) * width
        plt.bar([v + offset for v in x], ys, width=width, label=metric)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("rate")
    plt.title("Final eval metrics")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "final_eval_bars.png", dpi=170)
    plt.close()


def write_summary(rows: list[dict], out_dir: Path) -> None:
    keys = [
        "run_dir",
        "step",
        "coord_acc",
        "pre_rerank_full_match",
        "direct_full_match",
        "candidate_hit_rate",
        "post_rerank_full_match",
        "post_verifier_full_match",
        "rerank_success_given_hit",
        "support_precision",
        "support_recall",
        "h_abs_error",
        "pred_h_mean",
        "true_h_mean",
        "residual_gap",
        "best_residual_score",
        "candidate_count",
        "reduction_factor",
    ]
    with (out_dir / "summary.csv").open("w", encoding="utf-8") as fp:
        fp.write(",".join(keys) + "\n")
        for row in rows:
            fp.write(",".join(str(row.get(key, "")) for key in keys) + "\n")


def discover_run_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if (path / "metrics.jsonl").exists())


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RHIE-CG metrics from metrics.jsonl")
    parser.add_argument("--runs", nargs="*", default=None, help="Run directories containing metrics.jsonl")
    parser.add_argument("--root", default="results/csv")
    parser.add_argument("--out_dir", default="results/figures/summary_plots")
    args = parser.parse_args()

    run_dirs = [Path(p) for p in args.runs] if args.runs else discover_run_dirs(Path(args.root))
    run_dirs = [p for p in run_dirs if (p / "metrics.jsonl").exists()]
    if not run_dirs:
        raise SystemExit("No run directories with metrics.jsonl found")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_train_loss(run_dirs, out_dir)
    for metric in [
        "candidate_hit_rate",
        "post_rerank_full_match",
        "post_verifier_full_match",
        "pre_rerank_full_match",
        "direct_full_match",
        "coord_acc",
        "support_precision",
        "support_recall",
        "rerank_success_given_hit",
        "h_abs_error",
        "pred_h_mean",
        "true_h_mean",
        "residual_gap",
        "best_residual_score",
        "candidate_count",
        "reduction_factor",
    ]:
        plot_eval_metric(run_dirs, out_dir, metric)
    rows = final_eval_rows(run_dirs)
    plot_final_bars(rows, out_dir, ["direct_full_match", "post_verifier_full_match", "post_rerank_full_match"])
    write_summary(rows, out_dir)
    print({"out_dir": str(out_dir), "runs": [str(p) for p in run_dirs]})


if __name__ == "__main__":
    main()
