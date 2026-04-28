from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def discover_runs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if (path / "config.json").exists() and (path / "metrics.jsonl").exists())


def run_command(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def write_index(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["run", "analysis_dir", "embedding_dir", "metrics_dir", "checkpoint", "config"]
    with path.open("w", encoding="utf-8") as fp:
        fp.write(",".join(keys) + "\n")
        for row in rows:
            fp.write(",".join(row.get(key, "") for key in keys) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_metric_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    train = [row for row in rows if row.get("split") == "train"]
    eval_rows = [row for row in rows if row.get("split") == "eval"]
    return train, eval_rows


def load_run_label(run_dir: Path) -> str:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return run_dir.name
    with config_path.open("r", encoding="utf-8") as fp:
        cfg = json.load(fp)
    lwe = cfg.get("lwe", {})
    run_name = cfg.get("run_name", run_dir.name)
    return f"{run_name} | h={lwe.get('h')} n={lwe.get('n')} M={lwe.get('M')} sigma={lwe.get('sigma_e')}"


def plot_series(
    rows: list[dict],
    metric: str,
    out_path: Path,
    title: str,
    ylabel: str | None = None,
    yscale: str | None = None,
    ylim: tuple[float, float] | None = None,
    marker: str | None = None,
) -> bool:
    points = [(row["step"], row[metric]) for row in rows if metric in row]
    if not points:
        return False
    xs, ys = zip(*points)
    plt.figure(figsize=(8, 4.8))
    plt.plot(xs, ys, marker=marker)
    plt.xlabel("step")
    plt.ylabel(ylabel or metric)
    plt.title(title)
    if yscale:
        plt.yscale(yscale)
    if ylim:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()
    return True


def plot_eval_rates(eval_rows: list[dict], out_path: Path, title: str) -> bool:
    metrics = [
        "coord_acc",
        "pre_rerank_full_match",
        "candidate_hit_rate",
        "post_rerank_full_match",
        "rerank_success_given_hit",
    ]
    plotted = False
    plt.figure(figsize=(8.5, 5))
    for metric in metrics:
        points = [(row["step"], row[metric]) for row in eval_rows if metric in row]
        if not points:
            continue
        xs, ys = zip(*points)
        plt.plot(xs, ys, marker="o", label=metric)
        plotted = True
    if not plotted:
        plt.close()
        return False
    plt.xlabel("step")
    plt.ylabel("rate")
    plt.ylim(0.0, 1.05)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()
    return True


def write_metric_tables(rows: list[dict], out_dir: Path) -> None:
    keys = sorted({key for row in rows for key in row.keys()})
    with (out_dir / "metrics_flat.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def generate_metric_artifacts(run_dir: Path, out_dir: Path) -> list[str]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(metrics_path)
    train_rows, eval_rows = split_metric_rows(rows)
    label = load_run_label(run_dir)
    written = []

    write_metric_tables(rows, out_dir)
    written.append(str(out_dir / "metrics_flat.csv"))

    plots = [
        plot_series(
            train_rows,
            "loss_support",
            out_dir / "train_loss_support.png",
            f"Train support loss\n{label}",
            ylabel="loss_support",
            yscale="log",
        ),
        plot_series(
            train_rows,
            "loss",
            out_dir / "train_loss.png",
            f"Train loss\n{label}",
            ylabel="loss",
            yscale="log",
        ),
        plot_eval_rates(eval_rows, out_dir / "eval_recovery_rates.png", f"Eval recovery rates\n{label}"),
        plot_series(
            eval_rows,
            "coord_acc",
            out_dir / "eval_coord_acc.png",
            f"Eval coordinate accuracy\n{label}",
            ylabel="coord_acc",
            ylim=(0.0, 1.05),
            marker="o",
        ),
        plot_series(
            eval_rows,
            "residual_gap",
            out_dir / "eval_residual_gap.png",
            f"Eval residual gap\n{label}",
            ylabel="residual_gap",
            marker="o",
        ),
        plot_series(
            eval_rows,
            "candidate_count",
            out_dir / "eval_candidate_count.png",
            f"Eval candidate count\n{label}",
            ylabel="candidate_count",
            marker="o",
        ),
        plot_series(
            eval_rows,
            "reduction_factor",
            out_dir / "eval_reduction_factor.png",
            f"Eval reduction factor\n{label}",
            ylabel="reduction_factor",
            marker="o",
        ),
    ]
    for path, made in [
        (out_dir / "train_loss_support.png", plots[0]),
        (out_dir / "train_loss.png", plots[1]),
        (out_dir / "eval_recovery_rates.png", plots[2]),
        (out_dir / "eval_coord_acc.png", plots[3]),
        (out_dir / "eval_residual_gap.png", plots[4]),
        (out_dir / "eval_candidate_count.png", plots[5]),
        (out_dir / "eval_reduction_factor.png", plots[6]),
    ]:
        if made:
            written.append(str(path))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-run analysis and embedding artifacts")
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--root", default="results/csv")
    parser.add_argument("--out_root", default="results/figures/run_artifacts")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--max_cases", type=int, default=2)
    parser.add_argument("--num_embedding_cases", type=int, default=1)
    parser.add_argument("--max_channels", type=int, default=36)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_analysis", action="store_true")
    parser.add_argument("--skip_embedding", action="store_true")
    parser.add_argument("--skip_metrics", action="store_true")
    args = parser.parse_args()

    run_dirs = [Path(p) for p in args.runs] if args.runs else discover_runs(Path(args.root))
    out_root = Path(args.out_root)
    rows = []
    for run_dir in run_dirs:
        config = run_dir / "config.json"
        checkpoint = run_dir / "best.pt"
        if not config.exists():
            print(f"skip {run_dir}: missing config.json", flush=True)
            continue
        run_name = run_dir.name
        analysis_dir = out_root / run_name / "analysis"
        embedding_dir = out_root / run_name / "embedding"
        metrics_dir = out_root / run_name / "metrics"

        if not args.skip_analysis:
            if checkpoint.exists():
                run_command(
                    [
                        sys.executable,
                        "analyze_binary.py",
                        "--checkpoint",
                        str(checkpoint),
                        "--out_dir",
                        str(analysis_dir),
                        "--batch_size",
                        str(args.batch_size),
                        "--num_batches",
                        str(args.num_batches),
                        "--max_cases",
                        str(args.max_cases),
                        "--include_successes",
                        "--device",
                        args.device,
                    ]
                )
            else:
                print(f"skip analysis for {run_dir}: missing best.pt", flush=True)

        if not args.skip_embedding:
            run_command(
                [
                    sys.executable,
                    "visualize_embedding.py",
                    "--config_json",
                    str(config),
                    "--out_dir",
                    str(embedding_dir),
                    "--num_cases",
                    str(args.num_embedding_cases),
                    "--max_channels",
                    str(args.max_channels),
                    "--device",
                    args.device,
                ]
            )

        if not args.skip_metrics:
            written = generate_metric_artifacts(run_dir, metrics_dir)
            print({"run": run_name, "metrics_artifacts": written}, flush=True)

        rows.append(
            {
                "run": run_name,
                "analysis_dir": str(analysis_dir),
                "embedding_dir": str(embedding_dir),
                "metrics_dir": str(metrics_dir),
                "checkpoint": str(checkpoint) if checkpoint.exists() else "",
                "config": str(config),
            }
        )
    write_index(rows, out_root / "artifact_index.csv")
    print({"out_root": str(out_root), "runs": [row["run"] for row in rows]}, flush=True)


if __name__ == "__main__":
    main()
