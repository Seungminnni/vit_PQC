from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LWE ViT training history curves.")
    parser.add_argument("history_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def read_history(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, float] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    continue
            rows.append(parsed)
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    return rows


def read_best_epoch(history_csv: Path) -> int | None:
    for candidate in (history_csv.parent / "summary.json", history_csv.parent / "metrics.json"):
        if not candidate.exists():
            continue
        payload = json.loads(candidate.read_text())
        value = payload.get("best_epoch")
        if value is not None:
            return int(value)
    return None


def series(rows: list[dict[str, float]], key: str) -> list[float]:
    return [row[key] for row in rows if key in row]


def plot_line(ax, epochs: list[float], rows: list[dict[str, float]], key: str, label: str) -> None:
    values = series(rows, key)
    if values:
        ax.plot(epochs[: len(values)], values, label=label, linewidth=1.8)


def mark_best_epoch(
    ax,
    rows: list[dict[str, float]],
    best_epoch: int | None,
    keys: list[str],
    *,
    label_once: bool = False,
) -> None:
    if best_epoch is None:
        return
    best_rows = [row for row in rows if int(row.get("epoch", -1)) == best_epoch]
    if not best_rows:
        return
    row = best_rows[0]
    first = True
    for key in keys:
        if key not in row:
            continue
        label = f"best/test epoch {best_epoch}" if label_once and first else None
        ax.scatter([best_epoch], [row[key]], s=22, c="black", marker="o", zorder=5, label=label)
        first = False


def main() -> int:
    args = parse_args()
    rows = read_history(args.history_csv)
    best_epoch = read_best_epoch(args.history_csv)
    epochs = series(rows, "epoch")
    if not epochs:
        epochs = list(range(1, len(rows) + 1))

    output = args.output
    if output is None:
        output = args.history_csv.parent / "plots" / "loss_accuracy.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    title = args.title or args.history_csv.parent.parent.name
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(title, fontsize=14)

    loss_ax = axes[0, 0]
    plot_line(loss_ax, epochs, rows, "train_loss", "train")
    plot_line(loss_ax, epochs, rows, "train_eval_loss", "train_eval")
    plot_line(loss_ax, epochs, rows, "val_loss", "val")
    mark_best_epoch(loss_ax, rows, best_epoch, ["val_loss"], label_once=True)
    loss_ax.set_title("Loss")
    loss_ax.set_xlabel("epoch")
    loss_ax.set_ylabel("loss")
    loss_ax.set_yscale("log")
    loss_ax.grid(True, alpha=0.3)
    loss_ax.legend()

    acc_ax = axes[0, 1]
    plot_line(acc_ax, epochs, rows, "train_metric_coord_acc", "train coord")
    plot_line(acc_ax, epochs, rows, "val_metric_coord_acc", "val coord")
    plot_line(acc_ax, epochs, rows, "train_metric_exact_match", "train exact")
    plot_line(acc_ax, epochs, rows, "val_metric_exact_match", "val exact")
    mark_best_epoch(acc_ax, rows, best_epoch, ["val_metric_coord_acc", "val_metric_exact_match"])
    acc_ax.set_title("Accuracy")
    acc_ax.set_xlabel("epoch")
    acc_ax.set_ylabel("score")
    acc_ax.set_ylim(-0.02, 1.02)
    acc_ax.grid(True, alpha=0.3)
    acc_ax.legend()

    support_ax = axes[1, 0]
    plot_line(support_ax, epochs, rows, "train_metric_support_f1", "train support_f1")
    plot_line(support_ax, epochs, rows, "val_metric_support_f1", "val support_f1")
    plot_line(support_ax, epochs, rows, "train_metric_support_precision", "train precision")
    plot_line(support_ax, epochs, rows, "val_metric_support_precision", "val precision")
    plot_line(support_ax, epochs, rows, "train_metric_support_recall", "train recall")
    plot_line(support_ax, epochs, rows, "val_metric_support_recall", "val recall")
    mark_best_epoch(support_ax, rows, best_epoch, ["val_metric_support_f1"])
    support_ax.set_title("Support Metrics")
    support_ax.set_xlabel("epoch")
    support_ax.set_ylabel("score")
    support_ax.set_ylim(-0.02, 1.02)
    support_ax.grid(True, alpha=0.3)
    support_ax.legend(ncol=2, fontsize=8)

    residual_ax = axes[1, 1]
    plot_line(residual_ax, epochs, rows, "train_metric_pred_residual_std_mean", "train pred residual std")
    plot_line(residual_ax, epochs, rows, "val_metric_pred_residual_std_mean", "val pred residual std")
    plot_line(residual_ax, epochs, rows, "train_metric_oracle_residual_std_mean", "train oracle residual std")
    plot_line(residual_ax, epochs, rows, "val_metric_oracle_residual_std_mean", "val oracle residual std")
    mark_best_epoch(residual_ax, rows, best_epoch, ["val_metric_pred_residual_std_mean"])
    residual_ax.set_title("Residual Std")
    residual_ax.set_xlabel("epoch")
    residual_ax.set_ylabel("std")
    residual_ax.grid(True, alpha=0.3)
    residual_ax.legend(fontsize=8)

    fig.savefig(output, dpi=160)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
