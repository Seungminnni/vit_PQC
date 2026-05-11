import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import build_synthetic_lwe_datasets
from model import LWEColumnViT


def default_m(n: int) -> int:
    return 16 * n


def default_h_max(n: int) -> int:
    return max(2, n // 8)


def default_p_nonzero(n: int) -> float:
    return 3.0 / max(n, 1)


def default_runs_root() -> Path:
    return Path(__file__).resolve().parent / "runs"


def format_slug_value(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:g}".replace(".", "p")
    return str(value).replace("/", "_")


def default_run_name(args) -> str:
    h_part = {
        "fixed_h": f"h{args.fixed_h}",
        "variable_h": f"h{args.h_min}-{args.h_max}",
        "bernoulli": f"p{format_slug_value(args.p_nonzero)}",
    }[args.h_setting]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"raw1_n{args.n}_m{args.m}_q{args.q}_{h_part}_seed{args.seed}_{timestamp}"


def parse_args():
    parser = argparse.ArgumentParser(description="Train final LWEColumnViT on leakage-free synthetic LWE splits.")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--m", "--M", dest="m", type=int, default=None)
    parser.add_argument("--q", type=int, default=257)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument(
        "--noise-distribution",
        "--noise_distribution",
        dest="noise_distribution",
        choices=["discrete_gaussian", "bounded_integer"],
        default="discrete_gaussian",
    )
    parser.add_argument("--noise-bound", "--noise_bound", dest="noise_bound", type=int, default=None)
    parser.add_argument(
        "--h-setting",
        "--h_setting",
        dest="h_setting",
        choices=["variable_h", "variable", "fixed_h", "fixed", "bernoulli"],
        default="variable_h",
    )
    parser.add_argument("--fixed-h", "--fixed_h", dest="fixed_h", type=int, default=None)
    parser.add_argument("--h-min", "--h_min", dest="h_min", type=int, default=None)
    parser.add_argument("--h-max", "--h_max", dest="h_max", type=int, default=None)
    parser.add_argument("--p-nonzero", "--p_nonzero", dest="p_nonzero", type=float, default=None)
    parser.add_argument("--shared-a", "--shared_a", dest="shared_a", action="store_true")
    parser.add_argument(
        "--row-permutation",
        "--row_permutation",
        dest="row_permutation",
        choices=["none", "global", "per_sample"],
        default="none",
    )
    parser.add_argument("--num-train", "--num_train", dest="num_train", type=int, default=4096)
    parser.add_argument("--num-val", "--num_val", dest="num_val", type=int, default=1024)
    parser.add_argument("--num-test", "--num_test", dest="num_test", type=int, default=1024)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", "--embed_dim", dest="embed_dim", type=int, default=128)
    parser.add_argument("--num-heads", "--num_heads", dest="num_heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pos-weight-mode", "--pos_weight_mode", dest="pos_weight_mode", choices=["none", "auto"], default="auto")
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", "--run_dir", dest="run_dir", type=Path, default=default_runs_root())
    parser.add_argument("--run-name", "--run_name", dest="run_name", default=None)
    parser.add_argument("--no-save-checkpoint", "--no_save_checkpoint", dest="save_checkpoint", action="store_false")
    parser.set_defaults(save_checkpoint=True)
    return normalize_args(parser.parse_args())


def normalize_args(args):
    args.m = args.m if args.m is not None else default_m(args.n)
    args.h_setting = {"fixed": "fixed_h", "variable": "variable_h"}.get(args.h_setting, args.h_setting)

    if args.fixed_h is not None and args.h_setting == "variable_h" and args.h_min is None and args.h_max is None and args.p_nonzero is None:
        args.h_setting = "fixed_h"
    elif args.p_nonzero is not None and args.h_setting == "variable_h" and args.h_min is None and args.h_max is None:
        args.h_setting = "bernoulli"

    if args.h_setting == "variable_h":
        args.h_min = 1 if args.h_min is None else args.h_min
        args.h_max = default_h_max(args.n) if args.h_max is None else args.h_max
    elif args.h_setting == "fixed_h":
        if args.fixed_h is None:
            raise ValueError("fixed_h is required when h_setting='fixed_h'")
    elif args.h_setting == "bernoulli":
        args.p_nonzero = default_p_nonzero(args.n) if args.p_nonzero is None else args.p_nonzero
    else:
        raise ValueError(f"Unknown h_setting: {args.h_setting}")

    if args.shared_a and args.row_permutation == "per_sample":
        raise ValueError("row_permutation='per_sample' is not supported with shared_a=True")
    args.run_dir = args.run_dir.expanduser()
    if args.run_name is None:
        args.run_name = default_run_name(args)
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataset_summary(dataset) -> dict[str, float]:
    secret = dataset.secret.to(torch.float32)
    h_values = secret.sum(dim=1)
    zero_pred = torch.zeros_like(secret)
    return {
        "samples": float(len(dataset)),
        "avg_h": float(h_values.mean().item()),
        "nonzero_rate": float(secret.mean().item()),
        "all_zero_coord_acc": float((zero_pred == secret).float().mean().item()),
        "all_zero_exact": float((zero_pred == secret).all(dim=1).float().mean().item()),
    }


def build_loss(args, train_dataset, device):
    if args.pos_weight_mode == "none":
        return nn.BCEWithLogitsLoss()
    pos_rate = train_dataset.secret.to(torch.float32).mean().item()
    pos_rate = min(max(pos_rate, 1e-6), 1.0 - 1e-6)
    pos_weight = torch.tensor([(1.0 - pos_rate) / pos_rate], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    preds = (logits > 0).to(torch.int64)
    labels = labels.to(torch.int64)
    true_support = labels != 0
    pred_support = preds != 0
    tp = torch.logical_and(true_support, pred_support).sum(dim=1).to(torch.float32)
    fp = torch.logical_and(~true_support, pred_support).sum(dim=1).to(torch.float32)
    fn = torch.logical_and(true_support, ~pred_support).sum(dim=1).to(torch.float32)
    precision = tp / (tp + fp).clamp(min=1.0)
    recall = tp / (tp + fn).clamp(min=1.0)
    f1_den = precision + recall
    empty_match = torch.logical_and(true_support.sum(dim=1) == 0, pred_support.sum(dim=1) == 0)
    f1 = torch.where(f1_den > 0, 2.0 * precision * recall / f1_den, empty_match.to(torch.float32))
    return {
        "coord_acc": float((preds == labels).float().mean().item()),
        "exact_match": float((preds == labels).all(dim=1).float().mean().item()),
        "hamming": float((preds != labels).float().sum(dim=1).mean().item()),
        "pred_h": float(pred_support.sum(dim=1).float().mean().item()),
        "true_h": float(true_support.sum(dim=1).float().mean().item()),
        "support_precision": float(precision.mean().item()),
        "support_recall": float(recall.mean().item()),
        "support_f1": float(f1.mean().item()),
    }


def merge_metric_sums(total: dict[str, float], metrics: dict[str, float], batch_size: int) -> None:
    for key, value in metrics.items():
        total[key] = total.get(key, 0.0) + value * batch_size


def average_metrics(total: dict[str, float], total_samples: int) -> dict[str, float]:
    return {key: value / max(total_samples, 1) for key, value in total.items()}


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True))


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(json_safe(rows))


def flatten_prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def run_epoch(loader, model, criterion, optimizer, device, training: bool) -> tuple[float, dict[str, float]]:
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    metric_total: dict[str, float] = {}
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="train" if training else "eval", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["target"].to(device, non_blocking=True).float()
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            merge_metric_sums(metric_total, compute_metrics(logits.detach(), labels.detach()), batch_size)

    return total_loss / max(total_samples, 1), average_metrics(metric_total, total_samples)


def metric_line(prefix: str, loss: float, metrics: dict[str, float]) -> str:
    return (
        f"{prefix}_loss={loss:.4f} "
        f"{prefix}_coord={metrics['coord_acc']*100:.2f}% "
        f"{prefix}_exact={metrics['exact_match']*100:.2f}% "
        f"{prefix}_hamm={metrics['hamming']:.2f} "
        f"{prefix}_pred_h={metrics['pred_h']:.2f} "
        f"{prefix}_true_h={metrics['true_h']:.2f} "
        f"{prefix}_f1={metrics['support_f1']*100:.2f}%"
    )


def run_toy_experiment():
    args = parse_args()
    set_seed(args.seed)
    run_dir = args.run_dir / args.run_name / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "run.log"
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_handle = log_path.open("w", buffering=1)
    sys.stdout = TeeStream(original_stdout, log_handle)
    sys.stderr = TeeStream(original_stderr, log_handle)

    try:
        return run_toy_experiment_inner(args, run_dir)
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


def run_toy_experiment_inner(args, run_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = vars(args).copy()
    config["run_output_dir"] = run_dir
    save_json(run_dir / "config.json", config)

    print(f"[INFO] device={device}")
    print(f"[INFO] saving outputs to {run_dir}")
    print(
        "[INFO] dataset="
        f"n={args.n} m={args.m} q={args.q} sigma={args.sigma:g} "
        f"h_setting={args.h_setting} fixed_h={args.fixed_h} h_min={args.h_min} h_max={args.h_max} "
        f"p_nonzero={args.p_nonzero} shared_a={args.shared_a} input=raw1_[A|b]/q"
    )

    train_dataset, val_dataset, test_dataset = build_synthetic_lwe_datasets(args, args.seed)
    print(f"[DATA] train={dataset_summary(train_dataset)}")
    print(f"[DATA] val={dataset_summary(val_dataset)}")
    print(f"[DATA] test={dataset_summary(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = LWEColumnViT(
        M=args.m,
        n=args.n,
        in_channels=train_dataset.in_channels,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = build_loss(args, train_dataset, device)
    num_parameters = count_parameters(model)
    print(f"[INFO] model_parameters={num_parameters}")

    best_val_exact = -1.0
    best_val_metrics = None
    best_train_metrics = None
    best_epoch = 0
    best_state = None
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(train_loader, model, criterion, optimizer, device, training=True)
        val_loss, val_metrics = run_epoch(val_loader, model, criterion, optimizer, device, training=False)
        if val_metrics["exact_match"] > best_val_exact:
            best_val_exact = val_metrics["exact_match"]
            best_val_metrics = dict(val_metrics)
            best_train_metrics = dict(train_metrics)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if args.save_checkpoint:
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "config": json_safe(config),
                        "epoch": best_epoch,
                        "best_val_metrics": best_val_metrics,
                    },
                    run_dir / "best.pt",
                )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_epoch": best_epoch,
            "best_val_exact_match": best_val_exact,
            **flatten_prefixed("train_metric", train_metrics),
            **flatten_prefixed("val_metric", val_metrics),
        }
        history.append(record)
        save_json(run_dir / "history.json", history)
        save_csv(run_dir / "history.csv", history)
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"{metric_line('train', train_loss, train_metrics)} | "
            f"{metric_line('val', val_loss, val_metrics)}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_metrics = run_epoch(test_loader, model, criterion, optimizer, device, training=False)
    print(f"[TEST] {metric_line('test', test_loss, test_metrics)}")
    summary = {
        "run_output_dir": str(run_dir),
        "seed": args.seed,
        "num_parameters": num_parameters,
        "best_epoch": best_epoch,
        "best_val_exact_match": best_val_exact,
        "test_loss": test_loss,
        **flatten_prefixed("train_data", dataset_summary(train_dataset)),
        **flatten_prefixed("val_data", dataset_summary(val_dataset)),
        **flatten_prefixed("test_data", dataset_summary(test_dataset)),
        **flatten_prefixed("best_train_metric", best_train_metrics or {}),
        **flatten_prefixed("best_val_metric", best_val_metrics or {}),
        **flatten_prefixed("test_metric", test_metrics),
    }
    save_json(run_dir / "summary.json", summary)
    save_csv(run_dir / "summary.csv", [summary])
    save_json(run_dir / "metrics.json", summary)
    save_csv(run_dir / "metrics.csv", [summary])
    print(f"[INFO] saved outputs to {run_dir}")
    return {"history": history, "summary": summary}


if __name__ == "__main__":
    run_toy_experiment()
