import argparse
import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from lwe_image_experiment.data import DatasetSpec, SyntheticLWEDataset
    from lwe_image_experiment.metrics import batch_statistics, finalize_statistics, merge_statistics
    from lwe_image_experiment.models import build_model
else:
    from .data import DatasetSpec, SyntheticLWEDataset
    from .metrics import batch_statistics, finalize_statistics, merge_statistics
    from .models import build_model


CONFIG_KEYS = {
    "run_name",
    "n",
    "m",
    "q",
    "sigma",
    "secret_type",
    "encoding",
    "model",
    "h_setting",
    "fixed_h",
    "h_min",
    "h_max",
    "p_nonzero",
    "noise_distribution",
    "noise_bound",
    "row_permutation",
    "shared_a",
    "num_train",
    "num_val",
    "num_test",
    "train_eval_samples",
    "batch_size",
    "epochs",
    "warmup_epochs",
    "early_stopping_patience",
    "seed",
    "class_weight_mode",
    "residual_success_factor",
    "num_parameters",
}


MODEL_CHOICES = ("cnn", "alexnet", "resnet", "hybrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-free global LWE image experiment")
    parser.add_argument("--secret-type", "--secret_type", dest="secret_type", choices=["binary"], default="binary")
    parser.add_argument("--encoding", choices=["phase6"], default="phase6")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="hybrid")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--m", type=int, default=None, help="Defaults to 16 * n")
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
        "--row-permutation",
        "--row_permutation",
        dest="row_permutation",
        choices=["none", "global", "per_sample"],
        default="none",
        help="Ablation: permute LWE equation rows while preserving each equation's A,b,noise alignment.",
    )
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
    parser.add_argument("--num-train", "--num_train", "--train-samples", dest="num_train", type=int, default=4096)
    parser.add_argument("--num-val", "--num_val", "--val-samples", dest="num_val", type=int, default=1024)
    parser.add_argument("--num-test", "--num_test", "--test-samples", dest="num_test", type=int, default=1024)
    parser.add_argument(
        "--train-eval-samples",
        "--train_eval_samples",
        dest="train_eval_samples",
        type=int,
        default=4096,
        help="Fixed train subset size for eval-mode train metrics. Use 0 for the full train set.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--warmup-epochs",
        "--warmup_epochs",
        dest="warmup_epochs",
        type=int,
        default=0,
        help="Linearly warm up learning rate from 0 to --lr over this many epochs.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        "--early_stopping_patience",
        dest="early_stopping_patience",
        type=int,
        default=0,
        help="Stop after this many non-improving validation epochs. Use 0 to disable.",
    )
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", "--embed_dim", dest="embed_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num-heads", "--num_heads", dest="num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--class-weight-mode",
        "--class_weight_mode",
        dest="class_weight_mode",
        choices=["none", "inverse_prior"],
        default="inverse_prior",
    )
    parser.add_argument(
        "--residual-success-factor",
        "--residual_success_factor",
        dest="residual_success_factor",
        type=float,
        default=2.0,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated list, e.g. 0,1,2")
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=0)
    parser.add_argument("--run-name", "--run_name", dest="run_name", type=str, default=None)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--save-dir", dest="save_dir", type=Path, default=None, help="Deprecated alias for the final run directory")
    parser.add_argument("--save-best", "--save_best", dest="save_best", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_m(n: int) -> int:
    return 16 * n


def default_h_max(n: int) -> int:
    return max(2, n // 8)


def default_p_nonzero(n: int) -> float:
    return 3.0 / max(n, 1)


def infer_run_name(args: argparse.Namespace) -> str:
    shared = "sharedA" if args.shared_a else "freshA"
    result_encoding = getattr(args, "result_encoding", args.encoding)
    return (
        f"{args.secret_type}_{result_encoding}_{args.model}_"
        f"n{args.n}_m{args.m}_q{args.q}_sigma{args.sigma:g}_"
        f"{args.h_setting}_{shared}"
    )


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.m = args.m if args.m is not None else default_m(args.n)
    args.result_encoding = args.encoding
    h_setting_aliases = {
        "fixed": "fixed_h",
        "variable": "variable_h",
    }
    args.h_setting = h_setting_aliases.get(args.h_setting, args.h_setting)

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

    if args.run_name is None:
        args.run_name = infer_run_name(args)

    if args.warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")
    if args.warmup_epochs > args.epochs:
        raise ValueError("warmup_epochs must be <= epochs")
    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")

    if args.save_dir is not None:
        args.run_dir = args.save_dir
    else:
        args.run_dir = args.output_dir / args.run_name

    if args.shared_a and args.row_permutation == "per_sample":
        raise ValueError("row_permutation=per_sample is not supported with shared_a=True")

    return args


def parse_seed_list(args: argparse.Namespace) -> list[int]:
    if args.seeds is None:
        return [args.seed]
    return [int(token.strip()) for token in args.seeds.split(",") if token.strip()]


class TeeStream:
    def __init__(self, *streams):
        self.streams = list(streams)

    def add_stream(self, stream) -> None:
        self.streams.append(stream)

    def remove_stream(self, stream) -> None:
        self.streams.remove(stream)

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def require_cuda_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("WARNING: No CUDA GPU detected, falling back to CPU.")
    return torch.device("cpu")


def expected_nonzero_rate(args: argparse.Namespace) -> float:
    if args.h_setting == "fixed_h":
        return args.fixed_h / max(args.n, 1)
    if args.h_setting == "variable_h":
        return ((args.h_min + args.h_max) / 2.0) / max(args.n, 1)
    return float(args.p_nonzero)


def apply_row_permutation(dataset: SyntheticLWEDataset, mode: str, seed: int) -> None:
    if mode == "none":
        return

    generator = torch.Generator().manual_seed(seed)
    if mode == "global":
        permutation = torch.randperm(dataset.m, generator=generator)
        if dataset.A_shared is not None:
            dataset.A_shared = dataset.A_shared[permutation]
        elif dataset.A is not None:
            dataset.A = dataset.A[:, permutation, :]
        else:
            raise RuntimeError("Expected A or A_shared to be initialized.")
        dataset.b = dataset.b[:, permutation]
        dataset.noise = dataset.noise[:, permutation]
        dataset.oracle_residual = dataset.oracle_residual[:, permutation]
        return

    if mode == "per_sample":
        if dataset.A_shared is not None:
            raise ValueError("row_permutation=per_sample is not supported for shared A datasets")
        if dataset.A is None:
            raise RuntimeError("Expected per-sample A tensor to be initialized.")
        scores = torch.rand((dataset.num_samples, dataset.m), generator=generator)
        permutations = scores.argsort(dim=1)
        row_index = torch.arange(dataset.num_samples).unsqueeze(1)
        dataset.A = dataset.A[row_index, permutations, :]
        dataset.b = dataset.b[row_index, permutations]
        dataset.noise = dataset.noise[row_index, permutations]
        dataset.oracle_residual = dataset.oracle_residual[row_index, permutations]
        return

    raise ValueError(f"Unknown row_permutation: {mode}")


def build_datasets(args: argparse.Namespace, run_seed: int) -> tuple[SyntheticLWEDataset, SyntheticLWEDataset, SyntheticLWEDataset]:
    def make_spec(num_samples: int, split_offset: int) -> DatasetSpec:
        return DatasetSpec(
            num_samples=num_samples,
            m=args.m,
            n=args.n,
            q=args.q,
            secret_type=args.secret_type,
            h_setting=args.h_setting,
            p_nonzero=args.p_nonzero,
            fixed_h=args.fixed_h,
            h_min=args.h_min,
            h_max=args.h_max,
            sigma=args.sigma,
            noise_distribution=args.noise_distribution,
            noise_bound=args.noise_bound,
            encoding=args.encoding,
            shared_a=args.shared_a,
            seed=run_seed * 1000 + split_offset,
        )

    datasets = (
        SyntheticLWEDataset(make_spec(args.num_train, 11)),
        SyntheticLWEDataset(make_spec(args.num_val, 23)),
        SyntheticLWEDataset(make_spec(args.num_test, 37)),
    )
    for dataset, split_offset in zip(datasets, (11, 23, 37)):
        apply_row_permutation(dataset, mode=args.row_permutation, seed=run_seed * 1000 + split_offset + 101)
    return datasets


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dataset_statistics(dataset: SyntheticLWEDataset) -> dict[str, object]:
    secret = dataset.secret.to(torch.int64)
    support = secret != 0
    h_values = support.sum(dim=1).to(torch.float32)
    class_ids = secret
    value_names = ["0", "1"]

    one_hot = F.one_hot(class_ids, num_classes=dataset.num_classes).to(torch.float32)
    per_coord_probs = one_hot.mean(dim=0)
    overall_probs = one_hot.mean(dim=(0, 1))
    random_coord_acc = (per_coord_probs.pow(2).sum(dim=1)).mean().item()
    random_exact_match = per_coord_probs.pow(2).sum(dim=1).prod().item()
    zero_pred = torch.zeros_like(class_ids)
    all_zero_coord_acc = (zero_pred == class_ids).to(torch.float32).mean().item()
    all_zero_exact_match = (zero_pred == class_ids).all(dim=1).to(torch.float32).mean().item()

    stats: dict[str, object] = {
        "avg_h": h_values.mean().item(),
        "std_h": h_values.std(unbiased=False).item(),
        "nonzero_rate": support.to(torch.float32).mean().item(),
        "random_guess_coord_acc": random_coord_acc,
        "random_guess_exact_match": random_exact_match,
        "all_zero_coord_acc": all_zero_coord_acc,
        "all_zero_exact_match": all_zero_exact_match,
        "all_zero_active_recall": 0.0 if support.any().item() else 1.0,
    }

    for idx, value_name in enumerate(value_names):
        stats[f"class_prob_{value_name}"] = overall_probs[idx].item()

    return stats


def flatten_prefixed(prefix: str, payload: dict[str, object]) -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in payload.items():
        flat[f"{prefix}_{key}"] = value
    return flat


def build_loss(
    class_probs: list[float],
    class_weight_mode: str,
    device: torch.device,
) -> tuple[nn.Module, list[float]]:
    weights: torch.Tensor | None = None
    if class_weight_mode == "inverse_prior":
        probs = torch.tensor(class_probs, dtype=torch.float32, device=device).clamp(min=1e-6)
        weights = (1.0 / probs).clamp(max=100.0)
        weights = weights / weights.mean()
    elif class_weight_mode != "none":
        raise ValueError(f"Unknown class_weight_mode: {class_weight_mode}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    weight_list = weights.detach().cpu().tolist() if weights is not None else [1.0 for _ in class_probs]
    return criterion, weight_list


def model_input(model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    return batch["image"].to(device, non_blocking=True)


def metric_score(metrics: dict[str, float]) -> float:
    return metrics["exact_match"] + metrics["support_f1"]


def epoch_learning_rate(base_lr: float, epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0 or epoch > warmup_epochs:
        return base_lr
    return base_lr * epoch / warmup_epochs


def set_optimizer_lr(optimizer: optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    ordered = [
        "coord_acc",
        "exact_match",
        "support_f1",
        "support_precision",
        "support_recall",
        "pred_residual_std_mean",
        "pred_residual_std_gap_mean",
        "residual_success_rate",
    ]
    body = " ".join(f"{key}={metrics[key]:.4f}" for key in ordered if key in metrics)
    return f"{prefix} {body}"


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    secret_type: str,
    q: int,
    sigma: float,
    residual_success_factor: float,
) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    stat_list: list[dict[str, float]] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            inputs = model_input(model, batch, device=device)
            target = batch["target"].to(device, non_blocking=True)
            secret = batch["secret"].to(device, non_blocking=True)
            A = batch["A"].to(device, non_blocking=True)
            b = batch["b"].to(device, non_blocking=True)
            oracle_residual = batch["oracle_residual"].to(device, non_blocking=True)

            if training:
                optimizer.zero_grad()

            logits = model(inputs)
            num_classes = logits.shape[-1]
            loss = criterion(logits.reshape(-1, num_classes), target.reshape(-1))

            if training:
                loss.backward()
                optimizer.step()

            batch_size = target.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            stat_list.append(
                batch_statistics(
                    logits=logits,
                    secret=secret,
                    A=A,
                    b=b,
                    oracle_residual=oracle_residual,
                    q=q,
                    sigma=sigma,
                    residual_success_factor=residual_success_factor,
                    secret_type=secret_type,
                )
            )

    merged_stats = merge_statistics(stat_list)
    return total_loss / max(total_samples, 1), finalize_statistics(merged_stats, secret_type=secret_type)


def train_step_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train(True)
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        inputs = model_input(model, batch, device=device)
        target = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(inputs)
        num_classes = logits.shape[-1]
        loss = criterion(logits.reshape(-1, num_classes), target.reshape(-1))
        loss.backward()
        optimizer.step()

        batch_size = target.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def base_config_row(args: argparse.Namespace, run_seed: int, num_parameters: int, class_weights: list[float]) -> dict[str, object]:
    row = {
        "run_name": args.run_name,
        "n": args.n,
        "m": args.m,
        "q": args.q,
        "sigma": args.sigma,
        "secret_type": args.secret_type,
        "encoding": args.result_encoding,
        "model": args.model,
        "h_setting": args.h_setting,
        "fixed_h": args.fixed_h,
        "h_min": args.h_min,
        "h_max": args.h_max,
        "p_nonzero": args.p_nonzero,
        "noise_distribution": args.noise_distribution,
        "noise_bound": args.noise_bound,
        "row_permutation": args.row_permutation,
        "shared_a": args.shared_a,
        "num_train": args.num_train,
        "num_val": args.num_val,
        "num_test": args.num_test,
        "train_eval_samples": args.train_eval_samples,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "seed": run_seed,
        "class_weight_mode": args.class_weight_mode,
        "residual_success_factor": args.residual_success_factor,
        "num_parameters": num_parameters,
    }
    for idx, weight in enumerate(class_weights):
        row[f"class_weight_{idx}"] = weight
    return row


def save_run_metadata(run_dir: Path, args: argparse.Namespace, seed_list: list[int]) -> None:
    metadata_args = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
        if key != "run_dir"
    }
    metadata_args["encoding"] = args.result_encoding
    if args.result_encoding != args.encoding:
        metadata_args["internal_encoding"] = args.encoding
    payload = {
        "command": " ".join(sys.argv),
        "args": metadata_args,
        "seed_list": seed_list,
    }
    save_json(run_dir / "config.json", payload)
    (run_dir / "command.txt").write_text(" ".join(sys.argv))


def make_train_eval_dataset(train_dataset: SyntheticLWEDataset, train_eval_samples: int) -> SyntheticLWEDataset | Subset:
    if train_eval_samples == 0 or train_eval_samples >= len(train_dataset):
        return train_dataset
    if train_eval_samples < 0:
        raise ValueError("train_eval_samples must be non-negative")
    return Subset(train_dataset, list(range(train_eval_samples)))


def train_single_seed(args: argparse.Namespace, run_seed: int, device: torch.device) -> dict[str, object]:
    set_seed(run_seed)
    train_dataset, val_dataset, test_dataset = build_datasets(args, run_seed)
    train_stats = dataset_statistics(train_dataset)
    val_stats = dataset_statistics(val_dataset)
    test_stats = dataset_statistics(test_dataset)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    train_eval_dataset = make_train_eval_dataset(train_dataset, args.train_eval_samples)
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(
        model_name=args.model,
        in_channels=train_dataset.in_channels,
        m=train_dataset.m,
        n=train_dataset.n,
        secret_type=args.secret_type,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    num_parameters = sum(parameter.numel() for parameter in model.parameters())

    class_probs = [train_stats[f"class_prob_{name}"] for name in ["0", "1"]]
    criterion, class_weights = build_loss(
        class_probs=class_probs,
        class_weight_mode=args.class_weight_mode,
        device=device,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config_row = base_config_row(args, run_seed, num_parameters=num_parameters, class_weights=class_weights)
    summary_seed_dir = args.run_dir / f"seed_{run_seed}"
    summary_seed_dir.mkdir(parents=True, exist_ok=True)
    seed_log_handle = (summary_seed_dir / "run.log").open("w", buffering=1)
    if isinstance(sys.stdout, TeeStream):
        sys.stdout.add_stream(seed_log_handle)
    if isinstance(sys.stderr, TeeStream):
        sys.stderr.add_stream(seed_log_handle)

    save_json(summary_seed_dir / "config.json", config_row)

    try:
        print(f"Logging seed stdout/stderr to {summary_seed_dir / 'run.log'}")
        print("=== Seed Run ===")
        print(json.dumps(config_row, indent=2, sort_keys=True))
        print(
            f"[Baseline] random_guess_coord_acc={train_stats['random_guess_coord_acc']:.4f} "
            f"random_guess_exact_match={train_stats['random_guess_exact_match']:.6f} "
            f"avg_h={train_stats['avg_h']:.2f} nonzero_rate={train_stats['nonzero_rate']:.4f}"
        )

        history: list[dict[str, object]] = []
        best_state = None
        best_epoch = 0
        best_val_score = float("-inf")
        best_val_metrics: dict[str, float] | None = None
        best_train_loss = float("nan")
        best_train_eval_loss = float("nan")
        best_train_metrics: dict[str, float] | None = None
        epochs_without_improvement = 0
        early_stopped = False
        stopped_epoch = args.epochs
        train_start_time = time.perf_counter()

        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.perf_counter()
            current_lr = epoch_learning_rate(args.lr, epoch=epoch, warmup_epochs=args.warmup_epochs)
            set_optimizer_lr(optimizer, current_lr)
            train_loss = train_step_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
            train_eval_loss, train_metrics = run_epoch(
                model=model,
                loader=train_eval_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
                secret_type=args.secret_type,
                q=args.q,
                sigma=args.sigma,
                residual_success_factor=args.residual_success_factor,
            )
            val_loss, val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
                secret_type=args.secret_type,
                q=args.q,
                sigma=args.sigma,
                residual_success_factor=args.residual_success_factor,
            )

            score = metric_score(val_metrics)
            epoch_time_sec = time.perf_counter() - epoch_start_time

            record = {
                **config_row,
                "epoch": epoch,
                "lr": current_lr,
                "epoch_time_sec": epoch_time_sec,
                "train_loss": train_loss,
                "train_eval_loss": train_eval_loss,
                "val_loss": val_loss,
                **flatten_prefixed("train_metric", train_metrics),
                **flatten_prefixed("val_metric", val_metrics),
            }
            history.append(record)

            if score > best_val_score:
                best_val_score = score
                best_epoch = epoch
                best_val_metrics = dict(val_metrics)
                best_train_loss = train_loss
                best_train_eval_loss = train_eval_loss
                best_train_metrics = dict(train_metrics)
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(
                f"Seed {run_seed} | Epoch {epoch:03d}/{args.epochs} "
                f"lr={current_lr:.6g} train_loss={train_loss:.4f} train_eval_loss={train_eval_loss:.4f} "
                f"val_loss={val_loss:.4f} epoch_time_sec={epoch_time_sec:.2f}",
                flush=True,
            )
            print(format_metrics("  train_eval", train_metrics), flush=True)
            print(format_metrics("  val  ", val_metrics), flush=True)

            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                early_stopped = True
                stopped_epoch = epoch
                print(
                    f"Early stopping at epoch {epoch} after "
                    f"{args.early_stopping_patience} non-improving validation epochs."
                )
                break

        train_time_sec = time.perf_counter() - train_start_time

        if best_state is None or best_val_metrics is None or best_train_metrics is None:
            raise RuntimeError("Training did not produce a best checkpoint")

        final_record = history[-1]
        model.load_state_dict(best_state)
        _, test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            secret_type=args.secret_type,
            q=args.q,
            sigma=args.sigma,
            residual_success_factor=args.residual_success_factor,
        )

        summary = {
            **config_row,
            "best_epoch": best_epoch,
            "best_val_score": best_val_score,
            "stopped_epoch": stopped_epoch,
            "early_stopped": early_stopped,
            "train_time_sec": train_time_sec,
            "train_loss": best_train_loss,
            "train_eval_loss": best_train_eval_loss,
            "train_metric_exact_match": best_train_metrics["exact_match"],
            "train_metric_support_f1": best_train_metrics["support_f1"],
            "final_train_loss": final_record["train_loss"],
            "final_train_eval_loss": final_record["train_eval_loss"],
            "final_train_metric_exact_match": final_record["train_metric_exact_match"],
            "final_train_metric_support_f1": final_record["train_metric_support_f1"],
            **flatten_prefixed("train_data", train_stats),
            **flatten_prefixed("val_data", val_stats),
            **flatten_prefixed("test_data", test_stats),
            **flatten_prefixed("best_val_metric", best_val_metrics),
            **flatten_prefixed("test_metric", test_metrics),
        }

        save_json(summary_seed_dir / "history.json", history)
        save_csv(summary_seed_dir / "history.csv", history)
        save_json(summary_seed_dir / "metrics.json", summary)
        save_csv(summary_seed_dir / "metrics.csv", [summary])
        save_json(summary_seed_dir / "summary.json", summary)
        save_csv(summary_seed_dir / "summary.csv", [summary])

        if args.save_best:
            summary_seed_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, summary_seed_dir / "best.pt")

        print(format_metrics("  test ", test_metrics), flush=True)
        print(f"Saved seed outputs to {summary_seed_dir}", flush=True)
        return {"history": history, "summary": summary}
    finally:
        if isinstance(sys.stdout, TeeStream):
            sys.stdout.remove_stream(seed_log_handle)
        if isinstance(sys.stderr, TeeStream):
            sys.stderr.remove_stream(seed_log_handle)
        seed_log_handle.close()


def aggregate_seed_summaries(seed_summaries: list[dict[str, object]]) -> dict[str, object]:
    if not seed_summaries:
        raise ValueError("seed_summaries must not be empty")

    aggregate = {
        key: seed_summaries[0][key]
        for key in CONFIG_KEYS
        if key in seed_summaries[0] and key != "seed"
    }
    aggregate["num_seeds"] = len(seed_summaries)
    aggregate["seed_list"] = ",".join(str(int(row["seed"])) for row in seed_summaries)

    metric_keys = [
        key
        for key, value in seed_summaries[0].items()
        if key not in CONFIG_KEYS and isinstance(value, (int, float, bool))
    ]
    for key in metric_keys:
        values = [float(row[key]) for row in seed_summaries]
        aggregate[f"{key}_mean"] = statistics.fmean(values)
        aggregate[f"{key}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
    return aggregate


def run_main(args: argparse.Namespace) -> None:
    device = require_cuda_device()

    seed_list = parse_seed_list(args)
    save_run_metadata(args.run_dir, args, seed_list)
    seed_outputs = [train_single_seed(args, run_seed=seed, device=device) for seed in seed_list]
    seed_summaries = [output["summary"] for output in seed_outputs]
    aggregate_summary = aggregate_seed_summaries(seed_summaries)

    save_json(args.run_dir / "seed_summaries.json", seed_summaries)
    save_csv(args.run_dir / "seed_summaries.csv", seed_summaries)
    save_json(args.run_dir / "aggregate_summary.json", {"aggregate_summary": aggregate_summary, "seed_summaries": seed_summaries})
    save_csv(args.run_dir / "aggregate_summary.csv", [aggregate_summary])
    save_json(args.run_dir / "metrics.json", {"aggregate_summary": aggregate_summary, "seed_summaries": seed_summaries})
    save_csv(args.run_dir / "metrics.csv", [aggregate_summary])

    print("=== Aggregate Summary ===")
    print(json.dumps(aggregate_summary, indent=2, sort_keys=True))
    print(f"Saved aggregate outputs to {args.run_dir}")


def main() -> None:
    args = normalize_args(parse_args())
    args.run_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.run_dir / "run.log"

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", buffering=1) as log_handle:
        sys.stdout = TeeStream(original_stdout, log_handle)
        sys.stderr = TeeStream(original_stderr, log_handle)
        try:
            print(f"Logging stdout/stderr to {log_path}")
            run_main(args)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
