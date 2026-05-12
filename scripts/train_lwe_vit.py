from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lwe_vit import (  # noqa: E402
    LWEDatasetSpec,
    LWEParams,
    LWEViTConfig,
    LWEViTForSecret,
    RepresentationConfig,
    SyntheticLWEDataset,
    batch_statistics,
    dataset_statistics,
    finalize_statistics,
    merge_statistics,
    num_secret_classes,
    residual_consistency_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train relation-preserving LWE ViT secret recovery model.")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--q", type=int, default=257)
    parser.add_argument("--secret-dist", "--secret_dist", dest="secret_dist", choices=["binary", "ternary", "uniform"], default="binary")
    parser.add_argument("--noise-dist", "--noise_dist", dest="noise_dist", choices=["zero", "uniform_small", "discrete_gaussian"], default="discrete_gaussian")
    parser.add_argument("--noise-width", "--noise_width", dest="noise_width", type=float, default=1.0)
    parser.add_argument("--representation", choices=["relation_grid", "phase_grid", "row_equation_tokens"], default="relation_grid")
    parser.add_argument("--no-phase", dest="use_phase", action="store_false")
    parser.set_defaults(use_phase=True)
    parser.add_argument("--no-broadcast-b", dest="broadcast_b", action="store_false")
    parser.set_defaults(broadcast_b=True)
    parser.add_argument("--no-rhs-column", dest="add_rhs_column", action="store_false")
    parser.set_defaults(add_rhs_column=True)
    parser.add_argument("--patch-rows", "--patch_rows", dest="patch_rows", type=int, default=4)
    parser.add_argument("--patch-cols", "--patch_cols", dest="patch_cols", type=int, default=4)
    parser.add_argument("--h-setting", "--h_setting", dest="h_setting", choices=["iid", "fixed_h", "variable_h", "bernoulli"], default="variable_h")
    parser.add_argument("--fixed-h", "--fixed_h", dest="fixed_h", type=int, default=None)
    parser.add_argument("--h-min", "--h_min", dest="h_min", type=int, default=None)
    parser.add_argument("--h-max", "--h_max", dest="h_max", type=int, default=None)
    parser.add_argument("--p-nonzero", "--p_nonzero", dest="p_nonzero", type=float, default=None)
    parser.add_argument("--num-train", "--num_train", dest="num_train", type=int, default=4096)
    parser.add_argument("--num-val", "--num_val", dest="num_val", type=int, default=1024)
    parser.add_argument("--num-test", "--num_test", dest="num_test", type=int, default=1024)
    parser.add_argument("--train-eval-samples", "--train_eval_samples", dest="train_eval_samples", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", "--embed_dim", dest="embed_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num-heads", "--num_heads", dest="num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--class-weight-mode", "--class_weight_mode", dest="class_weight_mode", choices=["none", "inverse_prior"], default="inverse_prior")
    parser.add_argument("--residual-loss-weight", "--residual_loss_weight", dest="residual_loss_weight", type=float, default=0.05)
    parser.add_argument("--residual-success-factor", "--residual_success_factor", dest="residual_success_factor", type=float, default=2.0)
    parser.add_argument("--warmup-epochs", "--warmup_epochs", dest="warmup_epochs", type=int, default=0)
    parser.add_argument("--early-stopping-patience", "--early_stopping_patience", dest="early_stopping_patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, default=ROOT / "runs" / "lwe_vit")
    parser.add_argument("--run-name", "--run_name", dest="run_name", default=None)
    parser.add_argument("--save-best", "--save_best", dest="save_best", action="store_true")
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.m = args.m if args.m is not None else 8 * args.n
    if args.secret_dist != "binary" and args.h_setting != "iid":
        args.h_setting = "iid"
    if args.h_setting == "variable_h":
        args.h_min = 1 if args.h_min is None else args.h_min
        args.h_max = max(2, args.n // 8) if args.h_max is None else args.h_max
    if args.h_setting == "bernoulli" and args.p_nonzero is None:
        args.p_nonzero = min(1.0, 3.0 / max(args.n, 1))
    if args.representation == "row_equation_tokens" and args.patch_cols == 4:
        args.patch_cols = 1
    if args.run_name is None:
        args.run_name = (
            f"{args.representation}_{args.secret_dist}_n{args.n}_m{args.m}_q{args.q}_"
            f"noise{args.noise_width:g}_{args.h_setting}_seed{args.seed}"
        )
    args.run_dir = args.output_dir / args.run_name
    return args


def seed_list(args: argparse.Namespace) -> list[int]:
    if args.seeds is None:
        return [args.seed]
    return [int(part.strip()) for part in args.seeds.split(",") if part.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def flatten_prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def make_dataset(args: argparse.Namespace, run_seed: int, num_samples: int, offset: int) -> SyntheticLWEDataset:
    params = LWEParams(
        n=args.n,
        m=args.m,
        q=args.q,
        secret_dist=args.secret_dist,
        noise_dist=args.noise_dist,
        seed=run_seed * 1000 + offset,
        noise_width=args.noise_width,
    )
    rep = RepresentationConfig(
        name=args.representation,
        patch_rows=args.patch_rows,
        patch_cols=args.patch_cols,
        use_phase=args.use_phase,
        broadcast_b=args.broadcast_b,
        add_rhs_column=args.add_rhs_column,
    )
    return SyntheticLWEDataset(
        LWEDatasetSpec(
            num_samples=num_samples,
            params=params,
            representation=rep,
            h_setting=args.h_setting,
            p_nonzero=args.p_nonzero,
            fixed_h=args.fixed_h,
            h_min=args.h_min,
            h_max=args.h_max,
        )
    )


def make_train_eval_dataset(dataset, sample_count: int):
    if sample_count == 0 or sample_count >= len(dataset):
        return dataset
    if sample_count < 0:
        raise ValueError("train_eval_samples must be non-negative.")
    return Subset(dataset, list(range(sample_count)))


def build_class_weights(train_stats: dict[str, float], num_classes: int, mode: str, device: torch.device) -> tuple[torch.Tensor | None, list[float]]:
    if mode == "none":
        return None, [1.0 for _ in range(num_classes)]
    if mode != "inverse_prior":
        raise ValueError(f"Unknown class_weight_mode: {mode}")
    probs = torch.tensor([train_stats[f"class_prob_{idx}"] for idx in range(num_classes)], dtype=torch.float32, device=device)
    weights = (1.0 / probs.clamp(min=1e-6)).clamp(max=100.0)
    weights = weights / weights.mean()
    return weights, weights.detach().cpu().tolist()


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def epoch_lr(base_lr: float, epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0 or epoch > warmup_epochs:
        return base_lr
    return base_lr * epoch / warmup_epochs


def metric_score(metrics: dict[str, float]) -> float:
    return metrics["exact_match"] + metrics["support_f1"]


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    ordered = [
        "coord_acc",
        "exact_match",
        "support_f1",
        "support_precision",
        "support_recall",
        "macro_f1",
        "pred_residual_std_mean",
        "pred_residual_std_gap_mean",
        "residual_success_rate",
    ]
    body = " ".join(f"{key}={metrics[key]:.4f}" for key in ordered if key in metrics)
    return f"{prefix} {body}"


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    class_weights: torch.Tensor | None,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, float, float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_ce = 0.0
    total_residual = 0.0
    total_samples = 0
    stat_list: list[dict[str, float]] = []
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            secret = batch["secret"].to(device, non_blocking=True)
            A = batch["A"].to(device, non_blocking=True)
            b = batch["b"].to(device, non_blocking=True)
            oracle_residual = batch["oracle_residual"].to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            out = model(image, mask)
            logits = out.s_logits
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), weight=class_weights)
            residual_loss = residual_consistency_loss(
                A=A,
                b=b,
                s_logits=logits,
                q=args.q,
                secret_dist=args.secret_dist,
                noise_bound=args.noise_width * args.residual_success_factor,
            )
            loss = ce + args.residual_loss_weight * residual_loss

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = target.shape[0]
            total_loss += loss.item() * batch_size
            total_ce += ce.item() * batch_size
            total_residual += residual_loss.item() * batch_size
            total_samples += batch_size
            stat_list.append(
                batch_statistics(
                    logits=logits.detach(),
                    target_labels=target,
                    secret=secret,
                    A=A,
                    b=b,
                    oracle_residual=oracle_residual,
                    q=args.q,
                    noise_width=args.noise_width,
                    residual_success_factor=args.residual_success_factor,
                    secret_dist=args.secret_dist,
                )
            )

    merged = merge_statistics(stat_list)
    metrics = finalize_statistics(merged, num_classes=logits.shape[-1])
    denom = max(total_samples, 1)
    return total_loss / denom, total_ce / denom, total_residual / denom, metrics


def base_config(args: argparse.Namespace, run_seed: int, num_parameters: int, class_weight_list: list[float]) -> dict[str, object]:
    row = {
        "run_name": args.run_name,
        "seed": run_seed,
        "n": args.n,
        "m": args.m,
        "q": args.q,
        "secret_dist": args.secret_dist,
        "noise_dist": args.noise_dist,
        "noise_width": args.noise_width,
        "representation": args.representation,
        "use_phase": args.use_phase,
        "broadcast_b": args.broadcast_b,
        "add_rhs_column": args.add_rhs_column,
        "patch_rows": args.patch_rows,
        "patch_cols": args.patch_cols,
        "h_setting": args.h_setting,
        "fixed_h": args.fixed_h,
        "h_min": args.h_min,
        "h_max": args.h_max,
        "p_nonzero": args.p_nonzero,
        "num_train": args.num_train,
        "num_val": args.num_val,
        "num_test": args.num_test,
        "train_eval_samples": args.train_eval_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "class_weight_mode": args.class_weight_mode,
        "residual_loss_weight": args.residual_loss_weight,
        "residual_success_factor": args.residual_success_factor,
        "num_parameters": num_parameters,
    }
    for idx, weight in enumerate(class_weight_list):
        row[f"class_weight_{idx}"] = weight
    return row


def train_single_seed(args: argparse.Namespace, run_seed: int, device: torch.device) -> dict[str, object]:
    set_seed(run_seed)
    seed_dir = args.run_dir / f"seed_{run_seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = make_dataset(args, run_seed, args.num_train, 11)
    val_dataset = make_dataset(args, run_seed, args.num_val, 23)
    test_dataset = make_dataset(args, run_seed, args.num_test, 37)
    train_stats = dataset_statistics(train_dataset)
    val_stats = dataset_statistics(val_dataset)
    test_stats = dataset_statistics(test_dataset)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    train_eval_loader = DataLoader(make_train_eval_dataset(train_dataset, args.train_eval_samples), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    num_classes = num_secret_classes(args.secret_dist, q=args.q)
    model = LWEViTForSecret(
        LWEViTConfig(
            n=args.n,
            q=args.q,
            in_channels=train_dataset.in_channels,
            num_secret_classes=num_classes,
            patch_rows=args.patch_rows,
            patch_cols=args.patch_cols,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            dropout=args.dropout,
        )
    ).to(device)
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    class_weights, class_weight_list = build_class_weights(train_stats, num_classes, args.class_weight_mode, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    config = base_config(args, run_seed, num_parameters, class_weight_list)
    save_json(seed_dir / "config.json", config)

    print("=== Seed Run ===")
    print(json.dumps(config, indent=2, sort_keys=True))
    print(
        f"[Baseline] random_guess_coord_acc={train_stats['random_guess_coord_acc']:.4f} "
        f"random_guess_exact_match={train_stats['random_guess_exact_match']:.6f} "
        f"all_zero_coord_acc={train_stats['all_zero_coord_acc']:.4f} "
        f"avg_h={train_stats['avg_h']:.2f} nonzero_rate={train_stats['nonzero_rate']:.4f}"
    )

    history: list[dict[str, object]] = []
    best_state = None
    best_val_score = float("-inf")
    best_epoch = 0
    best_val_metrics: dict[str, float] | None = None
    best_train_metrics: dict[str, float] | None = None
    best_train_loss = float("nan")
    best_train_eval_loss = float("nan")
    epochs_without_improvement = 0
    early_stopped = False
    stopped_epoch = args.epochs
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        current_lr = epoch_lr(args.lr, epoch, args.warmup_epochs)
        set_optimizer_lr(optimizer, current_lr)
        train_loss, train_ce, train_res, _ = run_epoch(model, train_loader, optimizer, class_weights, args, device)
        train_eval_loss, train_eval_ce, train_eval_res, train_metrics = run_epoch(model, train_eval_loader, None, class_weights, args, device)
        val_loss, val_ce, val_res, val_metrics = run_epoch(model, val_loader, None, class_weights, args, device)
        score = metric_score(val_metrics)
        epoch_time_sec = time.perf_counter() - epoch_start

        record = {
            **config,
            "epoch": epoch,
            "lr": current_lr,
            "epoch_time_sec": epoch_time_sec,
            "train_loss": train_loss,
            "train_ce_loss": train_ce,
            "train_residual_loss": train_res,
            "train_eval_loss": train_eval_loss,
            "train_eval_ce_loss": train_eval_ce,
            "train_eval_residual_loss": train_eval_res,
            "val_loss": val_loss,
            "val_ce_loss": val_ce,
            "val_residual_loss": val_res,
            **flatten_prefixed("train_metric", train_metrics),
            **flatten_prefixed("val_metric", val_metrics),
        }
        history.append(record)

        if score > best_val_score:
            best_val_score = score
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            best_train_metrics = dict(train_metrics)
            best_train_loss = train_loss
            best_train_eval_loss = train_eval_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Seed {run_seed} | Epoch {epoch:03d}/{args.epochs} lr={current_lr:.6g} "
            f"train_loss={train_loss:.4f} train_eval_loss={train_eval_loss:.4f} "
            f"val_loss={val_loss:.4f} epoch_time_sec={epoch_time_sec:.2f}"
        )
        print(format_metrics("  train_eval", train_metrics))
        print(format_metrics("  val       ", val_metrics))

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            early_stopped = True
            stopped_epoch = epoch
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is None or best_val_metrics is None or best_train_metrics is None:
        raise RuntimeError("Training did not produce a best checkpoint.")

    model.load_state_dict(best_state)
    test_loss, test_ce, test_res, test_metrics = run_epoch(model, test_loader, None, class_weights, args, device)
    summary = {
        **config,
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "stopped_epoch": stopped_epoch,
        "early_stopped": early_stopped,
        "train_time_sec": time.perf_counter() - start,
        "train_loss": best_train_loss,
        "train_eval_loss": best_train_eval_loss,
        "test_loss": test_loss,
        "test_ce_loss": test_ce,
        "test_residual_loss": test_res,
        "train_metric_exact_match": best_train_metrics["exact_match"],
        "train_metric_support_f1": best_train_metrics["support_f1"],
        **flatten_prefixed("train_data", train_stats),
        **flatten_prefixed("val_data", val_stats),
        **flatten_prefixed("test_data", test_stats),
        **flatten_prefixed("best_val_metric", best_val_metrics),
        **flatten_prefixed("test_metric", test_metrics),
    }
    save_json(seed_dir / "history.json", history)
    save_csv(seed_dir / "history.csv", history)
    save_json(seed_dir / "metrics.json", summary)
    save_csv(seed_dir / "metrics.csv", [summary])
    save_json(seed_dir / "summary.json", summary)
    save_csv(seed_dir / "summary.csv", [summary])
    if args.save_best:
        torch.save(best_state, seed_dir / "best.pt")
    print(format_metrics("  test      ", test_metrics))
    print(f"Saved seed outputs to {seed_dir}")
    return {"history": history, "summary": summary}


def aggregate_summaries(summaries: list[dict[str, object]]) -> dict[str, object]:
    aggregate = {
        "run_name": summaries[0]["run_name"],
        "num_seeds": len(summaries),
        "seed_list": ",".join(str(summary["seed"]) for summary in summaries),
    }
    metric_keys = [
        key for key, value in summaries[0].items() if isinstance(value, (int, float, bool)) and key not in {"seed"}
    ]
    for key in metric_keys:
        values = [float(summary[key]) for summary in summaries]
        aggregate[f"{key}_mean"] = statistics.fmean(values)
        aggregate[f"{key}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
    return aggregate


def main() -> None:
    args = normalize_args(parse_args())
    args.run_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        args.run_dir / "config.json",
        {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items() if key != "run_dir"},
    )
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    outputs = [train_single_seed(args, run_seed=run_seed, device=device) for run_seed in seed_list(args)]
    summaries = [output["summary"] for output in outputs]
    aggregate = aggregate_summaries(summaries)
    save_json(args.run_dir / "seed_summaries.json", summaries)
    save_csv(args.run_dir / "seed_summaries.csv", summaries)
    save_json(args.run_dir / "aggregate_summary.json", {"aggregate_summary": aggregate, "seed_summaries": summaries})
    save_csv(args.run_dir / "aggregate_summary.csv", [aggregate])
    save_json(args.run_dir / "metrics.json", {"aggregate_summary": aggregate, "seed_summaries": summaries})
    save_csv(args.run_dir / "metrics.csv", [aggregate])
    print("=== Aggregate Summary ===")
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    print(f"Saved aggregate outputs to {args.run_dir}")


if __name__ == "__main__":
    main()
