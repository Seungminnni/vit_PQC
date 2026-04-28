import argparse
import copy
import csv
import json
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset

from model import COLUMN_IMAGE_CHANNELS, RecurrentResidualColumnViT, centered_mod, circular_loss
from secret_splits import (
    sample_fixed_h_from_pool,
    split_binary_supports_balanced,
    support_split_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Residual-Feedback ColumnViT on toy LWE data.")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--M", type=int, default=64)
    parser.add_argument("--q", type=int, default=127)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--secret_split", action="store_true")
    parser.add_argument("--train_secret_fraction", type=float, default=0.8)
    parser.add_argument("--val_secret_fraction", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=1234)
    parser.add_argument("--split_trials", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--sigma_e", type=float, default=0.0)
    parser.add_argument("--secret_mode", default="fixed", choices=["fixed", "per_sample", "fixed_h"])
    parser.add_argument("--h", type=int, default=1)
    parser.add_argument(
        "--blind_h",
        action="store_true",
        help="Use h only for data generation; do not give h prior, top-h decoding, or cardinality loss to the model/eval.",
    )
    parser.add_argument("--T", type=int, default=6)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--step_size", type=float, default=0.5)
    parser.add_argument("--loss_rec_weight", type=float, default=0.1)
    parser.add_argument("--loss_sparse_weight", type=float, default=0.01)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--save_best", action="store_true")
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sample_fixed_h(num_samples, n, h, device=None):
    if h < 0 or h > n:
        raise ValueError(f"h must be in [0, n], got h={h}, n={n}")
    if h == 0:
        return torch.zeros(num_samples, n, device=device)
    scores = torch.rand(num_samples, n, device=device)
    topk = torch.topk(scores, k=h, dim=1).indices
    s = torch.zeros(num_samples, n, device=device)
    s.scatter_(1, topk, 1.0)
    return s


def build_toy_dataset(args, num_samples=None, support_pool=None):
    if num_samples is None:
        num_samples = args.num_samples

    A = torch.randint(0, args.q, (num_samples, args.M, args.n), dtype=torch.float32)
    if args.secret_mode == "fixed":
        s_fixed = torch.randint(0, 2, (1, args.n), dtype=torch.float32)
        s = s_fixed.expand(num_samples, -1).clone()
    elif args.secret_mode == "per_sample":
        s = torch.randint(0, 2, (num_samples, args.n), dtype=torch.float32)
    else:
        if support_pool is None:
            s = sample_fixed_h(num_samples, args.n, args.h)
        else:
            s = sample_fixed_h_from_pool(num_samples, args.n, support_pool)

    s = s.unsqueeze(2)
    e = torch.round(torch.randn(num_samples, args.M, 1) * args.sigma_e)
    b = (torch.bmm(A, s) + e) % args.q
    b = b.squeeze(2)
    labels = s.squeeze(2)
    return TensorDataset(A, b, labels)


def predict_support(logits, h):
    if h is None:
        return (logits > 0).float()
    if h == 0:
        return torch.zeros_like(logits)
    topk = torch.topk(logits, k=h, dim=1).indices
    preds = torch.zeros_like(logits)
    preds.scatter_(1, topk, 1.0)
    return preds


def compute_metrics(logits, labels, A, b, h, q):
    p = torch.sigmoid(logits)
    preds = predict_support(logits, h)

    coord_acc = (preds == labels).float().mean().item()
    exact_match = (preds == labels).all(dim=1).float().mean().item()
    hamming_distance = (preds != labels).float().sum(dim=1).mean().item()

    tp = (preds * labels).sum(dim=1)
    pred_pos = preds.sum(dim=1)
    true_pos = labels.sum(dim=1)
    support_precision = (tp / pred_pos.clamp(min=1.0)).mean().item()
    support_recall = (tp / true_pos.clamp(min=1.0)).mean().item()

    b_hat = torch.bmm(A, p.unsqueeze(2)).squeeze(2)
    soft_residual = centered_mod(b - b_hat, q)
    soft_residual_abs_mean = soft_residual.abs().mean().item()
    soft_circular_loss = circular_loss(soft_residual, q).item()

    b_hat_hard = torch.bmm(A, preds.unsqueeze(2)).squeeze(2)
    hard_residual = centered_mod(b - b_hat_hard, q)
    hard_residual_abs_mean = hard_residual.abs().mean().item()
    hard_circular_loss = circular_loss(hard_residual, q).item()

    return {
        "coord_acc": coord_acc,
        "exact_match": exact_match,
        "hamming_distance": hamming_distance,
        "pred_pos_mean": pred_pos.mean().item(),
        "true_pos_mean": true_pos.mean().item(),
        "support_precision": support_precision,
        "support_recall": support_recall,
        "soft_residual_abs_mean": soft_residual_abs_mean,
        "soft_circular_loss": soft_circular_loss,
        "hard_residual_abs_mean": hard_residual_abs_mean,
        "hard_circular_loss": hard_circular_loss,
    }


def to_serializable_dict(d):
    out = {}
    for key, value in d.items():
        if torch.is_tensor(value):
            value = value.item()
        else:
            try:
                value = value.item()
            except AttributeError:
                pass

        if isinstance(value, bool):
            out[key] = bool(value)
        elif isinstance(value, int):
            out[key] = int(value)
        elif isinstance(value, float):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def write_log(log_path, line):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def default_run_dir(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    split_tag = "secret_split" if args.secret_split else "sample_split"
    mode_tag = args.secret_mode
    h_tag = "blind_h" if args.blind_h else "h_aware"
    run_name = (
        f"residual_colvit_{timestamp}_{split_tag}_{mode_tag}_{h_tag}_"
        f"n{args.n}_M{args.M}_q{args.q}_h{args.h}_sig{args.sigma_e}_"
        f"T{args.T}_bs{args.batch_size}_seed{args.seed}"
    )
    return os.path.join(script_dir, "runs", run_name)


def split_sample_sizes(args):
    train_size = int(args.num_samples * args.train_fraction)
    val_size = int(args.num_samples * args.val_fraction)
    train_size = max(1, min(args.num_samples - 2, train_size))
    val_size = max(1, min(args.num_samples - train_size - 1, val_size))
    test_size = args.num_samples - train_size - val_size
    if test_size < 1:
        val_size -= 1
        test_size = 1
    return train_size, val_size, test_size


def build_train_val_test_datasets(args, log_path):
    train_size, val_size, test_size = split_sample_sizes(args)

    if args.secret_split:
        if args.secret_mode != "fixed_h":
            raise ValueError("--secret_split is only defined for --secret_mode fixed_h")
        train_pool, val_pool, test_pool = split_binary_supports_balanced(
            args.n,
            args.h,
            train_fraction=args.train_secret_fraction,
            val_fraction=args.val_secret_fraction,
            seed=args.split_seed,
            trials=args.split_trials,
        )
        split_summary = support_split_summary((train_pool, val_pool, test_pool), args.n)
        info_line = (
            f"[SECRET SPLIT] enabled balanced train_supports={len(train_pool)} "
            f"val_supports={len(val_pool)} test_supports={len(test_pool)} "
            f"overlap_tv={split_summary['train_val_overlap']} "
            f"overlap_tt={split_summary['train_test_overlap']} "
            f"overlap_vt={split_summary['val_test_overlap']} "
            f"train_coord_minmax={split_summary['train_coord_min']}/{split_summary['train_coord_max']} "
            f"val_coord_minmax={split_summary['val_coord_min']}/{split_summary['val_coord_max']} "
            f"test_coord_minmax={split_summary['test_coord_min']}/{split_summary['test_coord_max']} "
            f"train_samples={train_size} val_samples={val_size} test_samples={test_size}"
        )
        print(info_line)
        write_log(log_path, info_line)
        train_dataset = build_toy_dataset(args, num_samples=train_size, support_pool=train_pool)
        val_dataset = build_toy_dataset(args, num_samples=val_size, support_pool=val_pool)
        test_dataset = build_toy_dataset(args, num_samples=test_size, support_pool=test_pool)
        return train_dataset, val_dataset, test_dataset

    dataset = build_toy_dataset(args)
    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(args.num_samples, generator=generator)
    train_end = train_size
    val_end = train_size + val_size
    train_dataset = Subset(dataset, perm[:train_end].tolist())
    val_dataset = Subset(dataset, perm[train_end:val_end].tolist())
    test_dataset = Subset(dataset, perm[val_end:].tolist())
    info_line = (
        f"[SECRET SPLIT] disabled sample_split train_samples={len(train_dataset)} "
        f"val_samples={len(val_dataset)} test_samples={len(test_dataset)}"
    )
    print(info_line)
    write_log(log_path, info_line)
    return train_dataset, val_dataset, test_dataset


def flatten_metrics(epoch, split, metrics):
    row = {
        "epoch": epoch,
        "split": split,
    }
    row.update(metrics)
    return row


def run_epoch(loader, training, model, optimizer, criterion, args, device):
    model.train(training)
    total_samples = 0
    total_loss = 0.0
    total_loss_sup = 0.0
    total_loss_rec = 0.0
    total_loss_sparse = 0.0
    totals = {
        "coord_acc": 0.0,
        "exact_match": 0.0,
        "hamming_distance": 0.0,
        "pred_pos_mean": 0.0,
        "true_pos_mean": 0.0,
        "support_precision": 0.0,
        "support_recall": 0.0,
        "soft_residual_abs_mean": 0.0,
        "soft_circular_loss": 0.0,
        "hard_residual_abs_mean": 0.0,
        "hard_circular_loss": 0.0,
    }

    h_for_pred = args.h if args.secret_mode == "fixed_h" and not args.blind_h else None
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for A, b, labels in loader:
            A = A.to(device)
            b = b.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            logits, _ = model(A, b)
            loss_sup = criterion(logits, labels)
            p = torch.sigmoid(logits)
            b_hat = torch.bmm(A, p.unsqueeze(2)).squeeze(2)
            final_residual = centered_mod(b - b_hat, args.q)
            loss_rec = circular_loss(final_residual, args.q)

            if args.secret_mode == "fixed_h" and not args.blind_h:
                loss_sparse = ((p.sum(dim=1) - args.h) ** 2).mean()
            else:
                loss_sparse = torch.tensor(0.0, device=device)

            loss = loss_sup + args.loss_rec_weight * loss_rec + args.loss_sparse_weight * loss_sparse

            if training:
                loss.backward()
                optimizer.step()

            metrics = compute_metrics(logits, labels, A, b, h_for_pred, args.q)
            batch_size = labels.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_loss_sup += loss_sup.item() * batch_size
            total_loss_rec += loss_rec.item() * batch_size
            total_loss_sparse += loss_sparse.item() * batch_size
            for key in totals:
                totals[key] += metrics[key] * batch_size

    results = {
        "loss": total_loss / total_samples,
        "loss_sup": total_loss_sup / total_samples,
        "loss_rec": total_loss_rec / total_samples,
        "loss_sparse": total_loss_sparse / total_samples,
    }
    for key in totals:
        results[key] = totals[key] / total_samples
    return results


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.run_dir is None:
        run_dir = default_run_dir(args)
    else:
        run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)

    args_path = os.path.join(run_dir, "args.json")
    metrics_path = os.path.join(run_dir, "metrics.csv")
    log_path = os.path.join(run_dir, "train.log")
    best_metrics_path = os.path.join(run_dir, "best_metrics.json")
    best_model_path = os.path.join(run_dir, "best_model.pt")

    print(f"[INFO] Run directory: {run_dir}")
    print(f"[INFO] Metrics CSV: {metrics_path}")
    print(f"[INFO] Train log: {log_path}")
    print(f"[INFO] Column image channels: {COLUMN_IMAGE_CHANNELS}")

    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    train_dataset, val_dataset, test_dataset = build_train_val_test_datasets(args, log_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    h_prior = args.h if args.secret_mode == "fixed_h" and not args.blind_h else None
    model = RecurrentResidualColumnViT(
        M=args.M,
        n=args.n,
        q=args.q,
        T=args.T,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        step_size=args.step_size,
        h_prior=h_prior,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    csv_columns = [
        "epoch",
        "split",
        "loss",
        "loss_sup",
        "loss_rec",
        "loss_sparse",
        "coord_acc",
        "exact_match",
        "hamming_distance",
        "pred_pos_mean",
        "true_pos_mean",
        "support_precision",
        "support_recall",
        "soft_residual_abs_mean",
        "soft_circular_loss",
        "hard_residual_abs_mean",
        "hard_circular_loss",
    ]

    best_epoch = None
    best_train_metrics = None
    best_test_metrics = None

    with open(metrics_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()

        for epoch in range(args.epochs):
            train_metrics = run_epoch(train_loader, True, model, optimizer, criterion, args, device)
            val_metrics = run_epoch(val_loader, False, model, optimizer, criterion, args, device)

            writer.writerow(
                flatten_metrics(epoch + 1, "train", to_serializable_dict(train_metrics))
            )
            writer.writerow(
                flatten_metrics(epoch + 1, "val", to_serializable_dict(val_metrics))
            )
            csv_file.flush()

            is_last = (epoch + 1) == args.epochs
            should_log = is_last or (args.log_interval > 0 and (epoch + 1) % args.log_interval == 0)
            log_line = (
                f"Epoch {epoch+1:3d}/{args.epochs} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['coord_acc']*100:.2f}% "
                f"train_exact={train_metrics['exact_match']*100:.2f}% "
                f"train_hamm={train_metrics['hamming_distance']:.2f} "
                f"train_pred_pos={train_metrics['pred_pos_mean']:.2f} "
                f"train_prec={train_metrics['support_precision']*100:.2f}% "
                f"train_rec={train_metrics['support_recall']*100:.2f}% "
                f"train_soft_res={train_metrics['soft_residual_abs_mean']:.3f} "
                f"train_soft_circ={train_metrics['soft_circular_loss']:.3f} "
                f"train_hard_res={train_metrics['hard_residual_abs_mean']:.3f} "
                f"train_hard_circ={train_metrics['hard_circular_loss']:.3f} | "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['coord_acc']*100:.2f}% "
                f"val_exact={val_metrics['exact_match']*100:.2f}% "
                f"val_hamm={val_metrics['hamming_distance']:.2f} "
                f"val_pred_pos={val_metrics['pred_pos_mean']:.2f} "
                f"val_prec={val_metrics['support_precision']*100:.2f}% "
                f"val_rec={val_metrics['support_recall']*100:.2f}% "
                f"val_soft_res={val_metrics['soft_residual_abs_mean']:.3f} "
                f"val_soft_circ={val_metrics['soft_circular_loss']:.3f} "
                f"val_hard_res={val_metrics['hard_residual_abs_mean']:.3f} "
                f"val_hard_circ={val_metrics['hard_circular_loss']:.3f}"
            )

            if should_log:
                print(log_line)
                write_log(log_path, log_line)

            update_best = False
            if best_test_metrics is None:
                update_best = True
            else:
                if val_metrics["exact_match"] > best_test_metrics["exact_match"]:
                    update_best = True
                elif val_metrics["exact_match"] == best_test_metrics["exact_match"]:
                    update_best = (
                        val_metrics["hard_residual_abs_mean"]
                        < best_test_metrics["hard_residual_abs_mean"]
                    )

            if update_best:
                best_epoch = epoch + 1
                best_train_metrics = to_serializable_dict(train_metrics)
                best_test_metrics = to_serializable_dict(val_metrics)
                best_state_dict = copy.deepcopy(model.state_dict())
                best_payload = {
                    "best_epoch": best_epoch,
                    "best_train_metrics": best_train_metrics,
                    "best_val_metrics": best_test_metrics,
                    "args": vars(args),
                }
                with open(best_metrics_path, "w", encoding="utf-8") as f:
                    json.dump(best_payload, f, indent=2, ensure_ascii=False)

                if args.save_best:
                    torch.save(
                        {
                            "epoch": best_epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "args": vars(args),
                            "train_metrics": best_train_metrics,
                            "val_metrics": best_test_metrics,
                        },
                        best_model_path,
                    )

    if best_epoch is not None:
        model.load_state_dict(best_state_dict)
        final_test_metrics = run_epoch(test_loader, False, model, optimizer, criterion, args, device)
        with open(metrics_path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            writer.writerow(flatten_metrics(best_epoch, "test", to_serializable_dict(final_test_metrics)))
        final_test_metrics = to_serializable_dict(final_test_metrics)
        with open(best_metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_epoch": best_epoch,
                    "best_train_metrics": best_train_metrics,
                    "best_val_metrics": best_test_metrics,
                    "final_test_metrics": final_test_metrics,
                    "args": vars(args),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        best_line = (
            f"[BEST] epoch={best_epoch} "
            f"val_exact={best_test_metrics['exact_match']*100:.2f}% "
            f"val_hard_res={best_test_metrics['hard_residual_abs_mean']:.3f} "
            f"val_hard_circ={best_test_metrics['hard_circular_loss']:.3f} | "
            f"final_test_exact={final_test_metrics['exact_match']*100:.2f}% "
            f"final_test_hard_res={final_test_metrics['hard_residual_abs_mean']:.3f} "
            f"final_test_hard_circ={final_test_metrics['hard_circular_loss']:.3f}"
        )
        info_line = f"[INFO] Run directory: {run_dir}"
        print(best_line)
        print(info_line)
        write_log(log_path, best_line)
        write_log(log_path, info_line)


if __name__ == "__main__":
    main()
