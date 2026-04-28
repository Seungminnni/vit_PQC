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

try:
    from model import COLUMN_IMAGE_CHANNELS, PairResidualColumnViT, centered_mod, circular_loss
    from secret_splits import sample_fixed_h_from_pool, split_binary_supports_balanced, support_split_summary
except ImportError:
    from idea4.model import COLUMN_IMAGE_CHANNELS, PairResidualColumnViT, centered_mod, circular_loss
    from idea4.secret_splits import sample_fixed_h_from_pool, split_binary_supports_balanced, support_split_summary


PAIR_H = 2


def parse_args():
    parser = argparse.ArgumentParser(description="Train PairResidualColumnViT on toy LWE data (h=2).")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--q", type=int, default=127)
    parser.add_argument("--num_samples", type=int, default=5000)
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
    parser.add_argument("--T", type=int, default=3)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pair_feature_mode", choices=["ordered", "symmetric"], default="symmetric")
    parser.add_argument("--residual_score_weight", type=float, default=0.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--use_pos_embed", dest="use_pos_embed", action="store_true")
    parser.add_argument("--no_pos_embed", dest="use_pos_embed", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss_marginal_weight", type=float, default=0.0)
    parser.add_argument("--loss_rec_weight", type=float, default=0.0)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--save_best", action="store_true")
    parser.add_argument("--log_interval", type=int, default=1)
    parser.set_defaults(use_pos_embed=True)
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
    if support_pool is None:
        s = sample_fixed_h(num_samples, args.n, PAIR_H)
    else:
        s = sample_fixed_h_from_pool(num_samples, args.n, support_pool)
    s = s.unsqueeze(2)
    e = torch.round(torch.randn(num_samples, args.M, 1) * args.sigma_e)
    b = (torch.bmm(A, s) + e) % args.q
    b = b.squeeze(2)
    labels = s.squeeze(2)
    return TensorDataset(A, b, labels)


def build_pair_lookup(pair_indices, n, device):
    lookup = torch.full((n, n), -1, dtype=torch.long, device=device)
    idx = torch.arange(pair_indices.size(0), device=device)
    lookup[pair_indices[:, 0], pair_indices[:, 1]] = idx
    return lookup


def labels_to_pair_index(labels, pair_lookup):
    pair = labels.topk(2, dim=1).indices
    pair = torch.sort(pair, dim=1).values
    return pair_lookup[pair[:, 0], pair[:, 1]]


def predict_pair_scores(pair_scores, pair_indices, n):
    pred_idx = pair_scores.argmax(dim=1)
    pairs = pair_indices[pred_idx]
    preds = torch.zeros((pair_scores.size(0), n), device=pair_scores.device, dtype=pair_scores.dtype)
    preds.scatter_(1, pairs, 1.0)
    return pred_idx, preds


def compute_metrics(pair_scores, pair_labels, labels, pair_indices, p_marginal, A, b, q):
    pred_idx, preds = predict_pair_scores(pair_scores, pair_indices, labels.size(1))

    pair_exact = (pred_idx == pair_labels).float().mean().item()
    coord_acc = (preds == labels).float().mean().item()
    hamming_distance = (preds != labels).float().sum(dim=1).mean().item()

    tp = (preds * labels).sum(dim=1)
    pred_pos = preds.sum(dim=1)
    true_pos = labels.sum(dim=1)
    support_precision = (tp / pred_pos.clamp(min=1.0)).mean().item()
    support_recall = (tp / true_pos.clamp(min=1.0)).mean().item()

    b_hat_soft = torch.bmm(A, p_marginal.unsqueeze(2)).squeeze(2)
    soft_residual = centered_mod(b - b_hat_soft, q)
    soft_residual_abs_mean = soft_residual.abs().mean().item()
    soft_circular_loss = circular_loss(soft_residual, q).item()

    b_hat_hard = torch.bmm(A, preds.unsqueeze(2)).squeeze(2)
    hard_residual = centered_mod(b - b_hat_hard, q)
    hard_residual_abs_mean = hard_residual.abs().mean().item()
    hard_circular_loss = circular_loss(hard_residual, q).item()

    return {
        "pair_exact": pair_exact,
        "coord_acc": coord_acc,
        "support_precision": support_precision,
        "support_recall": support_recall,
        "hamming_distance": hamming_distance,
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
    run_name = (
        f"pair_residual_colvit_{timestamp}_{split_tag}_"
        f"n{args.n}_M{args.M}_q{args.q}_h{PAIR_H}_sig{args.sigma_e}_"
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
        train_pool, val_pool, test_pool = split_binary_supports_balanced(
            args.n,
            PAIR_H,
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


def run_epoch(loader, training, model, pair_lookup, optimizer, args, device):
    model.train(training)
    total_samples = 0
    total_loss = 0.0
    total_loss_ce = 0.0
    total_loss_marginal = 0.0
    total_loss_rec = 0.0
    totals = {
        "pair_exact": 0.0,
        "coord_acc": 0.0,
        "support_precision": 0.0,
        "support_recall": 0.0,
        "hamming_distance": 0.0,
        "soft_residual_abs_mean": 0.0,
        "soft_circular_loss": 0.0,
        "hard_residual_abs_mean": 0.0,
        "hard_circular_loss": 0.0,
    }

    ce_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    bce_loss = nn.BCELoss()

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for A, b, labels in loader:
            A = A.to(device)
            b = b.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            pair_scores, pair_indices, p_marginal = model(A, b)
            pair_labels = labels_to_pair_index(labels, pair_lookup)
            loss_ce = ce_loss(pair_scores, pair_labels)

            if args.loss_marginal_weight > 0:
                loss_marginal = bce_loss(p_marginal, labels)
            else:
                loss_marginal = torch.tensor(0.0, device=device)

            if args.loss_rec_weight > 0:
                b_hat = torch.bmm(A, p_marginal.unsqueeze(2)).squeeze(2)
                residual = centered_mod(b - b_hat, args.q)
                loss_rec = circular_loss(residual, args.q)
            else:
                loss_rec = torch.tensor(0.0, device=device)

            loss = (
                loss_ce
                + args.loss_marginal_weight * loss_marginal
                + args.loss_rec_weight * loss_rec
            )

            if training:
                loss.backward()
                optimizer.step()

            metrics = compute_metrics(
                pair_scores,
                pair_labels,
                labels,
                pair_indices,
                p_marginal,
                A,
                b,
                args.q,
            )

            batch_size = labels.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_loss_ce += loss_ce.item() * batch_size
            total_loss_marginal += loss_marginal.item() * batch_size
            total_loss_rec += loss_rec.item() * batch_size
            for key in totals:
                totals[key] += metrics[key] * batch_size

    results = {
        "loss": total_loss / total_samples,
        "loss_ce": total_loss_ce / total_samples,
        "loss_marginal": total_loss_marginal / total_samples,
        "loss_rec": total_loss_rec / total_samples,
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

    model = PairResidualColumnViT(
        M=args.M,
        n=args.n,
        q=args.q,
        T=args.T,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        dropout=args.dropout,
        pair_feature_mode=args.pair_feature_mode,
        residual_score_weight=args.residual_score_weight,
        use_pos_embed=args.use_pos_embed,
        h_prior=PAIR_H,
    ).to(device)

    pair_lookup = build_pair_lookup(model.pair_indices, args.n, device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    csv_columns = [
        "epoch",
        "split",
        "loss",
        "loss_ce",
        "loss_marginal",
        "loss_rec",
        "pair_exact",
        "coord_acc",
        "support_precision",
        "support_recall",
        "hamming_distance",
        "soft_residual_abs_mean",
        "soft_circular_loss",
        "hard_residual_abs_mean",
        "hard_circular_loss",
    ]

    best_epoch = None
    best_train_metrics = None
    best_val_metrics = None
    best_state_dict = None
    early_stop_count = 0
    best_for_early_stop = None

    with open(metrics_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()

        for epoch in range(args.epochs):
            train_metrics = run_epoch(train_loader, True, model, pair_lookup, optimizer, args, device)
            val_metrics = run_epoch(val_loader, False, model, pair_lookup, optimizer, args, device)

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
                f"train_pair={train_metrics['pair_exact']*100:.2f}% "
                f"train_coord={train_metrics['coord_acc']*100:.2f}% "
                f"train_hamm={train_metrics['hamming_distance']:.2f} "
                f"train_prec={train_metrics['support_precision']*100:.2f}% "
                f"train_rec={train_metrics['support_recall']*100:.2f}% "
                f"train_soft_res={train_metrics['soft_residual_abs_mean']:.3f} "
                f"train_soft_circ={train_metrics['soft_circular_loss']:.3f} "
                f"train_hard_res={train_metrics['hard_residual_abs_mean']:.3f} "
                f"train_hard_circ={train_metrics['hard_circular_loss']:.3f} | "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_pair={val_metrics['pair_exact']*100:.2f}% "
                f"val_coord={val_metrics['coord_acc']*100:.2f}% "
                f"val_hamm={val_metrics['hamming_distance']:.2f} "
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
            if best_val_metrics is None:
                update_best = True
            else:
                if val_metrics["pair_exact"] > best_val_metrics["pair_exact"]:
                    update_best = True
                elif val_metrics["pair_exact"] == best_val_metrics["pair_exact"]:
                    update_best = (
                        val_metrics["hard_residual_abs_mean"]
                        < best_val_metrics["hard_residual_abs_mean"]
                    )

            if update_best:
                best_epoch = epoch + 1
                best_train_metrics = to_serializable_dict(train_metrics)
                best_val_metrics = to_serializable_dict(val_metrics)
                best_state_dict = copy.deepcopy(model.state_dict())
                best_payload = {
                    "best_epoch": best_epoch,
                    "best_train_metrics": best_train_metrics,
                    "best_val_metrics": best_val_metrics,
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
                            "val_metrics": best_val_metrics,
                        },
                        best_model_path,
                    )

            if args.early_stop_patience > 0:
                current_value = val_metrics["pair_exact"]
                if best_for_early_stop is None or current_value > best_for_early_stop + args.min_delta:
                    best_for_early_stop = current_value
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                    if early_stop_count >= args.early_stop_patience:
                        stop_line = (
                            f"[EARLY STOP] epoch={epoch+1} "
                            f"best_pair={best_for_early_stop*100:.2f}%"
                        )
                        print(stop_line)
                        write_log(log_path, stop_line)
                        break

    if best_epoch is not None:
        model.load_state_dict(best_state_dict)
        final_test_metrics = run_epoch(test_loader, False, model, pair_lookup, optimizer, args, device)
        final_test_metrics = to_serializable_dict(final_test_metrics)
        with open(metrics_path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            writer.writerow(flatten_metrics(best_epoch, "test", final_test_metrics))
        with open(best_metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_epoch": best_epoch,
                    "best_train_metrics": best_train_metrics,
                    "best_val_metrics": best_val_metrics,
                    "final_test_metrics": final_test_metrics,
                    "args": vars(args),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        best_line = (
            f"[BEST] epoch={best_epoch} "
            f"val_pair={best_val_metrics['pair_exact']*100:.2f}% "
            f"val_hard_res={best_val_metrics['hard_residual_abs_mean']:.3f} "
            f"val_hard_circ={best_val_metrics['hard_circular_loss']:.3f} | "
            f"final_test_pair={final_test_metrics['pair_exact']*100:.2f}% "
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
