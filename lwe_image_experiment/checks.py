import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from lwe_image_experiment.data import (
        DatasetSpec,
        SyntheticLWEDataset,
        centered_mod,
    )
    from lwe_image_experiment.metrics import batch_statistics, centered_residual, finalize_statistics
    from lwe_image_experiment.models import build_model
    from lwe_image_experiment.train import normalize_args, require_cuda_device, train_single_seed
else:
    from .data import DatasetSpec, SyntheticLWEDataset, centered_mod
    from .metrics import batch_statistics, centered_residual, finalize_statistics
    from .models import build_model
    from .train import normalize_args, require_cuda_device, train_single_seed


def make_args(**overrides):
    base = dict(
        secret_type="binary",
        encoding="phase6",
        model="hybrid",
        n=8,
        m=64,
        q=257,
        sigma=1.0,
        noise_distribution="discrete_gaussian",
        noise_bound=None,
        row_permutation="none",
        h_setting="variable_h",
        fixed_h=None,
        h_min=1,
        h_max=2,
        p_nonzero=None,
        shared_a=False,
        num_train=64,
        num_val=32,
        num_test=32,
        train_eval_samples=4096,
        epochs=2,
        warmup_epochs=0,
        early_stopping_patience=0,
        batch_size=16,
        lr=3e-4,
        weight_decay=1e-4,
        embed_dim=32,
        depth=2,
        num_heads=4,
        dropout=0.1,
        class_weight_mode="inverse_prior",
        residual_success_factor=2.0,
        seed=0,
        seeds=None,
        num_workers=0,
        run_name="audit_run",
        output_dir=Path("audit_runs"),
        save_dir=None,
        save_best=False,
    )
    base.update(overrides)
    return normalize_args(SimpleNamespace(**base))


def hash_tensor(tensor: torch.Tensor) -> bytes:
    return tensor.cpu().numpy().tobytes()


def shape_check() -> dict[str, object]:
    dataset = SyntheticLWEDataset(
        DatasetSpec(
            num_samples=8,
            m=64,
            n=8,
            q=257,
            secret_type="binary",
            h_setting="variable_h",
            p_nonzero=None,
            fixed_h=None,
            h_min=1,
            h_max=2,
            sigma=1.0,
            noise_distribution="discrete_gaussian",
            noise_bound=None,
            encoding="phase6",
            shared_a=False,
            seed=123,
        )
    )
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))
    image_shape = tuple(batch["image"].shape)
    target_shape = tuple(batch["target"].shape)

    models_to_check = [
        ("cnn", "phase6"),
        ("alexnet", "phase6"),
        ("resnet", "phase6"),
        ("hybrid", "phase6"),
    ]
    logits_shapes: dict[str, tuple[int, ...]] = {}
    for model_name, encoding in models_to_check:
        local_dataset = SyntheticLWEDataset(
            DatasetSpec(
                num_samples=8,
                m=64,
                n=8,
                q=257,
                secret_type="binary",
                h_setting="variable_h",
                p_nonzero=None,
                fixed_h=None,
                h_min=1,
                h_max=2,
                sigma=1.0,
                noise_distribution="discrete_gaussian",
                noise_bound=None,
                encoding=encoding,
                shared_a=False,
                seed=100 + len(logits_shapes),
            )
        )
        local_batch = next(iter(DataLoader(local_dataset, batch_size=4, shuffle=False)))
        model = build_model(
            model_name=model_name,
            in_channels=local_dataset.in_channels,
            m=local_dataset.m,
            n=local_dataset.n,
            secret_type="binary",
            embed_dim=32,
            depth=2,
            num_heads=4,
            dropout=0.1,
        )
        logits_shapes[model_name] = tuple(model(local_batch["image"]).shape)

    assert image_shape == (4, 6, 64, 8)
    assert target_shape == (4, 8)
    for shape in logits_shapes.values():
        assert shape == (4, 8, 2)

    return {
        "image_shape": image_shape,
        "target_shape": target_shape,
        "logits_shapes": logits_shapes,
    }


def arithmetic_check() -> dict[str, object]:
    dataset = SyntheticLWEDataset(
        DatasetSpec(
            num_samples=4,
            m=32,
            n=8,
            q=257,
            secret_type="binary",
            h_setting="variable_h",
            p_nonzero=None,
            fixed_h=None,
            h_min=1,
            h_max=3,
            sigma=1.0,
            noise_distribution="discrete_gaussian",
            noise_bound=None,
            encoding="phase6",
            shared_a=False,
            seed=777,
        )
    )
    for idx in range(len(dataset)):
        item = dataset[idx]
        lhs = item["b"]
        rhs = torch.remainder(torch.matmul(item["A"], item["secret"]) + item["noise"], dataset.q)
        assert torch.equal(lhs, rhs)
    return {"checked_samples": len(dataset)}


def centered_mod_check() -> dict[str, object]:
    x = torch.tensor([-260, -129, -128, -1, 0, 1, 128, 129, 260], dtype=torch.int64)
    centered = centered_mod(x, 257)
    assert centered.min().item() >= -(257 // 2)
    assert centered.max().item() <= 257 // 2

    dataset = SyntheticLWEDataset(
        DatasetSpec(
            num_samples=2,
            m=16,
            n=8,
            q=257,
            secret_type="binary",
            h_setting="fixed_h",
            p_nonzero=None,
            fixed_h=1,
            h_min=None,
            h_max=None,
            sigma=0.0,
            noise_distribution="discrete_gaussian",
            noise_bound=None,
            encoding="phase6",
            shared_a=False,
            seed=3,
        )
    )
    item = dataset[0]
    residual = centered_residual(item["A"].unsqueeze(0), item["b"].unsqueeze(0), item["secret"].unsqueeze(0), dataset.q)
    assert residual.eq(0).all().item()
    return {"centered_values": centered.tolist()}


def metric_semantics_check() -> dict[str, object]:
    secret = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 0],
        ],
        dtype=torch.int64,
    )
    pred = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=torch.int64,
    )
    logits = torch.full((*pred.shape, 2), -5.0)
    logits.scatter_(-1, pred.unsqueeze(-1), 5.0)

    A = torch.zeros((3, 4, 4), dtype=torch.int64)
    b = torch.zeros((3, 4), dtype=torch.int64)
    oracle_residual = torch.zeros((3, 4), dtype=torch.int64)
    stats = batch_statistics(
        logits=logits,
        secret=secret,
        A=A,
        b=b,
        oracle_residual=oracle_residual,
        q=257,
        sigma=0.0,
        residual_success_factor=2.0,
        secret_type="binary",
    )
    metrics = finalize_statistics(stats, secret_type="binary")

    assert metrics["exact_match_direct"] == 1.0 / 3.0
    assert metrics["exact_match"] == metrics["exact_match_direct"]
    assert metrics["support_exact_match"] == 1.0 / 3.0
    assert stats["support_tp"] == 2.0
    assert stats["support_fp"] == 1.0
    assert stats["support_fn"] == 2.0
    assert stats["support_tn"] == 7.0

    expected_micro_f1 = 4.0 / 7.0
    expected_sample_f1 = (1.0 + 0.5 + 0.0) / 3.0
    assert abs(metrics["support_f1_micro"] - expected_micro_f1) < 1e-12
    assert abs(metrics["support_f1_sample_mean"] - expected_sample_f1) < 1e-12
    assert metrics["residual_success_rate"] == 1.0

    return {
        "exact_match_direct": metrics["exact_match_direct"],
        "support_f1_micro": metrics["support_f1_micro"],
        "support_f1_sample_mean": metrics["support_f1_sample_mean"],
        "support_exact_match": metrics["support_exact_match"],
        "support_tp": stats["support_tp"],
        "support_fp": stats["support_fp"],
        "support_fn": stats["support_fn"],
        "support_tn": stats["support_tn"],
        "residual_success_rate": metrics["residual_success_rate"],
    }


def leakage_check() -> dict[str, object]:
    train_dataset = SyntheticLWEDataset(
        DatasetSpec(
            num_samples=16,
            m=32,
            n=8,
            q=257,
            secret_type="binary",
            h_setting="variable_h",
            p_nonzero=None,
            fixed_h=None,
            h_min=1,
            h_max=2,
            sigma=1.0,
            noise_distribution="discrete_gaussian",
            noise_bound=None,
            encoding="phase6",
            shared_a=False,
            seed=101,
        )
    )
    val_dataset = SyntheticLWEDataset(
        DatasetSpec(
            num_samples=16,
            m=32,
            n=8,
            q=257,
            secret_type="binary",
            h_setting="variable_h",
            p_nonzero=None,
            fixed_h=None,
            h_min=1,
            h_max=2,
            sigma=1.0,
            noise_distribution="discrete_gaussian",
            noise_bound=None,
            encoding="phase6",
            shared_a=False,
            seed=202,
        )
    )

    train_A_hashes = {hash_tensor(train_dataset._get_A(i)) for i in range(len(train_dataset))}
    val_A_hashes = {hash_tensor(val_dataset._get_A(i)) for i in range(len(val_dataset))}
    train_secret_hashes = {hash_tensor(train_dataset.secret[i]) for i in range(len(train_dataset))}
    val_secret_hashes = {hash_tensor(val_dataset.secret[i]) for i in range(len(val_dataset))}
    train_sample_hashes = {
        (hash_tensor(train_dataset._get_A(i)), hash_tensor(train_dataset.secret[i]), hash_tensor(train_dataset.b[i]))
        for i in range(len(train_dataset))
    }
    val_sample_hashes = {
        (hash_tensor(val_dataset._get_A(i)), hash_tensor(val_dataset.secret[i]), hash_tensor(val_dataset.b[i]))
        for i in range(len(val_dataset))
    }

    assert len(train_A_hashes) > 1
    assert len(train_secret_hashes) > 1
    assert train_A_hashes.isdisjoint(val_A_hashes)
    assert train_sample_hashes.isdisjoint(val_sample_hashes)
    assert train_secret_hashes != {next(iter(train_secret_hashes))}
    assert val_secret_hashes != {next(iter(val_secret_hashes))}

    return {
        "unique_train_A": len(train_A_hashes),
        "unique_train_secret": len(train_secret_hashes),
        "train_val_sample_overlap": 0,
    }


def sanity_overfit_check(device: torch.device) -> dict[str, object]:
    args = make_args(
        secret_type="binary",
        encoding="phase6",
        model="hybrid",
        n=8,
        m=128,
        q=257,
        sigma=0.0,
        h_setting="fixed_h",
        fixed_h=1,
        num_train=128,
        num_val=64,
        num_test=64,
        epochs=12,
        batch_size=32,
        embed_dim=64,
        depth=2,
        lr=1e-3,
        class_weight_mode="none",
        run_name="sanity_fixed_h1_sigma0",
    )
    result = train_single_seed(args, run_seed=0, device=device)["summary"]
    assert result["best_val_metric_exact_match"] >= 0.7
    return {
        "best_val_exact_match": result["best_val_metric_exact_match"],
        "test_exact_match": result["test_metric_exact_match"],
    }


def harder_smoke_check(device: torch.device) -> dict[str, object]:
    binary_args = make_args(
        secret_type="binary",
        encoding="phase6",
        model="hybrid",
        n=8,
        m=64,
        q=257,
        sigma=1.0,
        h_setting="variable_h",
        h_min=1,
        h_max=2,
        num_train=64,
        num_val=32,
        num_test=32,
        epochs=2,
        batch_size=16,
        run_name="smoke_binary_variable_h",
    )
    binary_result = train_single_seed(binary_args, run_seed=0, device=device)["summary"]
    return {
        "binary_test_coord_acc": binary_result["test_metric_coord_acc"],
    }


def run_all_checks(output_path: Path) -> dict[str, object]:
    device = require_cuda_device()
    report = {
        "shape_check": shape_check(),
        "arithmetic_check": arithmetic_check(),
        "centered_mod_check": centered_mod_check(),
        "metric_semantics_check": metric_semantics_check(),
        "leakage_check": leakage_check(),
        "sanity_overfit_check": sanity_overfit_check(device),
        "harder_smoke_check": harder_smoke_check(device),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run verification checks for the global LWE image experiment")
    parser.add_argument("--output", type=Path, default=Path("verification_report.json"))
    args = parser.parse_args()
    report = run_all_checks(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
