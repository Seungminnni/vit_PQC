import argparse
import csv
import gc
import html
import io
import json
import math
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from PIL import Image
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

from data import DatasetSpec, SyntheticLWEDataset
from models import build_model
from train import apply_row_permutation, model_input


MODEL_LABELS = {
    "hybrid": "CNN+Transformer",
    "cnn": "CNN",
    "alexnet": "AlexNet",
    "resnet": "ResNet",
}

REP_LABELS = {
    "phase6_coord_input": "Phase6 input",
    "cnn_coord_embeddings": "CNN coordinate",
    "coord_embeddings": "Transformer coordinate",
    "model_coord_embeddings": "Model coordinate",
}

BLUE = "#2563eb"
RED = "#dc2626"
BLUE_X = "#1d4ed8"
RED_X = "#991b1b"
EDGE = (0.0, 0.0, 0.0, 0.58)


def format_slug_number(value: float) -> str:
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:g}".replace(".", "p")


def figure_slug(run: str, title: str) -> str:
    if "alexnet" in run.lower() or "AlexNet" in title:
        model_slug = "AlexNet"
    elif "resnet" in run.lower() or "ResNet" in title:
        model_slug = "ResNet"
    elif ("cnn" in run.lower() or re.search(r"\bCNN\b", title)) and "CNN+Transformer" not in title:
        model_slug = "CNN"
    else:
        model_slug = "Hy"
    n_match = re.search(r"n\s*=\s*(\d+)", title)
    sigma_match = re.search(r"(?:sigma|sig|\u03c3)\s*=\s*([0-9.]+)", title)
    if n_match and sigma_match:
        return f"{model_slug}_n{n_match.group(1)}_sig{format_slug_number(float(sigma_match.group(1)))}"
    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Common coordinate-level 2D/3D t-SNE exporter for embedding figures.",
        allow_abbrev=False,
    )
    parser.add_argument("--plot-result-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "result")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories under --plot-result-dir.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--png-dpi", type=int, default=240)
    parser.add_argument("--gif-frames", type=int, default=36)
    parser.add_argument("--gif-duration-ms", type=int, default=90)
    parser.add_argument("--dims", nargs="+", choices=("2d", "3d"), default=["2d", "3d"])
    parser.add_argument("--no-gif", action="store_true", help="Write 3D HTML/CSV but skip GIF rendering.")
    parser.add_argument(
        "--reuse-phase6-from",
        type=Path,
        default=None,
        help=(
            "Optional existing result directory whose phase6_coord_input_tsne(.csv) and "
            "tsne3d/phase6_coord_input_tsne3d.csv should be reused. Useful for models "
            "trained on the same H2 dataset."
        ),
    )
    parser.add_argument(
        "--force-tsne",
        action="store_true",
        help="Recompute t-SNE CSVs even if they already exist.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def fmt_num(value: object) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def choose_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def output_dir(result_dir: Path, run: str, seed: int) -> Path:
    return result_dir / f"{run}_seed{seed}"


def load_run_config(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    payload = json.loads(config_path.read_text())
    run_args = dict(payload.get("args", payload))
    if "internal_encoding" in run_args:
        run_args["encoding"] = run_args["internal_encoding"]
    return payload, run_args


def load_run_config_for_seed(run_dir: Path, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if (run_dir / "config.json").exists():
        return load_run_config(run_dir)

    seed_config = run_dir / f"seed_{seed}" / "config.json"
    if not seed_config.exists():
        raise FileNotFoundError(f"Missing config.json in {run_dir} or {seed_config.parent}")
    payload = json.loads(seed_config.read_text())
    run_args = dict(payload.get("args", payload))
    if "internal_encoding" in run_args:
        run_args["encoding"] = run_args["internal_encoding"]
    return payload, run_args


def build_test_dataset(run_args: dict[str, Any], seed: int) -> tuple[SyntheticLWEDataset, int, int]:
    dataset_seed = seed * 1000 + 37
    row_permutation_seed = dataset_seed + 101
    spec = DatasetSpec(
        num_samples=int(run_args["num_test"]),
        m=int(run_args["m"]),
        n=int(run_args["n"]),
        q=int(run_args["q"]),
        secret_type=str(run_args["secret_type"]),
        h_setting=str(run_args["h_setting"]),
        p_nonzero=run_args.get("p_nonzero"),
        fixed_h=run_args.get("fixed_h"),
        h_min=run_args.get("h_min"),
        h_max=run_args.get("h_max"),
        sigma=float(run_args["sigma"]),
        noise_distribution=str(run_args.get("noise_distribution", "discrete_gaussian")),
        noise_bound=run_args.get("noise_bound"),
        encoding=str(run_args["encoding"]),
        shared_a=bool(run_args.get("shared_a", False)),
        seed=dataset_seed,
    )
    dataset = SyntheticLWEDataset(spec)
    apply_row_permutation(
        dataset,
        mode=str(run_args.get("row_permutation", "none")),
        seed=row_permutation_seed,
    )
    return dataset, dataset_seed, row_permutation_seed


def extract_state_dict(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model", "best_state"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(value, torch.Tensor) for value in payload.values()):
            return payload
    raise TypeError("Checkpoint payload does not contain a recognizable state dict")


def predict_from_logits(logits: torch.Tensor, secret_type: str) -> tuple[torch.Tensor, torch.Tensor]:
    if secret_type != "binary":
        raise ValueError(f"Only binary secret_type is supported, got {secret_type}")
    probs = torch.softmax(logits, dim=-1)
    pred_class = probs.argmax(dim=-1)
    return pred_class.to(torch.int64), probs[..., 1]


def model_label(run_args: dict[str, Any]) -> str:
    return MODEL_LABELS.get(str(run_args["model"]), str(run_args["model"]))


def title_from_config(run_args: dict[str, Any]) -> str:
    return (
        f"{model_label(run_args)} | n = {int(run_args['n'])} | "
        f"h = {fmt_num(run_args.get('fixed_h'))} | \u03c3 = {fmt_num(run_args.get('sigma'))}"
    )


def infer_float32_dim(path: Path, row_count: int) -> int:
    size = path.stat().st_size
    if size % row_count != 0:
        raise ValueError(f"Cannot infer dim for {path}: size={size}, rows={row_count}")
    bytes_per_row = size // row_count
    if bytes_per_row % 4 != 0:
        raise ValueError(f"Cannot infer float32 dim for {path}: bytes_per_row={bytes_per_row}")
    return bytes_per_row // 4


def cache_paths(out_dir: Path) -> dict[str, Path]:
    cache_dir = out_dir / "cache"
    return {
        "cache_dir": cache_dir,
        "phase6_coord_input": cache_dir / "phase6_coord_input.float32",
        "cnn_coord_embeddings": cache_dir / "cnn_coord_embeddings.float32",
        "coord_embeddings": cache_dir / "coord_embeddings.float32",
        "model_coord_embeddings": cache_dir / "model_coord_embeddings.float32",
        "labels": cache_dir / "labels.int8",
        "preds": cache_dir / "preds.int8",
    }


def build_run_model(run_args: dict[str, Any], dataset, device: torch.device) -> torch.nn.Module:
    return build_model(
        model_name=str(run_args["model"]),
        in_channels=dataset.in_channels,
        m=dataset.m,
        n=dataset.n,
        secret_type=str(run_args["secret_type"]),
        embed_dim=int(run_args.get("embed_dim", 128)),
        depth=int(run_args.get("depth", 3)),
        num_heads=int(run_args.get("num_heads", 4)),
        dropout=float(run_args.get("dropout", 0.1)),
    ).to(device)


def load_checkpoint_flexible(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(payload)
    model.load_state_dict(state_dict, strict=True)
    return model


@torch.no_grad()
def forward_representations(model: torch.nn.Module, inputs: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    class_name = model.__class__.__name__

    if class_name == "Hybrid":
        cnn_stem = model.stem(inputs)
        batch_size, embed_dim, height, width = cnn_stem.shape
        tokens = cnn_stem.permute(0, 3, 1, 2).reshape(batch_size, width, embed_dim * height)
        coord_projected = model.coord_proj(tokens)
        transformer_input = coord_projected + model.pos_embed
        coord_embeddings = model.norm(model.transformer(transformer_input))
        logits = model.head(coord_embeddings)
        return {
            "cnn_coord_embeddings": cnn_stem.mean(dim=2).permute(0, 2, 1),
            "coord_embeddings": coord_embeddings,
        }, logits

    if class_name == "CNN":
        z = model.encoder(inputs)
        pooled = z.mean(dim=2)
        logits = model.head(model.dropout(pooled)).permute(0, 2, 1)
        return {"model_coord_embeddings": pooled.permute(0, 2, 1)}, logits

    if class_name == "AlexNet":
        z = model.features(inputs)
        pooled = z.mean(dim=2)
        logits = model.head(model.dropout(pooled)).permute(0, 2, 1)
        return {"model_coord_embeddings": pooled.permute(0, 2, 1)}, logits

    if class_name == "ResNet":
        z = model.stem(inputs)
        z = model.layer1(z)
        z = model.layer2(z)
        z = model.layer3(z)
        pooled = z.mean(dim=2)
        logits = model.head(model.dropout(pooled)).permute(0, 2, 1)
        return {"model_coord_embeddings": pooled.permute(0, 2, 1)}, logits

    raise TypeError(f"Unsupported model class for coordinate embeddings: {class_name}")


def representation_order(run_args: dict[str, Any]) -> list[tuple[str, str]]:
    if str(run_args["model"]) == "hybrid":
        return [
            ("phase6_coord_input", "Phase6 input"),
            ("cnn_coord_embeddings", "CNN coordinate"),
            ("coord_embeddings", "Transformer coordinate"),
        ]
    return [
        ("phase6_coord_input", "Phase6 input"),
        ("model_coord_embeddings", f"{model_label(run_args)} coordinate"),
    ]


def load_feature_cache(
    out_dir: Path,
    reps: list[tuple[str, str]],
    point_count: int,
) -> tuple[dict[str, np.memmap], np.ndarray, np.ndarray, np.ndarray] | None:
    paths = cache_paths(out_dir)
    needed = [name for name, _title in reps]
    if not all(paths[name].exists() for name in [*needed, "labels", "preds"]):
        return None
    features = {
        name: np.memmap(paths[name], dtype=np.float32, mode="r", shape=(point_count, infer_float32_dim(paths[name], point_count)))
        for name in needed
    }
    labels = np.memmap(paths["labels"], dtype=np.int8, mode="r", shape=(point_count,))
    preds = np.memmap(paths["preds"], dtype=np.int8, mode="r", shape=(point_count,))
    correct = np.asarray(labels) == np.asarray(preds)
    return features, np.asarray(labels), np.asarray(preds), correct


def extract_feature_cache(
    run: str,
    seed: int,
    run_args: dict[str, Any],
    plot_result_dir: Path,
    out_dir: Path,
    reps: list[tuple[str, str]],
    num_samples: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, np.memmap], np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    n = int(run_args["n"])
    m = int(run_args["m"])
    point_count = num_samples * n
    cached = load_feature_cache(out_dir, reps, point_count)
    if cached is not None:
        print(f"{run}: using feature cache", flush=True)
        features, labels, preds, correct = cached
        return features, labels, preds, correct, {
            "n": n,
            "m": m,
            "point_count": point_count,
            "dataset_seed": seed * 1000 + 37,
            "row_permutation_seed": seed * 1000 + 37 + 101,
        }

    print(f"{run}: extracting coordinate feature cache", flush=True)
    paths = cache_paths(out_dir)
    paths["cache_dir"].mkdir(parents=True, exist_ok=True)
    run_dir = plot_result_dir / run
    dataset, dataset_seed, row_permutation_seed = build_test_dataset(run_args, seed)
    model = build_run_model(run_args, dataset, device)
    model = load_checkpoint_flexible(model, run_dir / f"seed_{seed}" / "best.pt", device)
    model.eval()

    memmaps: dict[str, np.memmap] = {}
    labels = np.memmap(paths["labels"], dtype=np.int8, mode="w+", shape=(point_count,))
    preds = np.memmap(paths["preds"], dtype=np.int8, mode="w+", shape=(point_count,))
    rep_names = {name for name, _title in reps}
    loader = DataLoader(
        Subset(dataset, list(range(num_samples))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    offset = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            inputs = model_input(model, batch, device=device)
            rep_tensors, logits = forward_representations(model, inputs)
            if "phase6_coord_input" in rep_names:
                rep_tensors["phase6_coord_input"] = inputs.permute(0, 3, 1, 2).reshape(inputs.shape[0], n, -1)
            pred_secret, _ = predict_from_logits(logits.detach().cpu(), str(run_args["secret_type"]))
            secret = batch["secret"].detach().cpu().to(torch.int64)
            bsz = int(secret.shape[0])
            rows = bsz * n
            for name in rep_names:
                rep = rep_tensors[name].detach().cpu().to(torch.float32)
                if name not in memmaps:
                    memmaps[name] = np.memmap(paths[name], dtype=np.float32, mode="w+", shape=(point_count, int(rep.shape[-1])))
                memmaps[name][offset : offset + rows] = rep.reshape(rows, -1).numpy()
            labels[offset : offset + rows] = secret.reshape(-1).numpy().astype(np.int8)
            preds[offset : offset + rows] = pred_secret.detach().cpu().reshape(-1).numpy().astype(np.int8)
            offset += rows
            if batch_idx == 0 or (batch_idx + 1) % 10 == 0:
                print(f"{run}: extracted batch {batch_idx + 1:04d}, points {offset}/{point_count}", flush=True)
            del inputs, rep_tensors, logits, pred_secret, secret

    for item in [*memmaps.values(), labels, preds]:
        item.flush()
    del model, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    correct = np.asarray(labels) == np.asarray(preds)
    return memmaps, np.asarray(labels), np.asarray(preds), correct, {
        "n": n,
        "m": m,
        "point_count": point_count,
        "dataset_seed": dataset_seed,
        "row_permutation_seed": row_permutation_seed,
    }


def pca_reduce(name: str, x: np.ndarray, n_components: int, random_state: int) -> np.ndarray:
    n_components = min(n_components, x.shape[1], x.shape[0] - 1)
    if x.shape[1] <= n_components:
        return np.asarray(x, dtype=np.float32)
    if x.shape[1] > 1024:
        print(f"{name}: IncrementalPCA {x.shape[1]} -> {n_components}", flush=True)
        pca = IncrementalPCA(n_components=n_components, batch_size=2048)
        for start in range(0, x.shape[0], 2048):
            pca.partial_fit(np.asarray(x[start : start + 2048], dtype=np.float32))
        reduced = np.empty((x.shape[0], n_components), dtype=np.float32)
        for start in range(0, x.shape[0], 2048):
            reduced[start : start + 2048] = pca.transform(
                np.asarray(x[start : start + 2048], dtype=np.float32)
            ).astype(np.float32)
        return reduced
    print(f"{name}: PCA {x.shape[1]} -> {n_components}", flush=True)
    return PCA(n_components=n_components, random_state=random_state).fit_transform(
        np.asarray(x, dtype=np.float32)
    ).astype(np.float32)


def write_2d_csv(path: Path, coords: np.ndarray, labels: np.ndarray, preds: np.ndarray, correct: np.ndarray, n: int) -> None:
    sample_indices = np.repeat(np.arange(coords.shape[0] // n), n)
    coord_indices = np.tile(np.arange(n), coords.shape[0] // n)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("point_kind", "selected_dataset_index", "coordinate", "true_label", "pred_label", "correct", "tsne_x", "tsne_y"),
        )
        writer.writeheader()
        for idx in range(coords.shape[0]):
            writer.writerow(
                {
                    "point_kind": "coordinate",
                    "selected_dataset_index": int(sample_indices[idx]),
                    "coordinate": int(coord_indices[idx]),
                    "true_label": int(labels[idx]),
                    "pred_label": int(preds[idx]),
                    "correct": bool(correct[idx]),
                    "tsne_x": float(coords[idx, 0]),
                    "tsne_y": float(coords[idx, 1]),
                }
            )


def write_3d_csv(path: Path, coords: np.ndarray, labels: np.ndarray, preds: np.ndarray, correct: np.ndarray, n: int) -> None:
    sample_indices = np.repeat(np.arange(coords.shape[0] // n), n)
    coord_indices = np.tile(np.arange(n), coords.shape[0] // n)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "point_kind",
                "selected_dataset_index",
                "coordinate",
                "true_label",
                "pred_label",
                "correct",
                "tsne_x",
                "tsne_y",
                "tsne_z",
            ),
        )
        writer.writeheader()
        for idx in range(coords.shape[0]):
            writer.writerow(
                {
                    "point_kind": "coordinate",
                    "selected_dataset_index": int(sample_indices[idx]),
                    "coordinate": int(coord_indices[idx]),
                    "true_label": int(labels[idx]),
                    "pred_label": int(preds[idx]),
                    "correct": bool(correct[idx]),
                    "tsne_x": float(coords[idx, 0]),
                    "tsne_y": float(coords[idx, 1]),
                    "tsne_z": float(coords[idx, 2]),
                }
            )


def load_csv_coords(path: Path, dims: int) -> np.ndarray:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if dims == 2:
                rows.append((float(row["tsne_x"]), float(row["tsne_y"])))
            else:
                rows.append((float(row["tsne_x"]), float(row["tsne_y"]), float(row["tsne_z"])))
    return np.asarray(rows, dtype=np.float32)


def maybe_reuse_phase6(
    reuse_dir: Path | None,
    out_dir: Path,
    rep_name: str,
    dims: int,
    labels: np.ndarray,
    preds: np.ndarray,
    correct: np.ndarray,
    n: int,
) -> np.ndarray | None:
    if rep_name != "phase6_coord_input" or reuse_dir is None:
        return None
    src = reuse_dir / "phase6_coord_input_tsne.csv" if dims == 2 else reuse_dir / "tsne3d" / "phase6_coord_input_tsne3d.csv"
    if not src.exists():
        return None
    coords = load_csv_coords(src, dims)
    dst = out_dir / "phase6_coord_input_tsne.csv" if dims == 2 else out_dir / "tsne3d" / "phase6_coord_input_tsne3d.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dims == 2:
        write_2d_csv(dst, coords, labels, preds, correct, n)
    else:
        write_3d_csv(dst, coords, labels, preds, correct, n)
    return coords


def run_tsne_for_rep(
    run: str,
    rep_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    correct: np.ndarray,
    n: int,
    out_dir: Path,
    dims: int,
    args: argparse.Namespace,
) -> np.ndarray:
    reused = maybe_reuse_phase6(args.reuse_phase6_from, out_dir, rep_name, dims, labels, preds, correct, n)
    if reused is not None and not args.force_tsne:
        print(f"{run} {rep_name} {dims}D: reused {args.reuse_phase6_from}", flush=True)
        return reused

    suffix = "tsne" if dims == 2 else "tsne3d"
    csv_dir = out_dir if dims == 2 else out_dir / "tsne3d"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{rep_name}_{suffix}.csv"
    if csv_path.exists() and not args.force_tsne:
        print(f"{run} {rep_name} {dims}D: using existing {csv_path.name}", flush=True)
        return load_csv_coords(csv_path, dims)

    x_in = pca_reduce(f"{run} {rep_name} {dims}D", features, args.pca_dim, args.random_state)
    perplexity = min(float(args.perplexity), max(1.0, (x_in.shape[0] - 1) / 3.0))
    print(f"{run} {rep_name} {dims}D: TSNE input={x_in.shape}, perplexity={perplexity}", flush=True)
    coords = TSNE(
        n_components=dims,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.random_state,
        verbose=1,
        n_jobs=1,
    ).fit_transform(x_in).astype(np.float32)
    if dims == 2:
        write_2d_csv(csv_path, coords, labels, preds, correct, n)
    else:
        write_3d_csv(csv_path, coords, labels, preds, correct, n)
    return coords


def draw_interleaved_2d(axis, coords: np.ndarray, labels: np.ndarray, correct: np.ndarray) -> None:
    point_size = 5.6
    point_lw = 0.22
    x_size = 20.0
    x_lw = 0.95
    chunk = 450
    red_ok = np.flatnonzero((labels == 1) & correct)
    blue_ok = np.flatnonzero((labels == 0) & correct)
    layers = max(math.ceil(len(red_ok) / chunk), math.ceil(len(blue_ok) / chunk))
    for layer in range(layers):
        ordered = (
            ((red_ok, RED, 0.76), (blue_ok, BLUE, 0.58))
            if layer % 2 == 0
            else ((blue_ok, BLUE, 0.58), (red_ok, RED, 0.76))
        )
        for idxs, color, alpha in ordered:
            part = idxs[layer * chunk : (layer + 1) * chunk]
            if part.size:
                axis.scatter(
                    coords[part, 0],
                    coords[part, 1],
                    c=color,
                    s=point_size,
                    alpha=alpha,
                    marker="o",
                    edgecolors=[EDGE],
                    linewidths=point_lw,
                    rasterized=True,
                )
    for label, color in ((0, BLUE_X), (1, RED_X)):
        bad = np.flatnonzero((labels == label) & (~correct))
        if bad.size:
            axis.scatter(
                coords[bad, 0],
                coords[bad, 1],
                c=color,
                s=x_size,
                alpha=0.98,
                marker="x",
                linewidths=x_lw,
                rasterized=True,
            )


def legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=BLUE, markeredgecolor="#111111", markeredgewidth=0.7, markersize=7.0, label="true secret label 0"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=RED, markeredgecolor="#111111", markeredgewidth=0.7, markersize=7.0, label="true secret label 1"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="#111111", markeredgewidth=0.9, markersize=7.0, label="correct"),
        Line2D([0], [0], marker="x", linestyle="none", color="#111111", markeredgewidth=1.2, markersize=8.0, label="incorrect"),
    ]


def style_axis_2d(axis, show_ylabel: bool) -> None:
    axis.set_xlabel("t-SNE 1", fontsize=14)
    axis.set_ylabel("t-SNE 2" if show_ylabel else "", fontsize=14)
    axis.tick_params(axis="both", labelsize=11)
    axis.grid(True, alpha=0.16, linewidth=0.65)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#111111")


def save_2d_figures(
    run: str,
    seed: int,
    title: str,
    reps: list[tuple[str, str]],
    coords_by_name: dict[str, np.ndarray],
    labels: np.ndarray,
    correct: np.ndarray,
    out_dir: Path,
    png_dpi: int,
) -> list[Path]:
    outputs = []
    for name, panel_title in reps:
        fig, axis = plt.subplots(figsize=(7.0, 6.0), constrained_layout=False)
        draw_interleaved_2d(axis, coords_by_name[name], labels, correct)
        axis.set_title(f"{panel_title} t-SNE", fontsize=17, pad=10)
        style_axis_2d(axis, show_ylabel=True)
        fig.legend(handles=legend_handles(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.02), fontsize=12, frameon=True)
        fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.20)
        out = out_dir / f"{name}_tsne.png"
        fig.savefig(out, dpi=png_dpi, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)
        print(f"{run}: wrote {out}", flush=True)

    width = 6.3 * len(reps)
    fig, axes = plt.subplots(1, len(reps), figsize=(width, 5.8), constrained_layout=False)
    if len(reps) == 1:
        axes = [axes]
    for idx, ((name, panel_title), axis) in enumerate(zip(reps, axes)):
        draw_interleaved_2d(axis, coords_by_name[name], labels, correct)
        axis.set_title(panel_title, fontsize=17, pad=10)
        style_axis_2d(axis, show_ylabel=(idx == 0))
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.02), fontsize=12, frameon=True, columnspacing=2.0, handletextpad=0.6, borderpad=0.6)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.88, bottom=0.20, wspace=0.16)
    combined = out_dir / f"{figure_slug(run, title)}.png"
    fig.savefig(combined, dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)
    outputs.append(combined)
    print(f"{run}: wrote {combined}", flush=True)
    return outputs


def normalize_for_view(coords: np.ndarray) -> np.ndarray:
    centered = coords - coords.mean(axis=0, keepdims=True)
    scale = float(np.percentile(np.linalg.norm(centered, axis=1), 98.0))
    if not math.isfinite(scale) or scale <= 0:
        scale = float(np.max(np.abs(centered))) or 1.0
    return (centered / scale).astype(np.float32)


def round_nested(coords: np.ndarray) -> list[list[float]]:
    return np.round(normalize_for_view(coords), 5).tolist()


def build_interactive_html(
    title: str,
    reps: list[tuple[str, str]],
    coords_by_name: dict[str, np.ndarray],
    labels: np.ndarray,
    correct: np.ndarray,
) -> str:
    panels = [{"name": name, "title": panel_title, "coords": round_nested(coords_by_name[name])} for name, panel_title in reps]
    payload = {
        "title": f"{title} | 3D t-SNE",
        "labels": labels.astype(int).tolist(),
        "correct": correct.astype(int).tolist(),
        "panels": panels,
    }
    data_json = json.dumps(payload, separators=(",", ":"))
    escaped_title = html.escape(payload["title"])
    grid_cols = len(reps)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{escaped_title}</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #ffffff; color: #111827; }}
    .wrap {{ padding: 12px 14px 16px; }}
    .top {{ display: flex; gap: 14px; align-items: baseline; flex-wrap: wrap; }}
    h2 {{ margin: 0 0 4px; font-size: 18px; font-weight: 600; }}
    button {{ font-size: 12px; padding: 3px 8px; }}
    .meta {{ font-size: 12px; color: #4b5563; }}
    .grid {{ display: grid; grid-template-columns: repeat({grid_cols}, minmax(280px, 1fr)); gap: 10px; margin-top: 8px; }}
    .panel {{ border: 1px solid #d1d5db; border-radius: 5px; padding: 7px; background: #fff; }}
    .title {{ font-weight: 600; font-size: 13px; margin: 0 0 5px; }}
    canvas {{ width: 100%; height: 560px; display: block; background: #ffffff; cursor: grab; }}
    canvas:active {{ cursor: grabbing; }}
    .legend {{ margin-top: 7px; display: flex; gap: 12px; align-items: center; font-size: 12px; flex-wrap: wrap; }}
    .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; border: 1px solid #111; vertical-align: middle; margin-right: 4px; }}
    .blue {{ background: {BLUE}; }}
    .red {{ background: {RED}; }}
    .xmark {{ font-weight: 700; color: #111; padding-right: 2px; }}
  </style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <h2>{escaped_title}</h2>
    <button id="reset">Reset view</button>
    <div class="meta">Drag to rotate all panels. Wheel to zoom.</div>
  </div>
  <div class="legend">
    <span><span class="dot blue"></span>true secret label 0</span>
    <span><span class="dot red"></span>true secret label 1</span>
    <span>circle = correct coordinate</span>
    <span><span class="xmark">x</span> = incorrect coordinate</span>
  </div>
  <div id="grid" class="grid"></div>
</div>
<script>
const DATA = {data_json};
const grid = document.getElementById("grid");
let rotX = -0.55, rotY = 0.72, zoom = 1.0;
let dragging = false, lastX = 0, lastY = 0, raf = null;
const panels = [];
function color(label, alpha, bad=false) {{
  if (bad) return label === 1 ? `rgba(153,27,27,${{alpha}})` : `rgba(29,78,216,${{alpha}})`;
  return label === 1 ? `rgba(220,38,38,${{alpha}})` : `rgba(37,99,235,${{alpha}})`;
}}
for (const panel of DATA.panels) {{
  const div = document.createElement("div");
  div.className = "panel";
  div.innerHTML = `<div class="title">${{panel.title}}</div><canvas></canvas>`;
  grid.appendChild(div);
  const canvas = div.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  panels.push({{canvas, ctx, panel}});
  canvas.addEventListener("mousedown", e => {{ dragging = true; lastX = e.clientX; lastY = e.clientY; }});
  canvas.addEventListener("wheel", e => {{ e.preventDefault(); zoom *= Math.exp(-e.deltaY * 0.001); zoom = Math.max(0.35, Math.min(4.0, zoom)); schedule(); }}, {{passive:false}});
}}
window.addEventListener("mouseup", () => dragging = false);
window.addEventListener("mousemove", e => {{
  if (!dragging) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  rotY += dx * 0.008;
  rotX += dy * 0.008;
  schedule();
}});
window.addEventListener("resize", schedule);
document.getElementById("reset").onclick = () => {{ rotX = -0.55; rotY = 0.72; zoom = 1.0; schedule(); }};
function schedule() {{
  if (raf !== null) return;
  raf = requestAnimationFrame(() => {{ raf = null; drawAll(); }});
}}
function resize(canvas) {{
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(300, Math.floor(rect.width * dpr));
  const h = Math.max(360, Math.floor(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) {{ canvas.width = w; canvas.height = h; }}
  return [w, h, dpr];
}}
function projectPoint(p, w, h) {{
  let x = p[0], y = p[1], z = p[2];
  const cx = Math.cos(rotX), sx = Math.sin(rotX);
  const cy = Math.cos(rotY), sy = Math.sin(rotY);
  let y1 = y * cx - z * sx;
  let z1 = y * sx + z * cx;
  let x2 = x * cy + z1 * sy;
  let z2 = -x * sy + z1 * cy;
  const perspective = 1.9 / (2.35 - z2);
  const scale = Math.min(w, h) * 0.34 * zoom * perspective;
  return [w / 2 + x2 * scale, h / 2 - y1 * scale, z2];
}}
function drawPanel(obj) {{
  const [w, h, dpr] = resize(obj.canvas);
  const ctx = obj.ctx;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
  const coords = obj.panel.coords;
  const order = new Array(coords.length);
  for (let i = 0; i < coords.length; i++) {{
    const pr = projectPoint(coords[i], w, h);
    order[i] = [pr[2], pr[0], pr[1], i];
  }}
  order.sort((a, b) => a[0] - b[0]);
  const r = Math.max(1.0, 1.2 * dpr);
  const xsize = Math.max(3.0, 3.9 * dpr);
  for (const item of order) {{
    const px = item[1], py = item[2], i = item[3];
    const label = DATA.labels[i];
    const ok = !!DATA.correct[i];
    if (ok) {{
      ctx.beginPath();
      ctx.fillStyle = color(label, label === 1 ? 0.78 : 0.58);
      ctx.strokeStyle = "rgba(0,0,0,0.56)";
      ctx.lineWidth = 0.35 * dpr;
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }} else {{
      ctx.lineWidth = 1.15 * dpr;
      ctx.strokeStyle = color(label, 0.98, true);
      ctx.beginPath();
      ctx.moveTo(px - xsize, py - xsize);
      ctx.lineTo(px + xsize, py + xsize);
      ctx.moveTo(px + xsize, py - xsize);
      ctx.lineTo(px - xsize, py + xsize);
      ctx.stroke();
    }}
  }}
}}
function drawAll() {{ for (const obj of panels) drawPanel(obj); }}
schedule();
</script>
</body>
</html>
"""


def rotate_project(coords: np.ndarray, rot_x: float, rot_y: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    cx = math.cos(rot_x)
    sx = math.sin(rot_x)
    cy = math.cos(rot_y)
    sy = math.sin(rot_y)
    y1 = y * cx - z * sx
    z1 = y * sx + z * cx
    x2 = x * cy + z1 * sy
    z2 = -x * sy + z1 * cy
    perspective = 1.9 / (2.35 - z2)
    return x2 * perspective, y1 * perspective, z2


def draw_gif_panel(axis, coords: np.ndarray, labels: np.ndarray, correct: np.ndarray, panel_title: str, rot_y: float) -> None:
    norm = normalize_for_view(coords)
    px, py, depth = rotate_project(norm, rot_x=-0.50, rot_y=rot_y)
    order = np.argsort(depth)
    px = px[order]
    py = py[order]
    labs = labels[order]
    ok = correct[order]
    for label, color, alpha in ((1, RED, 0.74), (0, BLUE, 0.56)):
        mask = (labs == label) & ok
        if mask.any():
            axis.scatter(px[mask], py[mask], c=color, s=3.8, alpha=alpha, marker="o", edgecolors=(0, 0, 0, 0.48), linewidths=0.16, rasterized=True)
    for label, color in ((0, BLUE_X), (1, RED_X)):
        mask = (labs == label) & (~ok)
        if mask.any():
            axis.scatter(px[mask], py[mask], c=color, s=14.0, alpha=0.95, marker="x", linewidths=0.7, rasterized=True)
    axis.set_title(panel_title, fontsize=10, pad=4)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlim(-1.25, 1.25)
    axis.set_ylim(-1.18, 1.18)
    axis.grid(True, alpha=0.10, linewidth=0.45)
    for spine in axis.spines.values():
        spine.set_color("#111111")
        spine.set_linewidth(0.75)


def make_rotation_gif(
    run: str,
    seed: int,
    title: str,
    reps: list[tuple[str, str]],
    coords_by_name: dict[str, np.ndarray],
    labels: np.ndarray,
    correct: np.ndarray,
    out_dir: Path,
    frame_count: int,
    duration_ms: int,
) -> Path:
    gif_path = out_dir / "tsne3d" / f"{run}_seed{seed}_tsne3d_rotation.gif"
    frames: list[Image.Image] = []
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=BLUE, markeredgecolor="#111111", markeredgewidth=0.5, markersize=4.0, label="label 0"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=RED, markeredgecolor="#111111", markeredgewidth=0.5, markersize=4.0, label="label 1"),
        Line2D([0], [0], marker="x", linestyle="none", color="#111111", markeredgewidth=0.8, markersize=5.0, label="incorrect"),
    ]
    width = 3.9 * len(reps)
    for frame_idx in range(frame_count):
        rot_y = 2.0 * math.pi * frame_idx / frame_count
        fig, axes = plt.subplots(1, len(reps), figsize=(width, 3.85), constrained_layout=True)
        if len(reps) == 1:
            axes = [axes]
        fig.suptitle(f"{title} | 3D t-SNE rotation", fontsize=13)
        for (name, panel_title), axis in zip(reps, axes):
            draw_gif_panel(axis, coords_by_name[name], labels, correct, panel_title, rot_y)
        axes[-1].legend(handles=handles, loc="lower right", fontsize=6.3, frameon=True)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=120)
        plt.close(fig)
        buffer.seek(0)
        frames.append(Image.open(buffer).convert("P", palette=Image.Palette.ADAPTIVE).copy())
        if frame_idx == 0 or (frame_idx + 1) % 12 == 0 or frame_idx + 1 == frame_count:
            print(f"{run}: gif frame {frame_idx + 1}/{frame_count}", flush=True)
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=True)
    print(f"{run}: wrote {gif_path}", flush=True)
    return gif_path


def process_run(run: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    print(f"\n=== {run} seed {args.seed} ===", flush=True)
    out_dir = output_dir(args.result_dir, run, args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_payload, run_args = load_run_config_for_seed(args.plot_result_dir / run, args.seed)
    if run_args.get("encoding") != "phase6":
        raise ValueError(f"{run}: expected encoding=phase6, got {run_args.get('encoding')!r}")

    reps = representation_order(run_args)
    title = title_from_config(run_args)
    extract_reps = reps
    if args.reuse_phase6_from is not None:
        extract_reps = [(name, panel_title) for name, panel_title in reps if name != "phase6_coord_input"]
        if not extract_reps:
            extract_reps = reps

    features, labels, preds, correct, meta = extract_feature_cache(
        run=run,
        seed=args.seed,
        run_args=run_args,
        plot_result_dir=args.plot_result_dir,
        out_dir=out_dir,
        reps=extract_reps,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=device,
    )
    n = int(meta["n"])
    point_count = int(meta["point_count"])
    exact_rate = float(np.mean([bool((labels[i : i + n] == preds[i : i + n]).all()) for i in range(0, point_count, n)]))
    coord_acc = float(np.mean(correct))
    print(f"{run}: exact_rate={exact_rate:.4f}, coord_acc={coord_acc:.4f}", flush=True)

    outputs: dict[str, list[str]] = {"two_d": [], "three_d": []}
    coords_2d: dict[str, np.ndarray] = {}
    if "2d" in args.dims:
        for name, _title in reps:
            if name in features:
                coords_2d[name] = run_tsne_for_rep(run, name, features[name], labels, preds, correct, n, out_dir, 2, args)
            else:
                reused = maybe_reuse_phase6(args.reuse_phase6_from, out_dir, name, 2, labels, preds, correct, n)
                if reused is None:
                    raise RuntimeError(f"{run}: missing features for {name} and no reusable phase6 t-SNE was found")
                coords_2d[name] = reused
        two_d_paths = save_2d_figures(run, args.seed, title, reps, coords_2d, labels, correct, out_dir, args.png_dpi)
        outputs["two_d"] = [str(path.relative_to(out_dir)) for path in two_d_paths]

    html_path = None
    gif_path = None
    coords_3d: dict[str, np.ndarray] = {}
    if "3d" in args.dims:
        for name, _title in reps:
            if name in features:
                coords_3d[name] = run_tsne_for_rep(run, name, features[name], labels, preds, correct, n, out_dir, 3, args)
            else:
                reused = maybe_reuse_phase6(args.reuse_phase6_from, out_dir, name, 3, labels, preds, correct, n)
                if reused is None:
                    raise RuntimeError(f"{run}: missing features for {name} and no reusable phase6 t-SNE was found")
                coords_3d[name] = reused
        tsne3d_dir = out_dir / "tsne3d"
        html_path = tsne3d_dir / f"{run}_seed{args.seed}_tsne3d_interactive.html"
        html_path.write_text(build_interactive_html(title, reps, coords_3d, labels, correct), encoding="utf-8")
        print(f"{run}: wrote {html_path}", flush=True)
        outputs["three_d"].append(str(html_path.relative_to(out_dir)))
        for name, _title in reps:
            outputs["three_d"].append(str((tsne3d_dir / f"{name}_tsne3d.csv").relative_to(out_dir)))
        if not args.no_gif:
            gif_path = make_rotation_gif(run, args.seed, title, reps, coords_3d, labels, correct, out_dir, args.gif_frames, args.gif_duration_ms)
            outputs["three_d"].append(str(gif_path.relative_to(out_dir)))

    summary = {
        "run": run,
        "seed": args.seed,
        "title": title,
        "num_samples": args.num_samples,
        "n": n,
        "point_count": point_count,
        "exact_rate": exact_rate,
        "coord_acc": coord_acc,
        "dataset_seed": int(meta["dataset_seed"]),
        "row_permutation_seed": int(meta["row_permutation_seed"]),
        "config_run_name": run_args.get("run_name"),
        "model": run_args.get("model"),
        "representations": [name for name, _title in reps],
        "outputs": outputs,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if html_path is not None:
        (out_dir / "tsne3d" / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    gc.collect()
    return summary


def main() -> None:
    args = parse_args()
    args.plot_result_dir = args.plot_result_dir.resolve()
    args.result_dir = args.result_dir.resolve()
    if args.reuse_phase6_from is not None and not args.reuse_phase6_from.is_absolute():
        args.reuse_phase6_from = (args.result_dir / args.reuse_phase6_from).resolve()
    device = choose_device(args.device)
    summaries = [process_run(run, args, device) for run in args.runs]
    summary_stem = "_".join(args.runs)
    if len(summary_stem) > 120:
        summary_stem = f"{args.runs[0]}_to_{args.runs[-1]}_{len(args.runs)}runs"
    summary_path = args.result_dir / f"{summary_stem}_seed{args.seed}_embedding_tsne_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
