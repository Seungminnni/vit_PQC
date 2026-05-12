import argparse
import csv
import gc
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "result"
PLOT_RESULT_DIR = ROOT / "outputs"
COLLECT_DIR = RESULT_DIR / "embedding_tsne_figures"
DPI = 240

BLUE = "#2563eb"
RED = "#dc2626"
BLUE_X = "#1d4ed8"
RED_X = "#991b1b"
EDGE = (0.0, 0.0, 0.0, 0.58)

PANEL_ORDER = [
    ("phase6_coord_input", "Phase6 input"),
    ("cnn_coord_embeddings", "CNN coordinate"),
    ("coord_embeddings", "Transformer coordinate"),
    ("model_coord_embeddings", "Model coordinate"),
]

MODEL_SLUGS = {
    "hybrid": "Hy",
    "cnn": "CNN",
    "alexnet": "AlexNet",
    "resnet": "ResNet",
}

MODEL_LABELS = {
    "hybrid": "CNN+Transformer",
    "cnn": "CNN",
    "alexnet": "AlexNet",
    "resnet": "ResNet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restyle generated coordinate t-SNE figures.")
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--plot-result-dir", type=Path, default=PLOT_RESULT_DIR)
    parser.add_argument(
        "--collect-dir",
        type=Path,
        default=None,
        help="Directory for copied short-name figures. Defaults to <result-dir>/embedding_tsne_figures.",
    )
    return parser.parse_args()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(run: str) -> dict[str, Any]:
    for path in (
        PLOT_RESULT_DIR / run / "config.json",
        PLOT_RESULT_DIR / run / "seed_0" / "config.json",
    ):
        data = load_json(path)
        if data:
            return data.get("args", data)
    return {}


def parse_title_value(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def infer_metadata(result_dir: Path) -> tuple[str, str, int, str]:
    run = result_dir.name.removesuffix("_seed0")
    summary = load_json(result_dir / "summary.json")
    config = load_config(run)

    title = str(summary.get("title") or summary.get("plot_title") or summary.get("requested_title") or "")
    model = str(config.get("model") or summary.get("model") or "").lower()
    if not model:
        if "CNN+Transformer" in title:
            model = "hybrid"
        elif "AlexNet" in title:
            model = "alexnet"
        elif "ResNet" in title:
            model = "resnet"
        elif re.search(r"\bCNN\b", title):
            model = "cnn"
        else:
            model = "model"

    n_value = config.get("n") or summary.get("n") or parse_title_value(r"n\s*=\s*(\d+)", title)
    if n_value is None:
        raise ValueError(f"{result_dir}: could not infer n")
    n = int(n_value)

    sigma_value = config.get("sigma") or parse_title_value(r"(?:sigma|sig|s|\u03c3)\s*=\s*([0-9.]+)", title)
    if sigma_value is None:
        raise ValueError(f"{result_dir}: could not infer sigma")
    sigma = format_number(float(sigma_value))

    slug_model = MODEL_SLUGS.get(model, model[:1].upper() + model[1:])
    slug = f"{slug_model}_n{n}_sig{sigma}"
    return run, model, n, slug


def format_number(value: float) -> str:
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return (f"{value:g}").replace(".", "p")


def load_tsne_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords: list[tuple[float, float]] = []
    labels: list[int] = []
    correct: list[bool] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            coords.append((float(row["tsne_x"]), float(row["tsne_y"])))
            labels.append(int(row["true_label"]))
            correct.append(row["correct"].lower() == "true")
    return np.asarray(coords, dtype=np.float32), np.asarray(labels, dtype=np.int8), np.asarray(correct, dtype=bool)


def draw_interleaved_2d(axis, coords: np.ndarray, labels: np.ndarray, correct: np.ndarray) -> None:
    point_size = 7.0
    point_lw = 0.26
    x_size = 26.0
    x_lw = 1.1
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


def legend_handles(scale: float = 1.0) -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=BLUE, markeredgecolor="#111111", markeredgewidth=0.8 * scale, markersize=9.0 * scale, label="true secret label 0"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=RED, markeredgecolor="#111111", markeredgewidth=0.8 * scale, markersize=9.0 * scale, label="true secret label 1"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="#111111", markeredgewidth=1.0 * scale, markersize=9.0 * scale, label="correct"),
        Line2D([0], [0], marker="x", linestyle="none", color="#111111", markeredgewidth=1.5 * scale, markersize=10.5 * scale, label="incorrect"),
    ]


def style_axis(axis, show_ylabel: bool) -> None:
    axis.set_xlabel("t-SNE 1", fontsize=15)
    axis.set_ylabel("t-SNE 2" if show_ylabel else "", fontsize=15)
    axis.tick_params(axis="both", labelsize=14)
    axis.grid(True, alpha=0.16, linewidth=0.65)
    for spine in axis.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#111111")


def panel_title(stem: str, default_title: str, model: str) -> str:
    if stem == "model_coord_embeddings":
        return f"{MODEL_LABELS.get(model, 'Model')} coordinate"
    return default_title


def collect_panels(result_dir: Path, model: str) -> list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]]:
    panels = []
    for stem, title in PANEL_ORDER:
        csv_path = result_dir / f"{stem}_tsne.csv"
        if not csv_path.exists():
            continue
        coords, labels, correct = load_tsne_csv(csv_path)
        panels.append((stem, panel_title(stem, title, model), coords, labels, correct))
    if not panels:
        raise ValueError(f"{result_dir}: no 2D t-SNE CSV files found")
    return panels


def save_combined(result_dir: Path) -> list[Path]:
    run, model, _n, slug = infer_metadata(result_dir)
    summary = load_json(result_dir / "summary.json")
    file_run = str(summary.get("run") or run)
    seed = int(summary.get("seed") or 0)
    panels = collect_panels(result_dir, model)
    labels = panels[0][3]
    correct = panels[0][4]
    is_hybrid_figure = len(panels) == 3
    width = 6.8 * len(panels)
    fig_height = 7.8 if is_hybrid_figure else 7.0
    fig, axes = plt.subplots(1, len(panels), figsize=(width, fig_height), constrained_layout=False)
    if len(panels) == 1:
        axes = [axes]
    for idx, ((_stem, title, coords, panel_labels, panel_correct), axis) in enumerate(zip(panels, axes)):
        if not np.array_equal(labels, panel_labels) or not np.array_equal(correct, panel_correct):
            raise ValueError(f"{result_dir}: label/correct columns differ across panels")
        draw_interleaved_2d(axis, coords, labels, correct)
        axis.set_title(title, fontsize=22, pad=14)
        style_axis(axis, show_ylabel=(idx == 0))

    legend_fontsize = 21 if is_hybrid_figure else 16
    legend_scale = 1.45 if is_hybrid_figure else 1.0
    legend_y = 0.085 if is_hybrid_figure else 0.065
    legend = fig.legend(
        handles=legend_handles(legend_scale),
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, legend_y),
        fontsize=legend_fontsize,
        frameon=True,
        columnspacing=2.2,
        handletextpad=0.8,
        borderpad=0.7,
    )
    legend.get_frame().set_edgecolor("#000000")
    legend.get_frame().set_linewidth(2.2)
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_facecolor("#ffffff")
    bottom = 0.31 if is_hybrid_figure else 0.24
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=bottom, wspace=0.18)

    outputs = [
        result_dir / f"{file_run}_seed{seed}_representation_pipeline_tsne.png",
        result_dir / f"{slug}.png",
    ]
    if seed != 0:
        outputs.append(result_dir / f"{file_run}_seed{seed}.png")
    for alias in (result_dir / f"{slug}_combined.png", result_dir / f"{slug}_representation_pipeline_tsne.png"):
        if alias.exists():
            outputs.append(alias)
    outputs = list(dict.fromkeys(outputs))
    for out in outputs:
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    del fig, axes, panels
    gc.collect()
    return outputs


def replace_strings(payload: Any, replacements: dict[str, str]) -> tuple[Any, bool]:
    if isinstance(payload, str):
        replacement = payload
        for old, new in replacements.items():
            replacement = replacement.replace(old, new)
        return replacement, replacement != payload
    if isinstance(payload, list):
        changed = False
        updated = []
        for item in payload:
            new_item, item_changed = replace_strings(item, replacements)
            updated.append(new_item)
            changed = changed or item_changed
        return updated, changed
    if isinstance(payload, dict):
        changed = False
        updated = {}
        for key, value in payload.items():
            new_value, value_changed = replace_strings(value, replacements)
            updated[key] = new_value
            changed = changed or value_changed
        return updated, changed
    return payload, False


def write_json_if_changed(path: Path, replacements: dict[str, str]) -> None:
    data = load_json(path)
    if not data:
        return
    updated, changed = replace_strings(data, replacements)
    if changed:
        path.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("updated", display_path(path))


def collection_name(result_dir: Path) -> str:
    run, _model, _n, slug = infer_metadata(result_dir)
    summary = load_json(result_dir / "summary.json")
    seed = int(summary.get("seed") or 0)
    file_run = str(summary.get("run") or run)
    if seed == 0:
        return f"{slug}.png"
    return f"{file_run}_seed{seed}.png"


def update_summary_names(result_dirs: list[Path]) -> None:
    replacements: dict[str, str] = {}
    for result_dir in result_dirs:
        run, _model, _n, slug = infer_metadata(result_dir)
        replacements[f"{run}_seed0_representation_pipeline_tsne.png"] = f"{slug}.png"
        replacements[f"{slug}_representation_pipeline_tsne.png"] = f"{slug}.png"

    for result_dir in result_dirs:
        write_json_if_changed(result_dir / "summary.json", replacements)
        write_json_if_changed(result_dir / "tsne3d" / "summary.json", replacements)

    for path in sorted(RESULT_DIR.glob("*summary.json")):
        write_json_if_changed(path, replacements)


def collect_short_figures(result_dirs: list[Path]) -> list[Path]:
    COLLECT_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for result_dir in sorted(result_dirs):
        _run, _model, _n, slug = infer_metadata(result_dir)
        target_name = collection_name(result_dir)
        src = result_dir / target_name
        if not src.exists():
            src = result_dir / f"{slug}.png"
        if not src.exists():
            continue
        dst = COLLECT_DIR / target_name
        shutil.copy2(src, dst)
        copied.append(dst)
    print(f"collected {len(copied)} files in {display_path(COLLECT_DIR)}", flush=True)
    for path in copied:
        print(display_path(path), flush=True)
    return copied


def main() -> None:
    global RESULT_DIR, PLOT_RESULT_DIR, COLLECT_DIR

    args = parse_args()
    RESULT_DIR = args.result_dir
    PLOT_RESULT_DIR = args.plot_result_dir
    COLLECT_DIR = args.collect_dir if args.collect_dir is not None else RESULT_DIR / "embedding_tsne_figures"

    rendered_dirs = []
    for result_dir in sorted(RESULT_DIR.glob("*_seed0")):
        if not result_dir.is_dir():
            continue
        if not list(result_dir.glob("*_tsne.csv")):
            continue
        outputs = save_combined(result_dir)
        rendered_dirs.append(result_dir)
        print("wrote", " ".join(display_path(path) for path in outputs))
    update_summary_names(rendered_dirs)
    collect_short_figures(rendered_dirs)


if __name__ == "__main__":
    main()
