from __future__ import annotations

from pathlib import Path

import torch


def save_heatmap(tensor: torch.Tensor, path: str | Path, title: str | None = None) -> None:
    import matplotlib.pyplot as plt

    arr = tensor.detach().cpu().float().numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    if title:
        plt.title(title)
    plt.imshow(arr, aspect="auto", cmap="coolwarm")
    plt.colorbar(fraction=0.03, pad=0.02)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_scores_bar(scores: torch.Tensor, support: torch.Tensor, path: str | Path) -> None:
    import matplotlib.pyplot as plt

    probs = torch.sigmoid(scores.detach()).cpu().float().numpy()
    support_np = support.detach().cpu().numpy()
    colors = ["tab:red" if x else "tab:blue" for x in support_np]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 3))
    plt.bar(range(len(probs)), probs, color=colors)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

