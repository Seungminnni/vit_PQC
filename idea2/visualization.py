from __future__ import annotations

from pathlib import Path
import os

import torch

from candidate import binary_secret_from_support
from modular import centered_int, lwe_dot
from recovery import RecoveryTrace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def _ensure(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_support_bar(logits: torch.Tensor, trace: RecoveryTrace, path: str | Path) -> None:
    import matplotlib.pyplot as plt

    probs = torch.sigmoid(logits.detach()).cpu()
    colors = []
    true = set(trace.true_support)
    best = set(trace.best_support)
    for idx in range(len(probs)):
        if idx in true and idx in best:
            colors.append("tab:green")
        elif idx in true:
            colors.append("tab:red")
        elif idx in best:
            colors.append("tab:orange")
        else:
            colors.append("tab:blue")
    plt.figure(figsize=(10, 3))
    plt.bar(range(len(probs)), probs.numpy(), color=colors)
    plt.ylim(0.0, 1.0)
    plt.xlabel("coordinate")
    plt.ylabel("posterior")
    plt.title(f"true={trace.true_support} best={trace.best_support} topK={trace.topk}")
    plt.tight_layout()
    plt.savefig(_ensure(path), dpi=160)
    plt.close()


def save_candidate_score_plot(trace: RecoveryTrace, path: str | Path, max_candidates: int = 80) -> None:
    import matplotlib.pyplot as plt

    rows = trace.candidates[:max_candidates]
    xs = list(range(len(rows)))
    residual = [row.residual_score for row in rows]
    total = [row.total_score for row in rows]
    colors = ["tab:green" if row.is_true else "tab:gray" for row in rows]
    plt.figure(figsize=(10, 4))
    plt.scatter(xs, residual, c=colors, label="residual")
    if any(abs(a - b) > 1e-9 for a, b in zip(residual, total)):
        plt.plot(xs, total, color="tab:orange", linewidth=1, label="total")
    plt.xlabel("candidate rank")
    plt.ylabel("score")
    plt.title("candidate residual scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_ensure(path), dpi=160)
    plt.close()


def save_residual_histograms(A: torch.Tensor, b: torch.Tensor, trace: RecoveryTrace, q: int, path: str | Path) -> None:
    import matplotlib.pyplot as plt

    true_s = binary_secret_from_support(A.shape[1], trace.true_support, A.device)
    best_s = binary_secret_from_support(A.shape[1], trace.best_support, A.device)
    r_true = centered_int(torch.remainder(b - lwe_dot(A, true_s), q), q).detach().cpu().float()
    r_best = centered_int(torch.remainder(b - lwe_dot(A, best_s), q), q).detach().cpu().float()
    plt.figure(figsize=(8, 4))
    bins = min(40, max(8, int(q // 4)))
    plt.hist(r_true.numpy(), bins=bins, alpha=0.65, label="true")
    plt.hist(r_best.numpy(), bins=bins, alpha=0.65, label="best")
    plt.xlabel("centered residual")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_ensure(path), dpi=160)
    plt.close()


def save_feature_heatmaps(X: torch.Tensor, path_prefix: str | Path, max_channels: int = 6) -> None:
    import matplotlib.pyplot as plt

    prefix = Path(path_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    C = min(max_channels, X.shape[0])
    for ch in range(C):
        arr = X[ch].detach().cpu().float().numpy()
        plt.figure(figsize=(8, 4))
        plt.imshow(arr, aspect="auto", cmap="coolwarm")
        plt.colorbar(fraction=0.03, pad=0.02)
        plt.title(f"feature channel {ch}")
        plt.xlabel("coordinate")
        plt.ylabel("equation")
        plt.tight_layout()
        plt.savefig(prefix.parent / f"{prefix.name}_ch{ch}.png", dpi=160)
        plt.close()
