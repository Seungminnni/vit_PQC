import argparse
import itertools
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    name: str = "smoke"
    seed: int = 20260428
    n: int = 16
    M: int = 512
    q: int = 127
    mode: str = "fixed_h"  # fixed_h or bernoulli
    h: int = 3
    p: float = 0.25
    sigma_e: float = 0.0
    batch_size: int = 64
    eval_batch_size: int = 128
    epochs: int = 5
    steps_per_epoch: int = 50
    eval_batches: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.05
    embed_dim: int = 256
    num_heads: int = 8
    depth: int = 4
    dropout: float = 0.1
    log_every_steps: int = 25
    residual_topK: int = 10
    residual_eval_samples: int = 32


PROFILES = {
    "smoke": Config(name="smoke", epochs=3, steps_per_epoch=20, batch_size=64, embed_dim=128, depth=2),
    "binary_h3_noiseless": Config(
        name="binary_h3_noiseless",
        mode="fixed_h",
        h=3,
        sigma_e=0.0,
        batch_size=256,
        epochs=80,
        steps_per_epoch=500,
        embed_dim=512,
        depth=6,
        lr=1e-4,
    ),
    "binary_h3_noise1": Config(
        name="binary_h3_noise1",
        mode="fixed_h",
        h=3,
        sigma_e=1.0,
        batch_size=256,
        epochs=100,
        steps_per_epoch=600,
        embed_dim=512,
        depth=6,
        lr=1e-4,
    ),
    "binary_bernoulli_noise1": Config(
        name="binary_bernoulli_noise1",
        mode="bernoulli",
        p=0.25,
        sigma_e=1.0,
        batch_size=256,
        epochs=100,
        steps_per_epoch=600,
        embed_dim=512,
        depth=6,
        lr=1e-4,
    ),
}


def generate_binary_s(cfg: Config, batch_size: int, device: torch.device) -> torch.Tensor:
    if cfg.mode == "fixed_h":
        s = torch.zeros(batch_size, cfg.n, dtype=torch.long, device=device)
        support = torch.rand(batch_size, cfg.n, device=device).topk(cfg.h, dim=1).indices
        s.scatter_(1, support, 1)
        return s
    if cfg.mode == "bernoulli":
        return (torch.rand(batch_size, cfg.n, device=device) < cfg.p).long()
    raise ValueError(f"unknown mode: {cfg.mode}")


def generate_lwe_batch(cfg: Config, batch_size: int, device: torch.device):
    s = generate_binary_s(cfg, batch_size, device)
    A = torch.randint(0, cfg.q, (batch_size, cfg.M, cfg.n), dtype=torch.long, device=device)
    if cfg.sigma_e > 0:
        e = torch.round(torch.randn(batch_size, cfg.M, device=device) * cfg.sigma_e).long()
    else:
        e = torch.zeros(batch_size, cfg.M, dtype=torch.long, device=device)
    prod = torch.bmm(A.float(), s.float().unsqueeze(-1)).squeeze(-1).round().long()
    b = torch.remainder(prod + e, cfg.q).unsqueeze(-1)
    return A, b, s, e.unsqueeze(-1)


def encode_rgb_3ch(A: torch.Tensor, b: torch.Tensor, q: int) -> torch.Tensor:
    matrix = torch.cat([A.float(), b.float()], dim=2)
    ch1 = matrix / q
    ch2 = torch.abs(matrix - q / 2) / (q / 2)
    ch3 = (torch.sin(2 * math.pi * matrix / q) + 1) / 2
    return torch.stack([ch1, ch2, ch3], dim=1).contiguous()


class RGBBinaryViT(nn.Module):
    def __init__(self, M: int, n: int, embed_dim: int, num_heads: int, depth: int, dropout: float):
        super().__init__()
        self.M = M
        self.n = n
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=(M, 1))
        self.pos_embed = nn.Parameter(torch.randn(1, n + 1, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.support_head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        if x.size(1) != 3:
            raise ValueError(f"expected RGB 3-channel input, got {x.size(1)} channels")
        z = self.patch_embed(x).squeeze(2).transpose(1, 2)
        if z.size(1) != self.n + 1:
            raise ValueError(f"expected {self.n + 1} column tokens, got {z.size(1)}")
        z = self.norm(z + self.pos_embed)
        z = self.transformer(z)
        return self.support_head(z[:, : self.n]).squeeze(-1)


def centered_int(x: torch.Tensor, q: int) -> torch.Tensor:
    half = q // 2
    return torch.remainder(x + half, q) - half


def residual_score(A: torch.Tensor, b: torch.Tensor, s_hat: torch.Tensor, q: int) -> torch.Tensor:
    pred = torch.bmm(A.float(), s_hat.float().unsqueeze(-1)).squeeze(-1).round().long()
    r = centered_int(b.squeeze(-1) - torch.remainder(pred, q), q)
    return (r.float() ** 2).mean(dim=1)


def decode_logits(logits: torch.Tensor, cfg: Config) -> torch.Tensor:
    if cfg.mode == "fixed_h":
        idx = torch.topk(logits, k=cfg.h, dim=1).indices
        pred = torch.zeros_like(logits, dtype=torch.long)
        pred.scatter_(1, idx, 1)
        return pred
    return (torch.sigmoid(logits) > 0.5).long()


def compute_metrics(pred_s: torch.Tensor, true_s: torch.Tensor, A: torch.Tensor, b: torch.Tensor, cfg: Config):
    eps = 1e-8
    pred_support = pred_s.bool()
    true_support = true_s.bool()
    tp = (pred_support & true_support).float().sum()
    fp = (pred_support & ~true_support).float().sum()
    fn = (~pred_support & true_support).float().sum()
    return {
        "coord_acc": (pred_s == true_s).float().mean().item(),
        "support_precision": (tp / (tp + fp + eps)).item(),
        "support_recall": (tp / (tp + fn + eps)).item(),
        "full_match": (pred_s == true_s).all(dim=1).float().mean().item(),
        "mean_hw_pred": pred_s.float().sum(dim=1).mean().item(),
        "mean_hw_true": true_s.float().sum(dim=1).mean().item(),
        "residual_score": residual_score(A, b, pred_s, cfg.q).mean().item(),
    }


def enumerate_binary_candidates(indices, n: int, fixed_h=None, device=None) -> torch.Tensor:
    indices = [int(i) for i in indices]
    if fixed_h is None:
        subsets = []
        for r in range(len(indices) + 1):
            subsets.extend(itertools.combinations(indices, r))
    else:
        subsets = list(itertools.combinations(indices, fixed_h))
    candidates = torch.zeros(len(subsets), n, dtype=torch.long, device=device)
    for row, subset in enumerate(subsets):
        if subset:
            candidates[row, list(subset)] = 1
    return candidates


@torch.no_grad()
def residual_rerank_one(A_one, b_one, logits_one, cfg: Config):
    topK = min(cfg.residual_topK, cfg.n)
    idx = torch.topk(logits_one, k=topK).indices.detach().cpu().tolist()
    fixed_h = cfg.h if cfg.mode == "fixed_h" else None
    candidates = enumerate_binary_candidates(idx, cfg.n, fixed_h=fixed_h, device=logits_one.device)
    scores = []
    for cand in candidates.split(4096):
        scores.append(residual_score(A_one.unsqueeze(0).expand(cand.size(0), -1, -1), b_one.unsqueeze(0).expand(cand.size(0), -1, -1), cand, cfg.q))
    scores = torch.cat(scores, dim=0)
    return candidates[torch.argmin(scores)]


@torch.no_grad()
def evaluate(model, cfg: Config, device: torch.device):
    model.eval()
    rows = []
    rerank_rows = []
    for batch_idx in range(cfg.eval_batches):
        A, b, s, _ = generate_lwe_batch(cfg, cfg.eval_batch_size, device)
        logits = model(encode_rgb_3ch(A, b, cfg.q))
        pred = decode_logits(logits, cfg)
        rows.append(compute_metrics(pred, s, A, b, cfg))

        if batch_idx == 0 and cfg.residual_eval_samples > 0:
            limit = min(cfg.residual_eval_samples, A.size(0))
            rr = []
            for i in range(limit):
                rr.append(residual_rerank_one(A[i], b[i], logits[i], cfg))
            rr = torch.stack(rr, dim=0)
            rerank_rows.append(compute_metrics(rr, s[:limit], A[:limit], b[:limit], cfg))

    out = pd.DataFrame(rows).mean().to_dict()
    if rerank_rows:
        rr = pd.DataFrame(rerank_rows).mean().to_dict()
        out.update({f"rerank_{k}": v for k, v in rr.items()})
    return out


def train(cfg: Config):
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RGBBinaryViT(cfg.M, cfg.n, cfg.embed_dim, cfg.num_heads, cfg.depth, cfg.dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.mode == "fixed_h":
        pos_weight = torch.tensor([(cfg.n - cfg.h) / cfg.h], device=device)
    else:
        pos_weight = torch.tensor([(1 - cfg.p) / cfg.p], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    run_dir = Path("idea") / "runs" / f"{cfg.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    hist_path = run_dir / "history.csv"
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"run_dir={run_dir}")
    log(f"device={device}")
    log(f"config={cfg}")
    log(f"streamed_train_samples={cfg.epochs * cfg.steps_per_epoch * cfg.batch_size:,}")
    log(f"model_params={sum(p.numel() for p in model.parameters()):,}")

    history = []
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses, coord_accs, full_matches = [], [], []
        e0 = time.time()
        for step in range(1, cfg.steps_per_epoch + 1):
            A, b, s, _ = generate_lwe_batch(cfg, cfg.batch_size, device)
            logits = model(encode_rgb_3ch(A, b, cfg.q))
            loss = criterion(logits, s.float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                pred = decode_logits(logits, cfg)
                losses.append(loss.item())
                coord_accs.append((pred == s).float().mean().item())
                full_matches.append((pred == s).all(dim=1).float().mean().item())

            if step % cfg.log_every_steps == 0 or step == cfg.steps_per_epoch:
                recent = min(cfg.log_every_steps, len(losses))
                log(
                    f"epoch {epoch:03d} step {step:05d}/{cfg.steps_per_epoch} | "
                    f"loss={np.mean(losses[-recent:]):.4f} | "
                    f"coord={np.mean(coord_accs[-recent:]):.3f} | "
                    f"full={np.mean(full_matches[-recent:]):.3f} | "
                    f"elapsed_epoch={time.time() - e0:.1f}s"
                )

        val = evaluate(model, cfg, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_coord_acc": float(np.mean(coord_accs)),
            "train_full_match": float(np.mean(full_matches)),
            **{f"val_{k}": v for k, v in val.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(hist_path, index=False)
        log(
            f"epoch {epoch:03d} done | loss={row['train_loss']:.4f} | "
            f"train_full={row['train_full_match']:.3f} | "
            f"val_full={row['val_full_match']:.3f} | "
            f"val_coord={row['val_coord_acc']:.3f} | "
            f"val_recall={row['val_support_recall']:.3f} | "
            f"rerank_full={row.get('val_rerank_full_match', float('nan')):.3f} | "
            f"residual={row['val_residual_score']:.2f} | "
            f"elapsed={time.time() - t0:.1f}s"
        )
        if row["val_full_match"] >= 0.995 or row.get("val_rerank_full_match", 0.0) >= 0.995:
            log("early_stop=full_match_threshold")
            break

    log(f"final_history={hist_path}")
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="smoke", choices=sorted(PROFILES))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sigma-e", type=float)
    args = parser.parse_args()

    cfg = Config(**asdict(PROFILES[args.profile]))
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.steps_per_epoch is not None:
        cfg.steps_per_epoch = args.steps_per_epoch
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.sigma_e is not None:
        cfg.sigma_e = args.sigma_e
    train(cfg)


if __name__ == "__main__":
    main()
