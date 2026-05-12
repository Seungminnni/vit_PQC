from __future__ import annotations

import argparse
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lwe_vit import (  # noqa: E402
    LWEImageEncoder,
    LWEParams,
    LWEViTConfig,
    LWEViTForSecret,
    RepresentationConfig,
    num_secret_classes,
    residual_consistency_loss,
    sample_lwe_batch,
    secret_cross_entropy_loss,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dataset-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    params = LWEParams(
        n=8,
        m=24,
        q=17,
        secret_dist="binary",
        noise_dist="uniform_small",
        noise_width=1,
        seed=7,
    )
    sample = sample_lwe_batch(params, batch_size=args.dataset_size, device=args.device)
    rep = RepresentationConfig(name="relation_grid", patch_rows=4, patch_cols=4, use_phase=True)
    encoder = LWEImageEncoder(rep, q=params.q)
    image, mask = encoder.encode(sample.A, sample.b)

    model = LWEViTForSecret(
        LWEViTConfig(
            n=params.n,
            q=params.q,
            in_channels=encoder.num_channels(),
            num_secret_classes=num_secret_classes(params),
            patch_rows=rep.patch_rows,
            patch_cols=rep.patch_cols,
            embed_dim=64,
            depth=2,
            num_heads=4,
        )
    ).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    for step in range(1, args.steps + 1):
        idx = torch.randint(args.dataset_size, (args.batch_size,), device=args.device)
        out = model(image[idx], mask[idx])
        ce = secret_cross_entropy_loss(out.s_logits, sample.s_labels[idx])
        rc = residual_consistency_loss(
            sample.A[idx],
            sample.b[idx],
            out.s_logits,
            q=params.q,
            secret_dist=params.secret_dist,
            noise_bound=params.noise_width + 0.5,
        )
        loss = ce + 0.05 * rc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % 10 == 0:
            acc = (out.s_logits.argmax(dim=-1) == sample.s_labels[idx]).float().mean().item()
            print(f"step={step:03d} loss={loss.item():.4f} ce={ce.item():.4f} rc={rc.item():.4f} acc={acc:.3f}")


if __name__ == "__main__":
    main()
