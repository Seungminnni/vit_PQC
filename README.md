# LWE ViT

Relation-preserving encoders for experiments that predict an LWE secret `s`
from `A, b` where `b = A s + e (mod q)`.

The initial implementation focuses on small fixed LWE parameters and provides:

- synthetic LWE generation through `LWEParams`
- `relation_grid`, `phase_grid`, and `row_equation_tokens` encoders
- row-block tokenization over `X[i,j] = [ENC(A_ij), ENC(b_i)]`
- a compact ViT-style model with column query tokens for coordinate-wise secret prediction
- residual consistency helpers for checking `b - A s_hat (mod q)`

## Quick Check

Use the existing `lattice_env` environment:

```bash
/home/yu_mcc/miniconda3/envs/lattice_env/bin/python -m unittest discover -s tests
```

Run a tiny overfit sanity check:

```bash
PYTHONPATH=src /home/yu_mcc/miniconda3/envs/lattice_env/bin/python scripts/sanity_overfit.py
```

Run the metric-rich row-block trainer. This is now the main path: it builds
pixels as `[ENC(A_ij), ENC(b_i)]`, broadcasts noisy `b_i` across the row, then
flattens row/column blocks into Transformer tokens. With `n=16` and
`--block-cols 16`, each data token contains one full LWE equation row.

```bash
PYTHONPATH=src /home/yu_mcc/miniconda3/envs/lattice_env/bin/python scripts/train_lwe_vit.py \
  --model row_block \
  --run-name row_block_fixed_h2_n16_m128_q257 \
  --h-setting fixed_h --fixed-h 2 \
  --n 16 --m 128 --q 257 \
  --block-rows 1 --block-cols 16 --fourier-k 2 \
  --num-train 4096 --num-val 1024 --num-test 1024 \
  --train-eval-samples 1024 \
  --epochs 30 --batch-size 64 \
  --embed-dim 64 --depth 1 --num-heads 4 \
  --device auto --save-best
```

The trainer writes `history.json/csv`, `summary.json/csv`, aggregate metrics, and
an optional `best.pt` checkpoint under `runs/lwe_vit/<run-name>/`.

The older atom-level pair-token and image patch models are still available for
explicit comparison through `--model pair_token` and `--model vit_patch`, but
they are no longer the recommended main experiment.
