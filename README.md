# LWE ViT

Relation-preserving encoders for experiments that predict an LWE secret `s`
from `A, b` where `b = A s + e (mod q)`.

The initial implementation focuses on small fixed LWE parameters and provides:

- synthetic LWE generation through `LWEParams`
- `relation_grid`, `phase_grid`, and `row_equation_tokens` encoders
- row-block tokenization over `X[i,j] = [ENC(A_ij), ENC(b_i)]`
- a compact ViT-style model with column query tokens for coordinate-wise secret prediction
- residual consistency helpers for checking `b - A s_hat (mod q)`

The default training objective is now the defensible supervised baseline:
coordinate-wise weighted cross entropy. Residual consistency is still computed
and logged as a LWE-specific validation signal, but it is not part of the
training loss unless `--residual-loss-weight` is set above `0`.

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
  --h-setting fixed_h --fixed-h 2 \
  --n 16 --m 128 --q 257 \
  --block-rows 1 --block-cols 16 --residue-encoding phase \
  --num-train 4096 --num-val 1024 --num-test 1024 \
  --train-eval-samples 1024 \
  --epochs 30 --batch-size 64 \
  --embed-dim 64 --depth 1 --num-heads 4 \
  --device auto --save-best
```

The default run name is compact and experiment-readable, e.g.
`phase6_bc16_n16_m128_4096_1024_1024_ep30` or
`raw_bc1_n16_m256_100k_10k_10k_ep150`. The trainer writes `train.log`,
`command.txt`, `history.json/csv`, `summary.json/csv`, aggregate metrics, and an
optional `best.pt` checkpoint under `runs/lwe_vit/<run-name>/`.

Use `--residual-loss-weight 0.05` only for residual-loss ablations. Such runs
receive a suffix like `_resw0p05` in the default run name.

For very large synthetic datasets, add `--on-the-fly`. This avoids storing every
`A,b,s,e` sample in RAM and instead regenerates each sample deterministically
from its split seed and index. Train/val/test use separate seed offsets.

The active model set is intentionally small:

- `--model row_block`: ViT-style relation-grid block tokens
- `--model equation_transformer`: plain row-equation Transformer
- `--model row_cnn`: row-local CNN over the same relation rows
