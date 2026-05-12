# LWE ViT

Relation-preserving encoders for experiments that predict an LWE secret `s`
from `A, b` where `b = A s + e (mod q)`.

The initial implementation focuses on small fixed LWE parameters and provides:

- synthetic LWE generation through `LWEParams`
- `relation_grid`, `phase_grid`, and `row_equation_tokens` encoders
- rectangular patch tokenization with padding masks
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

Run the metric-rich pair-token trainer. This is the main path for feeding all
`(A_ij, b_i)` row-column atoms directly to a Transformer:

```bash
PYTHONPATH=src /home/yu_mcc/miniconda3/envs/lattice_env/bin/python scripts/train_lwe_vit.py \
  --model pair_token \
  --run-name pair_token_fixed_h2_n16_m128_q257 \
  --h-setting fixed_h --fixed-h 2 \
  --n 16 --m 128 --q 257 \
  --num-train 4096 --num-val 1024 --num-test 1024 \
  --epochs 30 --batch-size 2 \
  --embed-dim 64 --depth 1 --num-heads 4 \
  --device auto --save-best
```

The trainer writes `history.json/csv`, `summary.json/csv`, aggregate metrics, and
an optional `best.pt` checkpoint under `runs/lwe_vit/<run-name>/`.

The older image patch model is still available for comparison through
`--model vit_patch`, but it is no longer the recommended main experiment.
