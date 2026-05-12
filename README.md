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

Run the metric-rich trainer:

```bash
PYTHONPATH=src /home/yu_mcc/miniconda3/envs/lattice_env/bin/python scripts/train_lwe_vit.py \
  --run-name relation_grid_n16_m128_q257_proper \
  --n 16 --m 128 --q 257 \
  --num-train 4096 --num-val 1024 --num-test 1024 \
  --epochs 10 --batch-size 128 --device auto --save-best
```

The trainer writes `history.json/csv`, `summary.json/csv`, aggregate metrics, and
an optional `best.pt` checkpoint under `runs/lwe_vit/<run-name>/`.
