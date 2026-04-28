# RHIE-CG LWE Candidate Recovery

Implementation of `../document/RHIE-CG_LWE_Report_Reformatted.docx`.

RHIE-CG is not a direct secret classifier. It is a candidate compression framework:

1. Generate raw LWE `(A,b,q)`.
2. Encode it as a coordinate/equation crypto image packet.
3. Predict coordinate-wise support/value posterior.
4. Keep top-K coordinates or values.
5. Enumerate a small candidate set.
6. Select the final secret with a residual likelihood distinguisher.

This directory now focuses on the binary branch. Ternary/integer entry points remain as experimental extensions, but the completed workflow is binary RHIE-CG without BKZ/LLL/Flatter dimension-reduction preprocessing.

## Quick Start

```bash
cd idea2
../.venv_lwe18/bin/python sanity_checks.py
../.venv_lwe18/bin/python binary_sanity_suite.py --preset stage3_h3 --batch_size 32
../.venv_lwe18/bin/python train_binary.py --preset stage0 --steps 5 --batch_size 16
../.venv_lwe18/bin/python train_binary.py --preset stage3 --encoder_type rhie_cip --topK 8
```

## Main Files

- `configs.py`: presets and CLI config helpers.
- `modular.py`: centered representatives, phase features, circular loss.
- `data.py`: binary/ternary/integer LWE batch generation.
- `features/`: raw/baseline/RHIE-CIP encoders.
- `models/`: local embedding, axial blocks, pooling, coordinate transformer, decoder.
- `candidate.py`: top-K candidate generation and beam search.
- `distinguisher.py`: residual likelihood scoring.
- `pair_residual.py`: top-K pair residual pruning and diagnostics.
- `recovery.py`: per-instance binary recovery traces.
- `metrics.py`: candidate hit, rerank exact, residual gap, reduction factor.
- `train_binary.py`, `train_ternary.py`, `train_integer.py`: training entry points.
- `evaluate.py`, `ablation.py`, `sanity_checks.py`, `binary_sanity_suite.py`, `analyze_binary.py`: experiment utilities.

## Recommended Flow

```bash
../.venv_lwe18/bin/python train_binary.py --preset stage0 --sigma_e 0.0
../.venv_lwe18/bin/python train_binary.py --preset stage1_h1 --sigma_e 0.0
../.venv_lwe18/bin/python train_binary.py --preset stage2_h2 --sigma_e 0.0
../.venv_lwe18/bin/python train_binary.py --preset stage3_h3 --sigma_e 0.0
../.venv_lwe18/bin/python train_binary.py --preset stage3_h3 --use_pair_filter --pair_budget 16
../.venv_lwe18/bin/python train_binary.py --preset stage3_h3 --secret_split --train_secret_fraction 0.8
../.venv_lwe18/bin/python ablation.py --experiment candidate --preset stage3_h3 --use_pair_filter --pair_budget 16
../.venv_lwe18/bin/python ablation.py --experiment feature --preset stage3_h3 --steps 200
../.venv_lwe18/bin/python analyze_binary.py --checkpoint results/csv/<run>/best.pt --max_cases 8
```
