# LWE Image Embedding Attack

Implementation of the flow in `../document/LWE_Image_Embedding_Implementation_Flow_Report.docx`.

The main path is BKZ-free raw LWE:

1. Generate synthetic sparse LWE batches.
2. Build raw, phase, RHIE, and optional Gram views.
3. Train a CNN or column transformer to predict secret support.
4. Combine model posterior with residual verification for exact support recovery.

## Quick Checks

```bash
cd idea1
../.venv_lwe18/bin/python -m src.sanity --config configs/base.yaml
../.venv_lwe18/bin/python -m src.train --config configs/smoke.yaml
```

## Main Experiments

```bash
cd idea1
../.venv_lwe18/bin/python -m src.train --config configs/cnn_raw_binary.yaml
../.venv_lwe18/bin/python -m src.train --config configs/cnn_phase_binary.yaml
../.venv_lwe18/bin/python -m src.train --config configs/cnn_rhie_binary.yaml
../.venv_lwe18/bin/python -m src.attack_eval --config configs/cnn_rhie_binary.yaml --checkpoint outputs/logs/<run>/best.pt
../.venv_lwe18/bin/python -m src.train --config configs/transformer_rhie_binary.yaml
```

Training writes JSONL metrics and checkpoints under `outputs/logs/`.
