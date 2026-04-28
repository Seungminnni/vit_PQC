# Idea3: Binary RHIE-CIP H-Free Recovery

`idea3` follows `document/idea3.docx`.

Flow:

```text
A,b,q
-> 36-channel binary RHIE-CIP signal image
-> model outputs p_j = Pr(s_j = 1) for every coordinate j
-> threshold base secret
-> flip uncertain bits only
-> choose final s_hat by centered residual score b - A s_hat
```

## Main Difference From Idea2

- `idea2` is known-Hamming recovery: top-K coordinates and `combinations(topK, h)`.
- `idea3` is h-free inference: no top-h, no `combinations(topK, h)`, no `candidate_hit_rate(..., h, ...)`.
- Candidate space is `2^U`, where `U = --hfree_uncertain_K`.
- `C(50,3)` is only a fixed-h brute-force comparison point, not the idea3 recovery space.

## Secret Generation

Main idea3 setting:

```text
s_j ~ Bernoulli(p_nonzero)
```

For `n=50, p_nonzero=0.06`, the expected Hamming weight is `E[h]=3`, but each sample can have a different actual Hamming weight.

Fixed-H ablation:

```text
--fixed_train_h 3 --fixed_eval_h 3
```

This means fixed-H data, but inference is still h-free because the recovery step does not receive `h`.

## Loss

Default loss mode is:

```text
--loss_pos_weight_mode prior
pos_weight = (1 - p_nonzero) / p_nonzero
```

This is a Bernoulli sparsity prior, not a `combinations(topK,h)` recovery assumption. For calibration checks, use:

```text
--loss_pos_weight_mode none
```

or set a manual value:

```text
--loss_pos_weight_mode manual --pos_weight 4.0
```

## Run Main Bernoulli Experiment

```bash
cd /home/yu_mcc/vit_PQC/idea3
./scripts/run_n50_q127_hfree_e2.sh
```

Equivalent command:

```bash
../.venv_lwe18/bin/python train_binary.py \
  --preset stage3_h3 \
  --n 50 \
  --M 1000 \
  --q 127 \
  --h 3 \
  --secret_distribution bernoulli \
  --p_nonzero 0.06 \
  --sigma_e 2.0 \
  --steps 5000 \
  --batch_size 8 \
  --loss_pos_weight_mode prior \
  --hfree_uncertain_K 12 \
  --hfree_threshold 0.5 \
  --amp \
  --amp_dtype bf16
```

## Run Fixed-H Data Ablation

```bash
./scripts/run_n50_q127_fixed_train_h3_e2.sh
```

This uses fixed `h=3` for train/eval data, but still uses threshold + uncertain bit flip + residual verifier for recovery.

## Metrics

- `coord_acc`: thresholded coordinate accuracy.
- `support_precision`, `support_recall`, `support_f1`: support quality.
- `direct_full_match`: thresholded secret exact match before verifier.
- `h_abs_error`: evaluation-only `|h_hat - h_true|`.
- `candidate_contains_true`: whether the uncertain-bit candidate set contains the true secret.
- `post_verifier_full_match`: final exact recovery after residual verification.
- `candidate_count`: usually `2^U`.
- `residual_gap`: best wrong residual score minus true residual score.
