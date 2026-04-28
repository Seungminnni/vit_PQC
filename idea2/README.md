# RHIE-CG LWE Candidate Recovery

Implementation of `../document/RHIE-CG_LWE_Report_Reformatted.docx`.

RHIE-CG is not a direct secret classifier. It is a candidate compression framework:

1. Generate raw LWE `(A,b,q)`.
2. Encode it as a coordinate/equation crypto image packet.
3. Predict coordinate-wise support/value posterior.
4. Keep top-K coordinates or values.
5. Enumerate a small candidate set.
6. Select the final secret with a residual likelihood distinguisher.

For binary recovery, the model does not directly output the final full secret. It predicts coordinate-wise support scores.
The final secret is selected by enumerating Hamming-weight candidates inside the model's top-K coordinates and reranking
them with the LWE residual.

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
../.venv_lwe18/bin/python visualize_embedding.py --config_json results/csv/<run>/config.json --num_cases 2
../.venv_lwe18/bin/python generate_run_artifacts.py --runs results/csv/n10_m64_q127_h1_e0_s64000_<ts> results/csv/n10_m128_q127_h2_e0_s96000_<ts> results/csv/n16_m128_q127_h3_e0_s320000_<ts>
../.venv_lwe18/bin/python plot_results.py --runs results/csv/n10_m64_q127_h1_e0_s64000_<ts> results/csv/n10_m128_q127_h2_e0_s96000_<ts> results/csv/n16_m128_q127_h3_e0_s320000_<ts>
```

`generate_run_artifacts.py` writes per-run `analysis/`, `embedding/`, and `metrics/` directories under
`results/figures/run_artifacts` by default. Use `--skip_analysis --skip_embedding` to regenerate only the
loss/accuracy metric plots from `metrics.jsonl`.

Run directories use the default label `n{n}_m{M}_q{q}_h{h}_e{sigma}_s{steps*batch_size}` unless `--run_name` is passed explicitly.

## Metrics

`metrics.jsonl` contains two different kinds of rows.

Train rows measure the neural candidate generator:

```json
{"loss": 0.2398, "loss_support": 0.2398, "split": "train", "step": 1300}
```

- `loss_support`: binary cross-entropy between the true support coordinates and the model's coordinate-wise support logits.
- `loss`: total training loss. In the binary default setup this is the same as `loss_support` unless auxiliary losses are enabled.

Eval rows measure both candidate generation and final residual recovery:

```json
{
  "candidate_count": 56.0,
  "candidate_hit_rate": 0.925,
  "coord_acc": 0.9187,
  "post_rerank_full_match": 0.925,
  "pre_rerank_full_match": 0.6969,
  "rerank_success_given_hit": 1.0,
  "split": "eval",
  "step": 1200
}
```

- `coord_acc`: coordinate-level support accuracy of the model.
- `pre_rerank_full_match`: full support recovery if we take the model's top-h coordinates directly.
- `candidate_hit_rate`: whether the true support is contained in the enumerated top-K candidate set.
- `candidate_count`: number of enumerated candidates per instance. For `h=3, topK=8`, this is `C(8,3)=56`.
- `post_rerank_full_match`: final full support recovery after residual reranking over the candidate set.
- `rerank_success_given_hit`: reranker success conditional on the true support being present in the candidate set.
- `residual_gap`: average score gap between the best wrong residual and the true residual when available.
- `reduction_factor`: compression ratio from the full Hamming search space to the top-K candidate space.
- `pair_filter_enabled`: `1.0` if pair-residual candidate pruning was enabled, otherwise `0.0`.

The key reading is:

```text
model-only exact recovery       = pre_rerank_full_match
model candidate quality         = candidate_hit_rate
model + residual final recovery = post_rerank_full_match
```

If `candidate_hit_rate` and `post_rerank_full_match` are close, the residual solver is selecting the correct candidate once
the model places the true support inside the candidate set. In that case, failures mostly come from the model missing one or
more true coordinates in its top-K list.
