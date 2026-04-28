from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass
class LWEConfig:
    n: int = 16
    M: int = 128
    q: int = 127
    h: int = 3
    sigma_e: float = 0.0
    noise_type: str = "rounded_gaussian"
    secret_type: str = "binary"
    secret_distribution: str = "bernoulli"
    p_nonzero: float = 0.1875
    h_min: int = 2
    h_max: int = 4
    fixed_train_h: int | None = None
    fixed_eval_h: int | None = None


@dataclass
class FeatureConfig:
    encoder_type: str = "rhie_cip"
    freqs: tuple[int, ...] = (1, 2, 4)
    include_raw: bool = True
    include_phase: bool = True
    include_rhie: bool = True
    include_interaction: bool = True
    include_stats: bool = True


@dataclass
class ModelConfig:
    d_model: int = 96
    depth: int = 3
    heads: int = 4
    dropout: float = 0.1
    pooling: str = "attention"
    coordinate_transformer: bool = True
    axial_mode: str = "row_column"
    use_position: bool = False


@dataclass
class TrainConfig:
    batch_size: int = 64
    steps: int = 1000
    lr: float = 3e-4
    weight_decay: float = 1e-4
    log_every: int = 50
    eval_every: int = 200
    eval_batches: int = 10
    aux_residual_weight: float = 0.0
    seed: int = 42
    device: str = "cuda"
    amp: bool = False
    amp_dtype: str = "bf16"
    loss_pos_weight_mode: str = "prior"
    pos_weight: float = 1.0


@dataclass
class CandidateConfig:
    hfree_uncertain_K: int = 10
    hfree_threshold: float = 0.5
    score_type: str = "squared"
    posterior_weight: float = 0.0


@dataclass
class ExperimentConfig:
    lwe: LWEConfig
    features: FeatureConfig
    model: ModelConfig
    train: TrainConfig
    candidate: CandidateConfig
    run_name: str = "idea3_binary_hfree"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRESETS: dict[str, dict[str, Any]] = {
    "stage0": {"n": 8, "M": 32, "q": 127, "h": 1, "sigma_e": 0.0},
    "stage1": {"n": 10, "M": 64, "q": 127, "h": 1, "sigma_e": 0.0},
    "stage2": {"n": 10, "M": 128, "q": 127, "h": 2, "sigma_e": 0.0},
    "stage3": {"n": 16, "M": 128, "q": 127, "h": 3, "sigma_e": 0.0},
    "stage3_h3": {"n": 16, "M": 128, "q": 127, "h": 3, "sigma_e": 0.0},
    "noise": {"n": 16, "M": 256, "q": 127, "h": 3, "sigma_e": 1.0},
}


def default_h_range(n: int, h: int, h_min: int | None, h_max: int | None) -> tuple[int, int]:
    lo = int(h_min if h_min is not None else max(1, h - 1))
    hi = int(h_max if h_max is not None else min(n, h + 1))
    lo = max(0, min(lo, n))
    hi = max(lo, min(hi, n))
    return lo, hi


def expected_nonzero_probability(lwe: LWEConfig) -> float:
    if lwe.secret_distribution == "bernoulli":
        return min(max(float(lwe.p_nonzero), 1e-6), 1.0 - 1e-6)
    if lwe.secret_distribution == "h_range":
        return min(max(((lwe.h_min + lwe.h_max) / 2.0) / max(lwe.n, 1), 1e-6), 1.0 - 1e-6)
    return min(max(float(lwe.h) / max(lwe.n, 1), 1e-6), 1.0 - 1e-6)


def dimension_run_name(lwe: LWEConfig, train: TrainConfig | None = None) -> str:
    sigma = f"{lwe.sigma_e:g}".replace("-", "m").replace(".", "p")
    if lwe.secret_distribution == "bernoulli":
        dist = f"bern{lwe.p_nonzero:g}".replace(".", "p")
    elif lwe.secret_distribution == "h_range":
        dist = f"hr{lwe.h_min}-{lwe.h_max}"
    else:
        dist = f"fixedh{lwe.h}"
    if lwe.fixed_train_h is not None:
        dist = f"{dist}_trainh{lwe.fixed_train_h}"
    if lwe.fixed_eval_h is not None:
        dist = f"{dist}_evalh{lwe.fixed_eval_h}"
    name = f"n{lwe.n}_m{lwe.M}_q{lwe.q}_{dist}_hfree_e{sigma}"
    if train is not None:
        name = f"{name}_s{train.steps * train.batch_size}"
    return name


def build_config(args: argparse.Namespace, secret_type: str = "binary") -> ExperimentConfig:
    preset = PRESETS.get(args.preset, PRESETS["stage3"]).copy()
    n = int(args.n if args.n is not None else preset["n"])
    h = int(args.h if args.h is not None else preset["h"])
    p_nonzero = float(args.p_nonzero) if args.p_nonzero is not None else float(h) / max(n, 1)
    h_min, h_max = default_h_range(n, h, args.h_min, args.h_max)
    lwe = LWEConfig(
        n=n,
        M=int(args.M if args.M is not None else preset["M"]),
        q=int(args.q if args.q is not None else preset["q"]),
        h=h,
        sigma_e=float(args.sigma_e if args.sigma_e is not None else preset["sigma_e"]),
        secret_type=secret_type,
        secret_distribution=args.secret_distribution,
        p_nonzero=p_nonzero,
        h_min=h_min,
        h_max=h_max,
        fixed_train_h=args.fixed_train_h,
        fixed_eval_h=args.fixed_eval_h,
    )
    features = FeatureConfig(
        encoder_type=args.encoder_type,
        freqs=tuple(int(x) for x in args.freqs.split(",") if x),
        include_raw=not args.no_raw,
        include_phase=not args.no_phase,
        include_rhie=not args.no_rhie,
        include_interaction=not args.no_interaction,
        include_stats=not args.no_stats,
    )
    model = ModelConfig(
        d_model=args.d_model,
        depth=args.depth,
        heads=args.heads,
        dropout=args.dropout,
        pooling=args.pooling,
        coordinate_transformer=not args.no_coordinate_transformer,
        axial_mode=args.axial_mode,
        use_position=args.use_position,
    )
    train = TrainConfig(
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        aux_residual_weight=args.aux_residual_weight,
        seed=args.seed,
        device=args.device,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        loss_pos_weight_mode=args.loss_pos_weight_mode,
        pos_weight=args.pos_weight,
    )
    candidate = CandidateConfig(
        hfree_uncertain_K=args.hfree_uncertain_K,
        hfree_threshold=args.hfree_threshold,
        score_type=args.score_type,
        posterior_weight=args.posterior_weight,
    )
    run_name = args.run_name or dimension_run_name(lwe, train)
    return ExperimentConfig(lwe=lwe, features=features, model=model, train=train, candidate=candidate, run_name=run_name)


def config_from_dict(payload: dict[str, Any]) -> ExperimentConfig:
    def with_defaults(cls, raw: dict[str, Any]) -> dict[str, Any]:
        valid = {field.name for field in fields(cls)}
        merged = asdict(cls())
        merged.update({key: value for key, value in raw.items() if key in valid})
        return merged

    lwe_raw = with_defaults(LWEConfig, payload["lwe"])
    features_raw = with_defaults(FeatureConfig, payload["features"])
    model_raw = with_defaults(ModelConfig, payload["model"])
    train_raw = with_defaults(TrainConfig, payload["train"])
    candidate_raw = with_defaults(CandidateConfig, payload["candidate"])
    if isinstance(features_raw.get("freqs"), list):
        features_raw = {**features_raw, "freqs": tuple(features_raw["freqs"])}
    return ExperimentConfig(
        lwe=LWEConfig(**lwe_raw),
        features=FeatureConfig(**features_raw),
        model=ModelConfig(**model_raw),
        train=TrainConfig(**train_raw),
        candidate=CandidateConfig(**candidate_raw),
        run_name=payload.get("run_name", "idea3_binary_hfree"),
    )


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--preset", default="stage3")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--M", type=int, default=None)
    parser.add_argument("--q", type=int, default=None)
    parser.add_argument("--h", type=int, default=None, help="Reference sparsity: used for fixed-H data, default p_nonzero=h/n, h-range defaults, and run naming.")
    parser.add_argument("--sigma_e", type=float, default=None)
    parser.add_argument("--secret_distribution", default="bernoulli", choices=["h_range", "bernoulli", "fixed"])
    parser.add_argument("--h_min", type=int, default=None)
    parser.add_argument("--h_max", type=int, default=None)
    parser.add_argument("--p_nonzero", type=float, default=None)
    parser.add_argument("--fixed_train_h", type=int, default=None)
    parser.add_argument("--fixed_eval_h", type=int, default=None)
    parser.add_argument(
        "--encoder_type",
        default="rhie_cip",
        choices=["a_only", "baseline_8ch", "baseline_10ch", "baseline_14ch", "rhie_cip"],
    )
    parser.add_argument("--freqs", default="1,2,4")
    parser.add_argument("--no_raw", action="store_true")
    parser.add_argument("--no_phase", action="store_true")
    parser.add_argument("--no_rhie", action="store_true")
    parser.add_argument("--no_interaction", action="store_true")
    parser.add_argument("--no_stats", action="store_true")
    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", default="attention", choices=["attention", "mean", "meanmax"])
    parser.add_argument("--axial_mode", default="row_column", choices=["none", "row_only", "column_only", "row_column"])
    parser.add_argument("--no_coordinate_transformer", action="store_true")
    parser.add_argument("--use_position", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--aux_residual_weight", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--loss_pos_weight_mode", default="prior", choices=["none", "manual", "prior"])
    parser.add_argument("--pos_weight", type=float, default=1.0)
    parser.add_argument("--hfree_uncertain_K", type=int, default=10)
    parser.add_argument("--hfree_threshold", type=float, default=0.5)
    parser.add_argument("--score_type", default="squared", choices=["squared", "absolute", "gaussian"])
    parser.add_argument("--posterior_weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser
