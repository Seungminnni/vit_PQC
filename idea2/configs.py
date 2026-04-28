from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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
    integer_values: tuple[int, ...] = (-3, -2, -1, 1, 2, 3)
    secret_split: bool = False
    train_secret_fraction: float = 0.8
    split_seed: int = 1234


@dataclass
class FeatureConfig:
    encoder_type: str = "rhie_cip"
    freqs: tuple[int, ...] = (1, 2, 4)
    include_raw: bool = True
    include_phase: bool = True
    include_rhie: bool = True
    include_interaction: bool = True
    include_stats: bool = True
    include_pair_after_topk: bool = False


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


@dataclass
class CandidateConfig:
    topK: int = 8
    value_topr: int = 2
    beam_width: int = 64
    score_type: str = "squared"
    use_pair_filter: bool = False
    pair_budget: int = 64
    pair_score_weight: float = 0.0
    posterior_weight: float = 0.0


@dataclass
class ExperimentConfig:
    lwe: LWEConfig
    features: FeatureConfig
    model: ModelConfig
    train: TrainConfig
    candidate: CandidateConfig
    run_name: str = "rhie_cg"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dimension_run_name(lwe: LWEConfig, train: TrainConfig | None = None) -> str:
    sigma = f"{lwe.sigma_e:g}".replace("-", "m").replace(".", "p")
    name = f"n{lwe.n}_m{lwe.M}_q{lwe.q}_h{lwe.h}_e{sigma}"
    if train is not None:
        name = f"{name}_s{train.steps * train.batch_size}"
    return name


PRESETS: dict[str, dict[str, Any]] = {
    "stage0": {"n": 8, "M": 32, "q": 127, "h": 1, "sigma_e": 0.0, "topK": 4},
    "stage1": {"n": 10, "M": 64, "q": 127, "h": 1, "sigma_e": 0.0, "topK": 4},
    "stage1_h1": {"n": 10, "M": 64, "q": 127, "h": 1, "sigma_e": 0.0, "topK": 4},
    "stage2": {"n": 10, "M": 128, "q": 127, "h": 2, "sigma_e": 0.0, "topK": 6},
    "stage2_h2": {"n": 10, "M": 128, "q": 127, "h": 2, "sigma_e": 0.0, "topK": 6},
    "stage3": {"n": 16, "M": 128, "q": 127, "h": 3, "sigma_e": 0.0, "topK": 8},
    "stage3_h3": {"n": 16, "M": 128, "q": 127, "h": 3, "sigma_e": 0.0, "topK": 8},
    "noise": {"n": 16, "M": 256, "q": 127, "h": 3, "sigma_e": 1.0, "topK": 8},
    "ternary": {"n": 16, "M": 512, "q": 127, "h": 3, "sigma_e": 1.0, "topK": 8},
    "integer": {"n": 16, "M": 512, "q": 127, "h": 3, "sigma_e": 1.0, "topK": 8},
}


def build_config(args: argparse.Namespace, secret_type: str = "binary") -> ExperimentConfig:
    preset = PRESETS.get(args.preset, PRESETS["stage3"]).copy()
    lwe = LWEConfig(
        n=int(args.n if args.n is not None else preset["n"]),
        M=int(args.M if args.M is not None else preset["M"]),
        q=int(args.q if args.q is not None else preset["q"]),
        h=int(args.h if args.h is not None else preset["h"]),
        sigma_e=float(args.sigma_e if args.sigma_e is not None else preset["sigma_e"]),
        secret_type=secret_type,
        secret_split=args.secret_split,
        train_secret_fraction=args.train_secret_fraction,
        split_seed=args.split_seed,
    )
    features = FeatureConfig(
        encoder_type=args.encoder_type,
        freqs=tuple(int(x) for x in args.freqs.split(",")),
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
    )
    candidate = CandidateConfig(
        topK=int(args.topK if args.topK is not None else preset["topK"]),
        value_topr=args.value_topr,
        beam_width=args.beam_width,
        score_type=args.score_type,
        use_pair_filter=args.use_pair_filter,
        pair_budget=args.pair_budget,
        pair_score_weight=args.pair_score_weight,
        posterior_weight=args.posterior_weight,
    )
    run_name = args.run_name or dimension_run_name(lwe, train)
    return ExperimentConfig(lwe=lwe, features=features, model=model, train=train, candidate=candidate, run_name=run_name)


def config_from_dict(payload: dict[str, Any]) -> ExperimentConfig:
    def with_defaults(cls, raw: dict[str, Any]) -> dict[str, Any]:
        merged = asdict(cls())
        merged.update(raw)
        return merged

    lwe_raw = with_defaults(LWEConfig, payload["lwe"])
    features_raw = with_defaults(FeatureConfig, payload["features"])
    model_raw = with_defaults(ModelConfig, payload["model"])
    train_raw = with_defaults(TrainConfig, payload["train"])
    candidate_raw = with_defaults(CandidateConfig, payload["candidate"])
    if isinstance(lwe_raw.get("integer_values"), list):
        lwe_raw = {**lwe_raw, "integer_values": tuple(lwe_raw["integer_values"])}
    if isinstance(features_raw.get("freqs"), list):
        features_raw = {**features_raw, "freqs": tuple(features_raw["freqs"])}
    return ExperimentConfig(
        lwe=LWEConfig(**lwe_raw),
        features=FeatureConfig(**features_raw),
        model=ModelConfig(**model_raw),
        train=TrainConfig(**train_raw),
        candidate=CandidateConfig(**candidate_raw),
        run_name=payload.get("run_name", "rhie_cg"),
    )


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--preset", default="stage3")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--M", type=int, default=None)
    parser.add_argument("--q", type=int, default=None)
    parser.add_argument("--h", type=int, default=None)
    parser.add_argument("--sigma_e", type=float, default=None)
    parser.add_argument("--secret_split", action="store_true")
    parser.add_argument("--train_secret_fraction", type=float, default=0.8)
    parser.add_argument("--split_seed", type=int, default=1234)
    parser.add_argument(
        "--encoder_type",
        default="rhie_cip",
        choices=["a_only", "baseline_8ch", "baseline_10ch", "baseline_14ch", "rhie_cip", "rhie_cip_ternary", "rhie_cip_integer"],
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
    parser.add_argument("--topK", type=int, default=None)
    parser.add_argument("--value_topr", type=int, default=2)
    parser.add_argument("--beam_width", type=int, default=64)
    parser.add_argument("--score_type", default="squared", choices=["squared", "absolute", "gaussian"])
    parser.add_argument("--use_pair_filter", action="store_true")
    parser.add_argument("--pair_budget", type=int, default=64)
    parser.add_argument("--pair_score_weight", type=float, default=0.0)
    parser.add_argument("--posterior_weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser
