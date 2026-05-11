import argparse
from pathlib import Path
from types import SimpleNamespace

from torchvision.utils import save_image
from tqdm import tqdm

from dataset import build_synthetic_lwe_datasets


def default_m(n: int) -> int:
    return 16 * n


def default_h_max(n: int) -> int:
    return max(2, n // 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump final synthetic raw1 LWE images.")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="visualized_dataset")
    parser.add_argument("--num-samples", "--num_samples", dest="num_samples", type=int, default=64)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--m", "--M", dest="m", type=int, default=None)
    parser.add_argument("--q", type=int, default=257)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--fixed-h", "--fixed_h", dest="fixed_h", type=int, default=2)
    parser.add_argument(
        "--noise-distribution",
        "--noise_distribution",
        dest="noise_distribution",
        choices=["discrete_gaussian", "bounded_integer"],
        default="discrete_gaussian",
    )
    parser.add_argument("--noise-bound", "--noise_bound", dest="noise_bound", type=int, default=None)
    parser.add_argument("--shared-a", "--shared_a", dest="shared_a", action="store_true")
    parser.add_argument(
        "--row-permutation",
        "--row_permutation",
        dest="row_permutation",
        choices=["none", "global", "per_sample"],
        default="none",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def dataset_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        m=args.m if args.m is not None else default_m(args.n),
        n=args.n,
        q=args.q,
        h_setting="fixed_h",
        p_nonzero=None,
        fixed_h=args.fixed_h,
        h_min=None,
        h_max=default_h_max(args.n),
        sigma=args.sigma,
        noise_distribution=args.noise_distribution,
        noise_bound=args.noise_bound,
        shared_a=args.shared_a,
        row_permutation=args.row_permutation,
        num_train=args.num_samples,
        num_val=1,
        num_test=1,
    )


def dump_all_images():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, _, _ = build_synthetic_lwe_datasets(dataset_args(args), args.seed)
    print(
        f"Dumping {len(train_dataset)} raw1 images to {output_dir} "
        f"(shape={tuple(train_dataset[0]['image'].shape)}, q={train_dataset.q})"
    )

    for idx in tqdm(range(len(train_dataset)), desc="dump raw1 images"):
        item = train_dataset[idx]
        save_image(item["image"], output_dir / f"sample_{idx:05d}.png")

    (output_dir / "metadata.txt").write_text(
        "\n".join(
            [
                "encoding=raw1_[A|b]/q",
                f"num_samples={len(train_dataset)}",
                f"image_shape={tuple(train_dataset[0]['image'].shape)}",
                f"n={train_dataset.n}",
                f"m={train_dataset.m}",
                f"q={train_dataset.q}",
                "",
            ]
        )
    )
    print(f"Done. Wrote raw1 images to {output_dir}")

if __name__ == "__main__":
    dump_all_images()
