import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lwe_vit import (  # noqa: E402
    LWEImageEncoder,
    LWEParams,
    LWEDatasetSpec,
    LWEViTConfig,
    LWEViTForSecret,
    PairTokenLWEConfig,
    PairTokenLWETransformer,
    RectangularPatchTokenizer,
    RepresentationConfig,
    RowBlockLWEConfig,
    RowBlockLWETransformer,
    SyntheticLWEDataset,
    batch_statistics,
    dataset_statistics,
    finalize_statistics,
    centered_mod,
    merge_statistics,
    num_secret_classes,
    residual_consistency_loss,
    residual_from_secret,
    sample_lwe_batch,
)


class LWEViTTests(unittest.TestCase):
    def test_relation_grid_shape_rhs_and_broadcast(self) -> None:
        q = 17
        A = torch.arange(30, dtype=torch.long).reshape(6, 5) % q
        b = torch.tensor([0, 1, 8, 9, 16, 5], dtype=torch.long)
        config = RepresentationConfig(name="relation_grid", patch_rows=4, patch_cols=4)
        encoder = LWEImageEncoder(config, q=q)

        image, mask = encoder.encode(A, b)

        self.assertEqual(tuple(image.shape), (1, 7, 8, 8))
        self.assertEqual(tuple(mask.shape), (1, 8, 8))
        self.assertTrue(mask[0, :6, :6].all().item())
        self.assertFalse(mask[0, 6:, :].any().item())
        self.assertFalse(mask[0, :, 6:].any().item())

        names = encoder.channel_names()
        b_idx = names.index("b_centered_broadcast")
        rhs_idx = names.index("is_rhs_column")
        expected_b = centered_mod(b, q).float() / (q / 2.0)
        torch.testing.assert_close(image[0, b_idx, :6, 0], expected_b)
        self.assertTrue(torch.equal(image[0, rhs_idx, :6, 5], torch.ones(6)))
        self.assertTrue(torch.equal(image[0, rhs_idx, :6, :5], torch.zeros(6, 5)))

    def test_row_equation_tokens_shape(self) -> None:
        q = 17
        A = torch.arange(12, dtype=torch.long).reshape(3, 4) % q
        b = torch.tensor([3, 4, 5], dtype=torch.long)
        config = RepresentationConfig(
            name="row_equation_tokens",
            patch_rows=1,
            patch_cols=1,
            use_phase=False,
        )
        encoder = LWEImageEncoder(config, q=q)

        image, mask = encoder.encode(A, b)

        self.assertEqual(tuple(image.shape), (1, 5, 3, 1))
        self.assertTrue(mask.all().item())
        expected_b = centered_mod(b, q).float() / (q / 2.0)
        torch.testing.assert_close(image[0, 4, :, 0], expected_b)

    def test_residual_matches_sampled_noise(self) -> None:
        params = LWEParams(
            n=8,
            m=12,
            q=257,
            secret_dist="ternary",
            noise_dist="uniform_small",
            noise_width=2,
            seed=123,
        )
        sample = sample_lwe_batch(params, batch_size=4)

        residual = residual_from_secret(sample.A, sample.b, sample.s, params.q)

        self.assertTrue(torch.equal(residual, sample.e))

    def test_patch_tokenizer_and_model_forward(self) -> None:
        params = LWEParams(n=8, m=16, q=17, secret_dist="binary", noise_dist="zero", seed=1)
        sample = sample_lwe_batch(params, batch_size=2)
        rep = RepresentationConfig(name="relation_grid", patch_rows=4, patch_cols=4)
        encoder = LWEImageEncoder(rep, q=params.q)
        image, mask = encoder.encode(sample.A, sample.b)

        tokenizer = RectangularPatchTokenizer(
            in_channels=encoder.num_channels(),
            embed_dim=32,
            patch_rows=rep.patch_rows,
            patch_cols=rep.patch_cols,
        )
        patches = tokenizer(image, mask)
        self.assertEqual(patches.tokens.shape[:2], (2, 12))
        self.assertTrue(patches.mask.all().item())

        model = LWEViTForSecret(
            LWEViTConfig(
                n=params.n,
                q=params.q,
                in_channels=encoder.num_channels(),
                num_secret_classes=num_secret_classes(params),
                patch_rows=rep.patch_rows,
                patch_cols=rep.patch_cols,
                embed_dim=32,
                depth=1,
                num_heads=4,
            )
        )
        out = model(image, mask)
        self.assertEqual(tuple(out.s_logits.shape), (2, params.n, 2))
        self.assertEqual(tuple(out.residual_score.shape), (2,))

    def test_pair_token_model_forward_without_image_encoding(self) -> None:
        params = LWEParams(n=6, m=10, q=17, secret_dist="binary", noise_dist="zero", seed=11)
        dataset = SyntheticLWEDataset(
            LWEDatasetSpec(
                num_samples=3,
                params=params,
                representation=RepresentationConfig(name="relation_grid"),
                return_image=False,
                h_setting="fixed_h",
                fixed_h=2,
            )
        )
        item = dataset[0]
        self.assertNotIn("image", item)
        model = PairTokenLWETransformer(
            PairTokenLWEConfig(
                n=params.n,
                m=params.m,
                q=params.q,
                num_secret_classes=num_secret_classes(params),
                embed_dim=32,
                depth=1,
                num_heads=4,
            )
        )
        out = model(dataset.sample.A[:2], dataset.sample.b[:2])
        self.assertEqual(tuple(out.s_logits.shape), (2, params.n, 2))
        self.assertEqual(tuple(out.residual_score.shape), (2,))

    def test_row_block_model_forward_without_image_encoding(self) -> None:
        params = LWEParams(n=8, m=12, q=17, secret_dist="binary", noise_dist="zero", seed=12)
        dataset = SyntheticLWEDataset(
            LWEDatasetSpec(
                num_samples=3,
                params=params,
                representation=RepresentationConfig(name="relation_grid"),
                return_image=False,
                h_setting="fixed_h",
                fixed_h=2,
            )
        )
        item = dataset[0]
        self.assertNotIn("image", item)
        model = RowBlockLWETransformer(
            RowBlockLWEConfig(
                n=params.n,
                m=params.m,
                q=params.q,
                num_secret_classes=num_secret_classes(params),
                block_rows=1,
                block_cols=4,
                fourier_k=2,
                embed_dim=32,
                depth=1,
                num_heads=4,
            )
        )
        out = model(dataset.sample.A[:2], dataset.sample.b[:2])
        self.assertEqual(tuple(out.s_logits.shape), (2, params.n, 2))
        self.assertEqual(tuple(out.residual_score.shape), (2,))

    def test_residual_consistency_loss_runs(self) -> None:
        params = LWEParams(n=5, m=7, q=17, secret_dist="binary", noise_dist="zero", seed=9)
        sample = sample_lwe_batch(params, batch_size=3)
        logits = torch.zeros(3, params.n, num_secret_classes(params))

        loss = residual_consistency_loss(
            sample.A,
            sample.b,
            logits,
            q=params.q,
            secret_dist=params.secret_dist,
            noise_bound=1.0,
        )

        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss).item())

    def test_dataset_and_metrics_perfect_prediction(self) -> None:
        params = LWEParams(n=8, m=16, q=257, secret_dist="binary", noise_dist="zero", seed=5)
        rep = RepresentationConfig(name="relation_grid", patch_rows=4, patch_cols=4)
        dataset = SyntheticLWEDataset(
            LWEDatasetSpec(
                num_samples=12,
                params=params,
                representation=rep,
                h_setting="fixed_h",
                fixed_h=2,
            )
        )
        stats = dataset_statistics(dataset)
        self.assertAlmostEqual(stats["nonzero_rate"], 0.25)

        sample = dataset.sample
        logits = torch.full((12, params.n, 2), -10.0)
        logits.scatter_(-1, sample.s_labels.unsqueeze(-1), 10.0)
        batch_stats = batch_statistics(
            logits=logits,
            target_labels=sample.s_labels,
            secret=sample.s,
            A=sample.A,
            b=sample.b,
            oracle_residual=sample.e,
            q=params.q,
            noise_width=params.noise_width,
            residual_success_factor=2.0,
            secret_dist=params.secret_dist,
        )
        metrics = finalize_statistics(merge_statistics([batch_stats]), num_classes=2)
        self.assertEqual(metrics["coord_acc"], 1.0)
        self.assertEqual(metrics["exact_match"], 1.0)
        self.assertEqual(metrics["support_f1"], 1.0)
        self.assertEqual(metrics["residual_success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
