import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lwe_vit import (  # noqa: E402
    EquationLWETransformer,
    EquationTransformerConfig,
    LWEImageEncoder,
    LWEParams,
    LWEDatasetSpec,
    OnTheFlySyntheticLWEDataset,
    RectangularPatchTokenizer,
    RepresentationConfig,
    RowLocalCNNLWEConfig,
    RowLocalCNNLWEModel,
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

    def test_patch_tokenizer_forward(self) -> None:
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
                residue_encoding="phase",
                embed_dim=32,
                depth=1,
                num_heads=4,
            )
        )
        out = model(dataset.sample.A[:2], dataset.sample.b[:2])
        self.assertEqual(tuple(out.s_logits.shape), (2, params.n, 2))
        self.assertEqual(tuple(out.residual_score.shape), (2,))

    def test_row_block_model_forward_with_raw_residues(self) -> None:
        params = LWEParams(n=8, m=12, q=17, secret_dist="binary", noise_dist="zero", seed=13)
        sample = sample_lwe_batch(params, batch_size=2)
        model = RowBlockLWETransformer(
            RowBlockLWEConfig(
                n=params.n,
                m=params.m,
                q=params.q,
                num_secret_classes=num_secret_classes(params),
                block_rows=1,
                block_cols=4,
                residue_encoding="raw",
                embed_dim=32,
                depth=1,
                num_heads=4,
            )
        )
        out = model(sample.A, sample.b)
        self.assertEqual(tuple(out.s_logits.shape), (2, params.n, 2))
        self.assertEqual(tuple(out.residual_score.shape), (2,))

    def test_equation_transformer_forward_with_raw_residues(self) -> None:
        params = LWEParams(n=8, m=12, q=17, secret_dist="binary", noise_dist="zero", seed=14)
        sample = sample_lwe_batch(params, batch_size=2)
        model = EquationLWETransformer(
            EquationTransformerConfig(
                n=params.n,
                m=params.m,
                q=params.q,
                num_secret_classes=num_secret_classes(params),
                residue_encoding="raw",
                embed_dim=32,
                depth=1,
                num_heads=4,
            )
        )
        out = model(sample.A, sample.b)
        self.assertEqual(tuple(out.s_logits.shape), (2, params.n, 2))
        self.assertEqual(tuple(out.residual_score.shape), (2,))

    def test_row_local_cnn_forward_with_raw_residues(self) -> None:
        params = LWEParams(n=8, m=12, q=17, secret_dist="binary", noise_dist="zero", seed=15)
        sample = sample_lwe_batch(params, batch_size=2)
        model = RowLocalCNNLWEModel(
            RowLocalCNNLWEConfig(
                n=params.n,
                m=params.m,
                q=params.q,
                num_secret_classes=num_secret_classes(params),
                residue_encoding="raw",
                embed_dim=32,
                depth=1,
            )
        )
        out = model(sample.A, sample.b)
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

    def test_residual_consistency_loss_prefers_correct_secret(self) -> None:
        params = LWEParams(n=6, m=12, q=257, secret_dist="binary", noise_dist="zero", seed=21)
        dataset = SyntheticLWEDataset(
            LWEDatasetSpec(
                num_samples=8,
                params=params,
                representation=RepresentationConfig(name="relation_grid"),
                h_setting="fixed_h",
                fixed_h=2,
            )
        )
        sample = dataset.sample
        correct_logits = torch.full((8, params.n, 2), -40.0)
        correct_logits.scatter_(-1, sample.s_labels.unsqueeze(-1), 40.0)
        wrong_logits = torch.full_like(correct_logits, -40.0)
        wrong_logits[:, :, 0] = 40.0

        correct_loss = residual_consistency_loss(
            sample.A,
            sample.b,
            correct_logits,
            q=params.q,
            secret_dist=params.secret_dist,
            noise_bound=0.5,
        )
        wrong_loss = residual_consistency_loss(
            sample.A,
            sample.b,
            wrong_logits,
            q=params.q,
            secret_dist=params.secret_dist,
            noise_bound=0.5,
        )

        self.assertLess(correct_loss.item(), 1e-8)
        self.assertGreater(wrong_loss.item(), correct_loss.item())

    def test_residual_metrics_reject_wrong_secret(self) -> None:
        params = LWEParams(n=8, m=32, q=257, secret_dist="binary", noise_dist="zero", seed=22)
        dataset = SyntheticLWEDataset(
            LWEDatasetSpec(
                num_samples=16,
                params=params,
                representation=RepresentationConfig(name="relation_grid"),
                h_setting="fixed_h",
                fixed_h=2,
            )
        )
        sample = dataset.sample
        wrong_logits = torch.full((16, params.n, 2), -20.0)
        wrong_logits[:, :, 0] = 20.0

        batch_stats = batch_statistics(
            logits=wrong_logits,
            target_labels=sample.s_labels,
            secret=sample.s,
            A=sample.A,
            b=sample.b,
            oracle_residual=sample.e,
            q=params.q,
            noise_width=1.0,
            residual_success_factor=2.0,
            secret_dist=params.secret_dist,
        )
        metrics = finalize_statistics(merge_statistics([batch_stats]), num_classes=2)

        self.assertEqual(metrics["exact_match"], 0.0)
        self.assertEqual(metrics["support_recall"], 0.0)
        self.assertLess(metrics["residual_success_rate"], 1.0)
        self.assertGreater(metrics["pred_residual_std_mean"], metrics["oracle_residual_std_mean"])

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

    def test_on_the_fly_dataset_is_deterministic_and_split_separated(self) -> None:
        params = LWEParams(n=8, m=16, q=257, secret_dist="binary", noise_dist="discrete_gaussian", noise_width=1.0, seed=11)
        spec = LWEDatasetSpec(
            num_samples=1000000,
            params=params,
            representation=RepresentationConfig(name="relation_grid"),
            return_image=False,
            h_setting="fixed_h",
            fixed_h=2,
        )
        dataset = OnTheFlySyntheticLWEDataset(spec)

        first = dataset[123]
        again = dataset[123]
        for key in ("A", "b", "secret", "target", "noise", "oracle_residual"):
            self.assertTrue(torch.equal(first[key], again[key]), key)

        other_split = OnTheFlySyntheticLWEDataset(
            LWEDatasetSpec(
                num_samples=1000000,
                params=LWEParams(
                    n=8,
                    m=16,
                    q=257,
                    secret_dist="binary",
                    noise_dist="discrete_gaussian",
                    noise_width=1.0,
                    seed=23,
                ),
                representation=RepresentationConfig(name="relation_grid"),
                return_image=False,
                h_setting="fixed_h",
                fixed_h=2,
            )
        )
        self.assertFalse(torch.equal(first["A"], other_split[123]["A"]))
        self.assertTrue(torch.equal(residual_from_secret(first["A"], first["b"], first["secret"], params.q).squeeze(0), first["noise"]))

    def test_on_the_fly_dataset_statistics_are_analytic(self) -> None:
        params = LWEParams(n=16, m=512, q=257, secret_dist="binary", noise_dist="discrete_gaussian", noise_width=1.0, seed=11)
        dataset = OnTheFlySyntheticLWEDataset(
            LWEDatasetSpec(
                num_samples=1000000,
                params=params,
                representation=RepresentationConfig(name="relation_grid"),
                return_image=False,
                h_setting="fixed_h",
                fixed_h=2,
            )
        )

        stats = dataset_statistics(dataset)

        self.assertEqual(stats["avg_h"], 2.0)
        self.assertEqual(stats["std_h"], 0.0)
        self.assertEqual(stats["class_prob_0"], 0.875)
        self.assertEqual(stats["class_prob_1"], 0.125)
        self.assertEqual(stats["nonzero_rate"], 0.125)


if __name__ == "__main__":
    unittest.main()
