"""Offline checks for numerical behavior and the complete experiment pipeline."""

from dataclasses import replace
import json
import math
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset, TensorDataset

from adaptive_entropy import calculate_adaptive_lambda
from client import (
    FederatedClient, current_learning_rate, legacy_noise_std,
    perturb_soft_labels, prediction_entropy,
)
from data import DistillationDataset
from data_partition import dirichlet_partition, get_targets, split_public_private
from main import DATASET_DEFAULTS, build_config, run_experiment, validate_config
from models import ClientModel
from server import aggregate_soft_labels, communication_bytes


class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_global_upper_endpoint_and_inverse_gap_scaling(self):
        config = build_config()
        values = [torch.full((4,), 0.1 * index) for index in range(10)]
        result = calculate_adaptive_lambda(values, config)
        self.assertEqual((result.lower_rank, result.upper_rank), (3, 8))
        self.assertEqual((result.low_group_size, result.high_group_size), (3, 3))
        self.assertAlmostEqual(result.representative_gap, 0.5)
        self.assertAlmostEqual(result.lambda_value, math.log(10) / 0.5, places=5)
        self.assertAlmostEqual(result.lambda_lower, math.log(2) / 0.5, places=5)
        doubled = calculate_adaptive_lambda([2 * item for item in values], config)
        self.assertAlmostEqual(doubled.lambda_value, result.lambda_value / 2, places=5)
        self.assertFalse(result.used_fallback)

    def test_sample_quantile_not_average_and_clipping(self):
        config = build_config()
        gaps = torch.tensor([0.2, 0.4, 0.6, 0.8])
        values = [gaps * index / 5 for index in range(10)]
        result = calculate_adaptive_lambda(values, config)
        self.assertAlmostEqual(result.representative_gap, 0.26, places=6)
        self.assertAlmostEqual(result.lambda_value, math.log(10) / 0.26, places=5)
        tiny = calculate_adaptive_lambda([item * 0.001 for item in values], config)
        self.assertEqual(tiny.lambda_value, 20.0)
        self.assertGreater(tiny.raw_lambda_upper, tiny.lambda_upper)

    def test_ties_and_single_client_use_clipped_fallback(self):
        config = build_config(entropy_weight_scale=3.0)
        for values in ([torch.ones(3)] * 10, [torch.ones(3)]):
            result = calculate_adaptive_lambda(values, config)
            self.assertTrue(result.used_fallback)
            self.assertEqual(result.lambda_value, 3.0)
        result = calculate_adaptive_lambda(
            [torch.ones(3)] * 10, replace(config, entropy_weight_scale=100.0)
        )
        self.assertEqual(result.lambda_value, 20.0)

    def test_invalid_interval_is_rejected_before_training(self):
        config = build_config(adaptive_lambda_min_ratio=11.0)
        with self.assertRaisesRegex(ValueError, "Empty lambda interval"):
            validate_config(config)
        with self.assertRaises(ValueError):
            validate_config(build_config(dirichlet_alpha=0.0))

    def test_legacy_noise_equivalence_without_clipping(self):
        config = build_config()
        probabilities = torch.tensor([[0.99, 0.01], [0.3, 0.7]])
        sigma = 0.025 * math.sqrt(2 * math.log(1.25 / 1e-5)) / 1.5
        self.assertAlmostEqual(legacy_noise_std(config), sigma, places=12)
        for name in DATASET_DEFAULTS:
            with self.subTest(dataset=name):
                self.assertAlmostEqual(legacy_noise_std(build_config(name)), sigma, places=12)
        torch.manual_seed(43)
        noise = torch.normal(0, sigma, size=probabilities.shape)
        expected = torch.softmax((probabilities + noise) / 2, dim=1)
        torch.manual_seed(43)
        actual = perturb_soft_labels(probabilities, config)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual.sum(dim=1), torch.ones(2))
        self.assertTrue(torch.all(actual >= 0))
        entropy = prediction_entropy(torch.full((3, 10), 0.1))
        torch.testing.assert_close(entropy, torch.full((3,), math.log(10)))

    def test_aggregation_matches_samplewise_reference(self):
        predictions = [
            torch.tensor([[0.8, 0.2], [0.4, 0.6]]),
            torch.tensor([[0.1, 0.9], [0.9, 0.1]]),
        ]
        entropies = [torch.tensor([0.2, 0.7]), torch.tensor([0.8, 0.1])]
        expected = []
        for index in range(2):
            weights = np.exp(-3 * np.array([item[index].item() for item in entropies]))
            weights /= weights.sum()
            mixture = sum(float(w) * p[index] for w, p in zip(weights, predictions))
            expected.append(torch.softmax(mixture / 2, dim=0))
        soft, hard = aggregate_soft_labels(predictions, entropies, 3.0, 2.0)
        torch.testing.assert_close(soft, torch.stack(expected))
        self.assertEqual(hard.tolist(), [0, 0])
        upload, download = communication_bytes(predictions, entropies, soft, hard, 2)
        self.assertEqual(upload, 2 * (2 * 2 * 4 + 2 * 4))
        self.assertEqual(download, 2 * (2 * 2 * 4 + 2 * 8))

    def test_adam_and_manuscript_learning_rate_schedule(self):
        for name in DATASET_DEFAULTS:
            with self.subTest(dataset=name):
                config = build_config(name, device="cpu")
                self.assertEqual(config.optimizer, "adam")
                client = FederatedClient(0, ClientModel(config), None, config)
                for round_number in (1, 2, 20, 45, 46, 50, 100):
                    expected = max(1e-4 * 0.95 ** (round_number - 1), 1e-5)
                    self.assertAlmostEqual(
                        current_learning_rate(config, round_number), expected, places=14
                    )
                    optimizer = client._optimizer(round_number)
                    self.assertIsInstance(optimizer, torch.optim.Adam)
                    for group in optimizer.param_groups:
                        self.assertAlmostEqual(group["lr"], expected, places=14)

    def test_partition_is_disjoint_complete_and_reproducible(self):
        dataset = TensorDataset(torch.arange(620), torch.arange(620) % 62)
        nested = Subset(Subset(dataset, list(range(10, 610))), list(range(5, 595)))
        np.testing.assert_array_equal(get_targets(nested), np.arange(15, 605) % 62)
        private, public = split_public_private(nested, 0.05, 42)
        self.assertEqual(len(public), int(0.05 * len(nested)))
        self.assertFalse(set(private.indices) & set(public.indices))
        self.assertEqual(set(private.indices) | set(public.indices), set(range(len(nested))))
        for alpha in (0.01, 0.03, 0.1):
            for seed in (42, 43, 44, 45, 46):
                with self.subTest(alpha=alpha, seed=seed):
                    clients = dirichlet_partition(private, 10, 62, alpha, seed, 100)
                    repeated = dirichlet_partition(private, 10, 62, alpha, seed, 100)
                    indices = [index for client in clients for index in client.indices]
                    self.assertEqual(sorted(indices), list(range(len(private))))
                    self.assertTrue(all(len(client) > 0 for client in clients))
                    self.assertEqual(
                        [client.indices for client in clients],
                        [client.indices for client in repeated],
                    )

    def test_public_images_stay_aligned_with_teacher_targets(self):
        public = TensorDataset(torch.arange(5).reshape(5, 1), torch.zeros(5))
        soft = torch.arange(10).reshape(5, 2)
        hard = torch.arange(5)
        dataset = DistillationDataset(public, soft, hard)
        for index in (4, 0, 3, 1, 2):
            image, soft_target, hard_target = dataset[index]
            self.assertEqual(image.item(), hard_target.item())
            torch.testing.assert_close(soft_target, soft[index])

    def test_model_shapes_and_default_public_ratios(self):
        for name in DATASET_DEFAULTS:
            with self.subTest(dataset=name):
                config = build_config(name)
                model = ClientModel(config).eval()
                conv_count = sum(isinstance(layer, torch.nn.Conv2d) for layer in model.modules())
                self.assertEqual(conv_count, 4 if name == "svhn" else 3)
                if name == "emnist":
                    self.assertEqual(sum(p.numel() for p in model.parameters()), 757118)
                with torch.no_grad():
                    logits = model(torch.zeros(
                        2, config.input_channels, config.image_size, config.image_size
                    ))
                self.assertEqual(tuple(logits.shape), (2, config.num_classes))
                self.assertEqual(config.public_ratio, 0.05 if name == "emnist" else 0.1)
                self.assertTrue(config.adaptive_lambda_enabled)


class PipelineTests(unittest.TestCase):
    def test_all_datasets_complete_a_synthetic_round_and_export(self):
        with tempfile.TemporaryDirectory(prefix="fededp_tests_") as folder:
            for name in DATASET_DEFAULTS:
                with self.subTest(dataset=name):
                    config = build_config(
                        name, smoke_test=True, num_clients=2, num_rounds=1,
                        local_epochs=1, batch_size=8, public_batch_size=8,
                        test_batch_size=8, device="cpu", cpu_threads=2,
                        results_dir=folder, dirichlet_alpha=0.1,
                    )
                    path, last = run_experiment(config)
                    self.assertTrue(np.isfinite(last["mean_test_loss"]))
                    self.assertTrue(0 <= last["mean_accuracy_pct"] <= 100)
                    samples = int(config.smoke_train_size * config.public_ratio)
                    self.assertEqual(
                        last["upload_bytes"], 2 * samples * (config.num_classes + 1) * 4
                    )
                    self.assertEqual(
                        last["download_bytes"], 2 * samples * (config.num_classes * 4 + 8)
                    )
                    self.assertEqual(
                        last["cumulative_total_bytes"],
                        last["upload_bytes"] + last["download_bytes"],
                    )
                    self.assertGreater((path / "training_curves.png").stat().st_size, 1000)
                    summary = json.loads((path / "summary.json").read_text())
                    self.assertTrue(summary["smoke_test"])
                    self.assertEqual(summary["final_accuracy_pct"], last["mean_accuracy_pct"])
                    with pd.ExcelFile(path / "results.xlsx") as workbook:
                        self.assertIn("Adaptive entropy", workbook.sheet_names)
                        self.assertEqual(len(pd.read_excel(workbook, "Clients")), 2)

    def test_seed_reproducibility_and_partial_fixed_lambda_run(self):
        with tempfile.TemporaryDirectory(prefix="fededp_repro_") as folder:
            config = build_config(
                smoke_test=True, num_clients=2, client_fraction=0.5, num_rounds=2,
                local_epochs=1, batch_size=16, device="cpu", cpu_threads=2,
                adaptive_lambda_enabled=False, entropy_weight_scale=3.0,
                results_dir=folder, save_models=True,
            )
            first, last_a = run_experiment(config)
            second, last_b = run_experiment(config)
            self.assertEqual(last_a, last_b)
            pd.testing.assert_frame_equal(
                pd.read_csv(first / "clients.csv"), pd.read_csv(second / "clients.csv")
            )
            rows = pd.read_csv(first / "clients.csv")
            self.assertEqual(rows["participated"].sum(), 2)
            self.assertEqual(len(rows), 4)
            self.assertEqual(last_a["lambda_value"], 3.0)
            self.assertEqual(
                last_a["cumulative_total_bytes"], 2 * last_a["total_bytes"]
            )
            self.assertEqual(len(list((first / "models").glob("*.pt"))), 2)


if __name__ == "__main__":
    unittest.main()
