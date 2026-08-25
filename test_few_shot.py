"""Regression tests for the canonical few-shot protocol."""

import csv
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.client.local_train import _sample_kshot
from src.datasets.loader import ShenzhenDataset, get_eval_transform
from src.models.proto_head import PrototypicalHead


class FewShotTests(unittest.TestCase):
    def test_sampling_is_exact_and_deterministic(self):
        labels = torch.tensor([0] * 8 + [1] * 9)
        support_a, query_a = _sample_kshot(labels, 5, seed=42)
        support_b, query_b = _sample_kshot(labels, 5, seed=42)
        self.assertTrue(torch.equal(support_a, support_b))
        self.assertTrue(torch.equal(query_a, query_b))
        self.assertEqual(len(support_a), 10)
        self.assertEqual(int((labels[support_a] == 0).sum()), 5)
        self.assertEqual(int((labels[support_a] == 1).sum()), 5)
        self.assertEqual(set(support_a.tolist()) & set(query_a.tolist()), set())

    def test_projected_prototype_mean_and_gradient(self):
        head = PrototypicalHead(embed_dim=4, projection_dim=3)
        support = torch.randn(4, 4, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        projected = head.projection(support)
        prototypes = head.compute_prototypes(support, labels)
        self.assertTrue(torch.allclose(prototypes[0], projected[:2].mean(0)))
        self.assertEqual(tuple(prototypes.shape), (2, 3))
        loss, _ = head.prototypical_loss(torch.randn(2, 4), torch.tensor([0, 1]), prototypes)
        loss.backward()
        self.assertIsNotNone(support.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in head.projection.parameters()))

    def test_metadata_labels_override_matching_folder_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Normal").mkdir()
            (root / "TB").mkdir()
            Image.new("L", (4, 4)).save(root / "Normal" / "normal-a.png")
            Image.new("L", (4, 4)).save(root / "TB" / "tb-b.png")
            with (root / "shenzhen_metadata.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["study_id", "findings"])
                writer.writeheader()
                writer.writerow({"study_id": "normal-a", "findings": "Normal"})
                writer.writerow({"study_id": "tb-b", "findings": "Tuberculosis"})
            dataset = ShenzhenDataset(str(root), transform=get_eval_transform(4), image_size=4)
            self.assertEqual(dataset.get_labels(), [0, 1])
            self.assertEqual(dataset.get_study_id(0), "normal-a")


if __name__ == "__main__":
    unittest.main()
