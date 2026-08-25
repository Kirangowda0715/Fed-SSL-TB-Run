"""Focused checks for the federated FLAME contracts."""
import torch
import unittest

from src.client.flame_local_train import joint_loss
from src.server.aggregator import fedavg


class FlameArchitectureTests(unittest.TestCase):
    def test_joint_loss_weights(self):
        self.assertTrue(torch.isclose(joint_loss(torch.tensor(2.0), torch.tensor(1.0)), torch.tensor(1.7)))


    def test_proto_gradient_reaches_encoder(self):
        encoder = torch.nn.Linear(4, 3)
        head = torch.nn.Linear(3, 2)
        features = encoder(torch.randn(4, 4))
        loss = torch.nn.functional.cross_entropy(head(features), torch.tensor([0, 1, 0, 1]))
        loss.backward()
        self.assertIsNotNone(encoder.weight.grad)
        self.assertGreater(encoder.weight.grad.abs().sum(), 0)


    def test_full_model_aggregation(self):
        local = [
        {"encoder": {"weight": torch.tensor([1.0])}, "decoder": {"weight": torch.tensor([3.0])}, "proto_head": {"weight": torch.tensor([5.0])}},
        {"encoder": {"weight": torch.tensor([3.0])}, "decoder": {"weight": torch.tensor([5.0])}, "proto_head": {"weight": torch.tensor([7.0])}},
    ]
        result = fedavg(local, [1, 3])
        self.assertEqual(set(result), {"encoder", "decoder", "proto_head"})
        self.assertEqual(result["encoder"]["weight"].item(), 2.5)
