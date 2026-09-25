"""Unit tests verifying Stage 1 pure SSL federated contracts."""
import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils.config import load_config
from src.models.mae import build_mae
from src.client.ssl_train import ssl_local_train
from src.server.server import FederatedServer
from src.server.aggregator import fedavg, fedprox
from src.federated.simulation import _build_synthetic_hospital_loaders


class Stage1ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("configs/default.yaml")

    def test_stage1_local_train_pure_ssl(self):
        """Verify ssl_local_train produces encoder gradients but leaves proto_head untouched."""
        model = build_mae(self.config)
        loader = DataLoader(
            TensorDataset(torch.randn(4, 3, 224, 224)),
            batch_size=2,
            shuffle=False,
        )
        
        result = ssl_local_train(
            hospital_id=1,
            model=model,
            dataloader=loader,
            config=self.config,
            device=torch.device("cpu"),
        )
        
        # Check return dict
        self.assertIn("ssl_loss", result)
        self.assertIn("encoder_weights", result)
        self.assertNotIn("proto_loss", result)
        self.assertNotIn("mae_loss", result)
        
        # Check that encoder weights are present
        enc_weights = result["encoder_weights"]
        self.assertGreater(len(enc_weights), 100)
        
        # Check gradients: active encoder parameters must have gradients
        enc_grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
        active_enc_grads = [g for g in enc_grads if g is not None]
        self.assertGreater(len(active_enc_grads), 100)
        self.assertGreater(sum(g.abs().sum().item() for g in active_enc_grads), 0.0)
        
        # Check proto head gradients: MUST be None (no gradient calculated)
        proto_grads = [p.grad for p in model.proto_head.parameters()]
        for g in proto_grads:
            self.assertIsNone(g, "Proto head parameter should not receive any gradients during Stage 1.")

    def test_server_encoder_only_aggregation(self):
        """Verify FederatedServer broadcasts and updates only encoder weights for Stage 1."""
        server = FederatedServer(self.config)
        server.initialize_global_model()
        
        # Broadcast should return encoder weights dict
        broadcast_weights = server.broadcast()
        self.assertIsInstance(broadcast_weights, dict)
        self.assertTrue(all(k.startswith("vit.") or k.startswith("proj.") or not k.startswith("decoder.") for k in broadcast_weights.keys()))
        self.assertFalse(any("decoder" in k for k in broadcast_weights.keys()))
        self.assertFalse(any("proto_head" in k for k in broadcast_weights.keys()))
        
        # Aggregation of encoder weights across 2 clients
        w1 = {k: v.clone() for k, v in broadcast_weights.items()}
        w2 = {k: v.clone() + 1.0 for k, v in broadcast_weights.items()}
        
        agg_fedavg = fedavg([w1, w2], [10, 10])
        self.assertEqual(set(agg_fedavg.keys()), set(broadcast_weights.keys()))
        
        # Update server
        server.update_global_model(agg_fedavg)
        
        # FedProx aggregation
        agg_fedprox = fedprox(broadcast_weights, [w1, w2], [10, 10], mu=0.01)
        self.assertEqual(set(agg_fedprox.keys()), set(broadcast_weights.keys()))

    def test_synthetic_hospital_loaders_stage1(self):
        """Verify synthetic loaders return pure NIH dataloaders without support loaders."""
        loaders = _build_synthetic_hospital_loaders(num_hospitals=3, batch_size=4, image_size=224)
        self.assertEqual(len(loaders), 3)
        # Each hospital has a single DataLoader, not a (loader, support_loader) tuple
        for l in loaders:
            self.assertIsInstance(l, DataLoader)

    def test_active_simulation_uses_ssl_train_not_flame(self):
        """Verify simulation.py uses ssl_local_train and does not import or invoke flame_local_train."""
        import src.federated.simulation as sim
        from unittest.mock import patch
        
        # Verify module imports
        self.assertTrue(hasattr(sim, "ssl_local_train"))
        self.assertFalse(hasattr(sim, "flame_local_train"))
        
        # Verify _train_sequential calls ssl_local_train
        with patch("src.federated.simulation.ssl_local_train") as mock_ssl_train:
            mock_ssl_train.return_value = {
                "encoder_weights": {},
                "num_samples": 10,
                "epoch_losses": [1.0],
                "ssl_loss": 1.0,
            }
            dummy_model = build_mae(self.config)
            dummy_loader = DataLoader(TensorDataset(torch.randn(2, 3, 224, 224)))
            results = sim._train_sequential(
                global_model=dummy_model,
                global_weights={},
                hospital_loaders=[dummy_loader],
                config=self.config,
                device=torch.device("cpu"),
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(mock_ssl_train.call_count, 1)


if __name__ == "__main__":
    unittest.main()
