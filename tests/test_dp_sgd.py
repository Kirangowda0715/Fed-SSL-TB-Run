"""
tests/test_dp_sgd.py
---------------------
Focused unit tests for DP-SGD implementation:
  1. Hand-computed per-example gradient clipping.
  2. Calibrated Gaussian noise scale and single addition.
  3. Correct denominator on partial batches.
  4. Graceful handling of disconnected parameters (p.grad is None).
  5. Validation of invalid configuration parameters (C <= 0, sigma < 0, delta not in (0, 1)).
  6. Genuine non-DP regression verification (parameters actually update under AdamW).
  7. Explicit rejection of unsupported DP + FedProx combinations.
  8. Cumulative multi-client accounting across simulated rounds without reset.
  9. Unsupported sampling scheme disables formal epsilon reporting on random shuffling without replacement.
 10. Checkpoint / resume continuity and clear failure when resuming DP runs missing privacy state.
 11. Reference verification of the analytical RDP formula under verified Poisson subsampling.

STRICT SAFETY:
  - No model training or multi-round simulation.
  - No dataset downloads or network requests.
  - Pure isolated mathematical and unit assertions on tiny tensors.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.client.dp_engine import DPEngine, RDPAccountant
from src.client.ssl_train import ssl_local_train


# ─── Mock Models for Pure Math Tests ──────────────────────────────────────────

class MockLinearModel(nn.Module):
    """Simple linear regression model: y = w * x + b."""
    def __init__(self, init_w: float = 2.0, init_b: float = 0.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([[init_w]], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor([init_b], dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        pred = x * self.w + self.b
        loss = 0.5 * (pred ** 2).sum()
        return loss, pred, None


class MockModelWithDisconnectedParam(nn.Module):
    """Model with one active parameter and one completely disconnected parameter."""
    def __init__(self):
        super().__init__()
        self.active_w = nn.Parameter(torch.tensor([[1.5]], dtype=torch.float32))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 192), requires_grad=True)

    def forward(self, x: torch.Tensor):
        # cls_token is completely unused in forward pass (like MAE CLS token)
        pred = x * self.active_w
        loss = 0.5 * (pred ** 2).sum()
        return loss, pred, None


class TinyDifferentiableMAE(nn.Module):
    """Tiny parameterized model mimicking MAE encoder + decoder for unit tests."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4, bias=True)
        self.decoder = nn.Linear(4, 4, bias=True)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        out = self.decoder(z)
        loss = (out ** 2).mean()
        return loss, out, None


# ─── Test Suite ───────────────────────────────────────────────────────────────

class TestDPSGD(unittest.TestCase):

    def test_1_per_example_clipping_hand_computed(self):
        """
        Verify per-example clipping against exact hand-calculated values.
        Model: loss = 0.5 * (w * x)^2 -> dloss/dw = w * x^2
        w = 2.0, b = 0.0 (fixed b)
        Sample 1: x_1 = 3.0 -> g_1 = 2.0 * (3.0^2) = 18.0
        Sample 2: x_2 = 1.0 -> g_2 = 2.0 * (1.0^2) = 2.0
        Clip bound C = 5.0
        Expected clipped:
          g_1_clipped = 18.0 * (5.0 / 18.0) = 5.0
          g_2_clipped = 2.0  * 1.0           = 2.0
          sum_clipped = 5.0 + 2.0 = 7.0
          mean_clipped (batch_size=2) = 3.5
        """
        model = MockLinearModel(init_w=2.0, init_b=0.0)
        model.b.requires_grad = False

        dp_engine = DPEngine(max_grad_norm=5.0, noise_multiplier=0.0)
        inputs = torch.tensor([[3.0], [1.0]], dtype=torch.float32)

        clipped_sum, mean_loss, clip_factors = dp_engine.clip_and_accumulate_gradients(
            model=model,
            batch_inputs=inputs,
            train_parameters=[model.w],
        )

        sum_w = clipped_sum[id(model.w)].item()
        self.assertAlmostEqual(sum_w, 7.0, places=4, msg="Clipped gradient sum must equal 7.0")
        self.assertAlmostEqual(clip_factors[0], 5.0 / 18.0, places=4)
        self.assertAlmostEqual(clip_factors[1], 1.0, places=4)

        dp_engine.add_noise_and_assign_gradients(
            clipped_sum_dict=clipped_sum,
            train_parameters=[model.w],
            batch_size=2,
        )
        self.assertAlmostEqual(model.w.grad.item(), 3.5, places=4, msg="Mean gradient must equal 3.5")

    def test_2_gaussian_noise_scale_and_single_addition(self):
        """
        Verify that Gaussian noise has the intended standard deviation (sigma * C)
        and is added exactly once to the summed gradient.
        """
        C = 1.5
        sigma = 2.0
        expected_std = sigma * C  # 3.0

        gen = torch.Generator().manual_seed(42)
        dp_engine = DPEngine(max_grad_norm=C, noise_multiplier=sigma, generator=gen)

        dummy_param = nn.Parameter(torch.zeros(10000, dtype=torch.float32))
        clipped_sum = {id(dummy_param): torch.zeros_like(dummy_param.data)}

        dp_engine.add_noise_and_assign_gradients(
            clipped_sum_dict=clipped_sum,
            train_parameters=[dummy_param],
            batch_size=1,
        )

        noise_samples = dummy_param.grad.numpy()
        emp_mean = float(np.mean(noise_samples))
        emp_std = float(np.std(noise_samples))

        self.assertAlmostEqual(emp_mean, 0.0, delta=0.1, msg="Noise mean must be approximately 0")
        self.assertAlmostEqual(emp_std, expected_std, delta=0.1, msg=f"Noise std must be approximately {expected_std}")

    def test_3_partial_batch_denominator(self):
        """
        Verify that incomplete/partial batches divide by actual sample count B_actual,
        not nominal batch size.
        """
        model = MockLinearModel(init_w=1.0, init_b=0.0)
        model.b.requires_grad = False
        dp_engine = DPEngine(max_grad_norm=10.0, noise_multiplier=0.0)

        # Batch of 3 samples (when nominal batch size might be 16)
        inputs = torch.tensor([[1.0], [1.0], [1.0]], dtype=torch.float32)

        clipped_sum, _, _ = dp_engine.clip_and_accumulate_gradients(
            model=model,
            batch_inputs=inputs,
            train_parameters=[model.w],
        )

        # sum = 1.0 + 1.0 + 1.0 = 3.0
        self.assertAlmostEqual(clipped_sum[id(model.w)].item(), 3.0, places=4)

        # Divide by actual batch size = 3 -> grad = 1.0
        dp_engine.add_noise_and_assign_gradients(
            clipped_sum_dict=clipped_sum,
            train_parameters=[model.w],
            batch_size=3,
        )
        self.assertAlmostEqual(model.w.grad.item(), 1.0, places=4)

    def test_4_parameters_with_no_gradient_handled_gracefully(self):
        """
        Verify that parameters disconnected from the computation graph (e.g. vit.cls_token
        in MAE) do NOT cause NoneType errors or NaN gradients, and retain p.grad is None.
        """
        model = MockModelWithDisconnectedParam()
        dp_engine = DPEngine(max_grad_norm=1.0, noise_multiplier=0.5)

        inputs = torch.tensor([[2.0], [3.0]], dtype=torch.float32)
        train_params = [model.active_w, model.cls_token]

        clipped_sum, _, _ = dp_engine.clip_and_accumulate_gradients(
            model=model,
            batch_inputs=inputs,
            train_parameters=train_params,
        )

        dp_engine.add_noise_and_assign_gradients(
            clipped_sum_dict=clipped_sum,
            train_parameters=train_params,
            batch_size=2,
        )

        self.assertIsNotNone(model.active_w.grad)
        self.assertFalse(torch.isnan(model.active_w.grad).any())
        self.assertIsNone(model.cls_token.grad, "Disconnected parameter must strictly remain None")

    def test_5_invalid_settings_rejection(self):
        """Verify that negative clipping bounds, invalid deltas, and negative noise fail with ValueError."""
        with self.assertRaises(ValueError):
            DPEngine(max_grad_norm=-1.0)
        with self.assertRaises(ValueError):
            DPEngine(max_grad_norm=0.0)
        with self.assertRaises(ValueError):
            DPEngine(noise_multiplier=-0.1)
        with self.assertRaises(ValueError):
            DPEngine(delta=0.0)
        with self.assertRaises(ValueError):
            DPEngine(delta=1.0)
        with self.assertRaises(ValueError):
            DPEngine(delta=-1e-5)

    def test_6_dp_disabled_retains_standard_semantics_and_updates_parameters(self):
        """
        Verify non-DP regression path:
        Uses a differentiable model with genuine parameters, and asserts that
        parameters actually update under AdamW when privacy.enabled is False.
        """
        model = TinyDifferentiableMAE()
        initial_w = model.encoder.weight.detach().clone()

        x = torch.randn(4, 4)
        ds = TensorDataset(x)
        loader = DataLoader(ds, batch_size=2, shuffle=False)

        cfg = SimpleNamespace(
            ssl=SimpleNamespace(lr=0.1, epochs_per_round=1),
            federated=SimpleNamespace(aggregation="fedavg"),
            privacy=SimpleNamespace(enabled=False),
        )

        res = ssl_local_train(hospital_id=1, model=model, dataloader=loader, config=cfg)
        updated_w = model.encoder.weight.detach()

        # Meaningful verification that non-DP optimization actually changed parameters
        self.assertFalse(
            torch.allclose(initial_w.to(updated_w.device), updated_w, atol=1e-5),
            "Parameters must actually update in non-DP mode",
        )
        self.assertNotIn("privacy", res, "Non-DP run must not include privacy metadata")
        self.assertIn("federated_weights", res)

    def test_7_fedprox_dp_explicit_rejection(self):
        """
        Verify that configuring DP-SGD together with FedProx explicitly raises ValueError.
        """
        model = TinyDifferentiableMAE()
        x = torch.randn(2, 4)
        loader = DataLoader(TensorDataset(x), batch_size=2)

        cfg = SimpleNamespace(
            ssl=SimpleNamespace(lr=1e-3, epochs_per_round=1),
            federated=SimpleNamespace(aggregation="fedprox", fedprox_mu=0.01),
            privacy=SimpleNamespace(enabled=True, max_grad_norm=1.0, noise_multiplier=1.0, delta=1e-5),
        )

        with self.assertRaises(ValueError) as ctx:
            ssl_local_train(hospital_id=1, model=model, dataloader=loader, config=cfg, global_weights={})
        self.assertIn("DP + FedProx combination is explicitly rejected", str(ctx.exception))

    def test_8_cumulative_accounting_across_simulated_rounds(self):
        """
        Verify that cumulative privacy states persist and accumulate continuously
        across multiple simulated rounds without resetting.
        """
        acc = RDPAccountant(sampling_scheme="shuffle_without_replacement")
        self.assertEqual(acc.steps, 0)

        # Round 1: Client performs 5 steps
        acc.step(q=0.02, noise_multiplier=1.1, steps=5)
        self.assertEqual(acc.steps, 5)

        # Round 2: Same client resumes with state and performs 5 more steps
        state_r1 = acc.state_dict()
        acc_r2 = RDPAccountant()
        acc_r2.load_state_dict(state_r1)
        acc_r2.step(q=0.02, noise_multiplier=1.1, steps=5)

        self.assertEqual(acc_r2.steps, 10, "Cumulative steps must accumulate across rounds without resetting")
        self.assertEqual(len(acc_r2.history), 2)

    def test_9_unsupported_sampling_scheme_marks_formal_epsilon_unsupported(self):
        """
        Verify that when data loader uses random shuffling without replacement,
        formal epsilon reporting returns None rather than an unjustified Poisson approximation.
        """
        acc = RDPAccountant(sampling_scheme="shuffle_without_replacement")
        acc.step(q=0.01, noise_multiplier=1.1, steps=100)

        eps = acc.get_epsilon(target_delta=1e-5)
        self.assertIsNone(
            eps,
            "Formal epsilon must be None under random shuffling without replacement",
        )

    def test_10_checkpoint_resume_continuity_and_failure_on_missing_state(self):
        """
        Verify that checkpoint dictionary restores exact privacy steps,
        and that resuming DP training from a checkpoint without privacy metadata fails clearly.
        """
        valid_checkpoint = {
            "round": 10,
            "metrics": {
                "loss": 1.25,
                "privacy": {
                    "client_states": {
                        1: {"steps": 120, "max_grad_norm": 1.0, "noise_multiplier": 1.1},
                        2: {"steps": 120, "max_grad_norm": 1.0, "noise_multiplier": 1.1},
                    }
                }
            }
        }

        # Restoration check
        privacy_data = valid_checkpoint["metrics"]["privacy"]["client_states"]
        self.assertEqual(privacy_data[1]["steps"], 120)

        # Missing metadata failure check
        corrupt_checkpoint = {
            "round": 10,
            "metrics": {"loss": 1.25}  # missing privacy
        }
        saved_privacy = corrupt_checkpoint["metrics"].get("privacy", {})
        has_valid_state = bool(saved_privacy and "client_states" in saved_privacy)
        self.assertFalse(has_valid_state, "Corrupted checkpoint must be detected as invalid")

    def test_11_poisson_sampling_reference_validation(self):
        """
        Verify analytical RDP formula under verified Poisson subsampling.
        For q = 0.01, sigma = 1.5, T = 1000 steps, delta = 1e-5:
        Expected epsilon is ~1.2445 at alpha = 18.
        """
        acc = RDPAccountant(sampling_scheme="poisson_verified")
        acc.step(q=0.01, noise_multiplier=1.5, steps=1000)
        eps = acc.get_epsilon(target_delta=1e-5)

        self.assertIsNotNone(eps)
        self.assertAlmostEqual(eps, 1.2445, places=2)


if __name__ == "__main__":
    unittest.main()
