"""
src/client/dp_engine.py
-----------------------
Differentially Private Stochastic Gradient Descent (DP-SGD) engine for
Federated Self-Supervised Learning (Fed-SSL-TB).

Architectural Principles & Theoretical Grounding:
  1. Per-Example Clipping:
     Computes isolated per-sample gradients g_i across active trainable parameters,
     clips each to maximum L2 norm bound C: g_i * min(1.0, C / (||g_i||_2 + 1e-8)),
     and sums the clipped gradients.
  2. Noise Addition & Normalization:
     Adds calibrated Gaussian noise N(0, (sigma * C)^2 I) to the clipped sum,
     and normalizes by the actual batch size B_actual (handling partial batches correctly).
  3. Disconnected Parameter Safety:
     Parameters with grad is None (e.g. encoder.vit.cls_token in MAE patch masking)
     are strictly excluded from clipping norms and preserved as None.
  4. Explicit FedProx Rejection:
     FedProx proximal regularizers mixed into adaptive AdamW moment accumulators (v_t)
     violate standard contraction semantics and DP calibration. The combination of
     DP-SGD with FedProx is explicitly rejected.
  5. Privacy Accounting & Sampling Assumptions:
     Standard DataLoaders with shuffle=True use random shuffling without replacement.
     Because analytical subsampled Gaussian RDP assumes Poisson subsampling, formal
     epsilon reporting is strictly disabled and marked UNSUPPORTED under without-replacement
     loaders to prevent unjustified privacy claims.
  6. RNG Security Distinction:
     PyTorch PRNG (torch.randn / Philox / MT19937) is pseudorandom and suitable for
     academic simulation only. It does NOT provide cryptographically secure randomness.

Privacy Unit:
  Per-Image (Sample-Level) Differential Privacy within each local hospital dataset.
  Does NOT provide patient-level or client-level (cross-hospital) differential privacy.
"""

import math
import secrets
from typing import Dict, Any, List, Optional, Tuple, Union, Set
import numpy as np
from scipy.special import logsumexp, gammaln
import torch
import torch.nn as nn


# ─── Analytical RDP Accountant ────────────────────────────────────────────────

def _log_comb(n: int, k: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    """Compute log(n choose k) stably using log-gamma."""
    return gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)


class RDPAccountant:
    """
    Rényi Differential Privacy (RDP) Accountant for the Gaussian Mechanism.

    Supports:
      - Validated Poisson-subsampled Gaussian mechanism for verified integer orders alpha in [2, 64].
      - Conservative Gaussian mechanism (without subsampling amplification) for sensitivity C.
      - Explicit rejection/unsupported status when sampling assumptions (e.g. shuffle without replacement)
        do not match theoretical Poisson requirements.
    """

    def __init__(
        self,
        orders: Optional[List[int]] = None,
        sampling_scheme: str = "shuffle_without_replacement",
    ):
        # Strictly use verified integer orders alpha >= 2 (Mironov 2017, Wang et al. 2019)
        self.orders = orders or list(range(2, 65))
        for a in self.orders:
            if not isinstance(a, int) or a < 2:
                raise ValueError(f"Accountant orders must be integers >= 2, got {a}")

        self.sampling_scheme = sampling_scheme.lower()
        self.steps = 0
        self.history: List[Tuple[float, float, int]] = []  # (q, sigma, steps)

    def step(self, q: float, noise_multiplier: float, steps: int = 1) -> None:
        """Record executed optimizer steps."""
        if q <= 0.0 or q > 1.0:
            raise ValueError(f"Subsampling ratio q must be in (0, 1], got {q}")
        if noise_multiplier < 0.0:
            raise ValueError(f"Noise multiplier must be non-negative, got {noise_multiplier}")
        if steps <= 0:
            return

        self.steps += steps
        self.history.append((float(q), float(noise_multiplier), int(steps)))

    @staticmethod
    def compute_rdp_poisson_subsampled(q: float, sigma: float, alpha: int) -> float:
        """
        Compute RDP of integer order alpha for Poisson-subsampled Gaussian mechanism
        using numerically stable log-space binomial expansion (Wang et al., ICML 2019).
        """
        if sigma <= 0.0:
            return float("inf")
        if q == 1.0:
            return alpha / (2.0 * (sigma ** 2))
        if q == 0.0:
            return 0.0

        k = np.arange(alpha + 1)
        log_coeffs = (
            _log_comb(alpha, k)
            + k * math.log(q)
            + (alpha - k) * math.log(1.0 - q)
        )
        log_exps = k * (k - 1.0) / (2.0 * (sigma ** 2))
        log_terms = log_coeffs + log_exps
        total_log = logsumexp(log_terms)
        return float(total_log / (alpha - 1.0))

    def get_epsilon(self, target_delta: float) -> Optional[float]:
        """
        Compute epsilon for the given target_delta.

        NOTE: If sampling_scheme is 'shuffle_without_replacement' (the actual DataLoader mechanism),
        formal epsilon is marked UNSUPPORTED to avoid unsubstantiated privacy claims,
        returning None.

        If sampling_scheme is 'poisson_verified', computes exact analytical RDP bound.
        """
        if target_delta <= 0.0 or target_delta >= 1.0:
            raise ValueError(f"target_delta must be in (0, 1), got {target_delta}")

        if self.steps == 0 or not self.history:
            return 0.0

        # Check for zero noise
        if any(sigma == 0.0 for _, sigma, _ in self.history):
            return float("inf")

        # Refuse to return an unverified epsilon if sampling scheme is not Poisson
        if self.sampling_scheme != "poisson_verified":
            return None

        # Accumulate RDP across steps for each integer order alpha
        total_rdp = {a: 0.0 for a in self.orders}
        for q, sigma, steps in self.history:
            for a in self.orders:
                rdp_step = self.compute_rdp_poisson_subsampled(q, sigma, a)
                total_rdp[a] += steps * rdp_step

        # Convert RDP to standard DP: eps(alpha) = RDP(alpha) + log(1/delta) / (alpha - 1)
        epsilons = [
            total_rdp[a] + math.log(1.0 / target_delta) / (a - 1.0)
            for a in self.orders
        ]
        return float(min(epsilons))

    def state_dict(self) -> Dict[str, Any]:
        """Serialize accountant state for cumulative checkpointing."""
        return {
            "steps": self.steps,
            "history": list(self.history),
            "orders": list(self.orders),
            "sampling_scheme": self.sampling_scheme,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore accountant state from checkpoint."""
        self.steps = int(state.get("steps", 0))
        self.history = [
            (float(q), float(sigma), int(s))
            for q, sigma, s in state.get("history", [])
        ]
        if "orders" in state:
            self.orders = [int(a) for a in state["orders"]]
        if "sampling_scheme" in state:
            self.sampling_scheme = str(state["sampling_scheme"])


# ─── DP-SGD Engine ────────────────────────────────────────────────────────────

class DPEngine:
    """
    Per-example gradient clipping, calibrated noise addition, and batch normalization.

    RNG Security Notice:
      The default PyTorch Generator uses Philox/MT19937 PRNG algorithms.
      While standard for empirical machine learning research, PRNG is NOT a
      cryptographically secure random number generator (CSPRNG).
      Production deployments require a certified hardware TRNG or CSPRNG.

    Args:
        max_grad_norm: L2 gradient norm clipping threshold C > 0
        noise_multiplier: Noise standard deviation multiplier sigma >= 0
        delta: Target delta for privacy bookkeeping (default 1e-5)
        generator: Optional isolated torch.Generator for thread-safe RNG
        sampling_scheme: Underlying data-sampling scheme ('shuffle_without_replacement' or 'poisson_verified')
    """

    def __init__(
        self,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        delta: float = 1e-5,
        generator: Optional[torch.Generator] = None,
        sampling_scheme: str = "shuffle_without_replacement",
        device: Optional[Union[str, torch.device]] = None,
    ):
        if max_grad_norm <= 0.0:
            raise ValueError(f"max_grad_norm must be positive, got {max_grad_norm}")
        if noise_multiplier < 0.0:
            raise ValueError(f"noise_multiplier must be non-negative, got {noise_multiplier}")
        if delta <= 0.0 or delta >= 1.0:
            raise ValueError(f"delta must be in (0, 1), got {delta}")

        self.max_grad_norm = float(max_grad_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.delta = float(delta)
        self.device = torch.device(device) if device is not None else None
        self.generator = generator or self._create_os_seeded_generator(self.device)
        self.accountant = RDPAccountant(sampling_scheme=sampling_scheme)

    @staticmethod
    def _create_os_seeded_generator(device: Optional[Union[str, torch.device]] = None) -> torch.Generator:
        """Create a torch.Generator seeded with 64 bits from the OS cryptographic entropy pool."""
        gen = torch.Generator(device=device) if device is not None else torch.Generator()
        # Seed from OS entropy (secrets module) to avoid default fixed seed
        seed = secrets.randbits(64)
        gen.manual_seed(seed)
        return gen

    def clip_and_accumulate_gradients(
        self,
        model: nn.Module,
        batch_inputs: torch.Tensor,
        train_parameters: List[nn.Parameter],
    ) -> Tuple[Dict[int, torch.Tensor], float, List[float]]:
        """
        Compute per-example gradients, clip each to L2 norm bound C, and sum them.

        Args:
            model: Active training model
            batch_inputs: Tensor of shape (B, C, H, W)
            train_parameters: List of candidate model parameters

        Returns:
            clipped_sum_dict: Dict mapping id(param) to summed clipped gradient tensor
            total_batch_loss: Float mean loss across the batch
            sample_clip_factors: List of clipping scale factors applied to each sample
        """
        batch_size = batch_inputs.shape[0]
        if batch_size == 0:
            raise ValueError("Cannot process empty batch in DP engine.")

        active_params = [p for p in train_parameters if p.requires_grad]
        clipped_sum: Dict[int, torch.Tensor] = {}

        total_loss = 0.0
        sample_clip_factors: List[float] = []

        # Process each sample independently to isolate per-example gradient g_i
        for i in range(batch_size):
            model.zero_grad(set_to_none=True)
            single_input = batch_inputs[i : i + 1]

            loss, _, _ = model(single_input)
            loss.backward()
            total_loss += float(loss.item())

            # Isolate parameters that actually received gradients for this sample
            sample_grads = []
            param_refs = []
            for p in active_params:
                if p.grad is not None:
                    sample_grads.append(p.grad)
                    param_refs.append(p)

            # Compute L2 norm across all active parameters: ||g_i||_2
            if sample_grads:
                norm_sq = sum(torch.sum(g ** 2) for g in sample_grads)
                sample_norm = float(torch.sqrt(norm_sq).item())
                clip_factor = min(1.0, self.max_grad_norm / (sample_norm + 1e-8))
            else:
                sample_norm = 0.0
                clip_factor = 1.0

            sample_clip_factors.append(clip_factor)

            # Accumulate clipped gradients: g_sum += clip_factor * g_i
            for p, g in zip(param_refs, sample_grads):
                if id(p) not in clipped_sum:
                    clipped_sum[id(p)] = torch.zeros_like(p.data)
                clipped_sum[id(p)].add_(g, alpha=clip_factor)

        model.zero_grad(set_to_none=True)
        mean_loss = total_loss / batch_size
        return clipped_sum, mean_loss, sample_clip_factors

    def add_noise_and_assign_gradients(
        self,
        clipped_sum_dict: Dict[int, torch.Tensor],
        train_parameters: List[nn.Parameter],
        batch_size: int,
    ) -> None:
        """
        Add calibrated Gaussian noise to summed clipped gradients, divide by batch size,
        and assign to param.grad. Parameters not in clipped_sum_dict are set to None.

        Mathematics:
            noise_std = noise_multiplier * max_grad_norm
            g_privatized = (sum(g_clipped) + N(0, noise_std^2 I)) / batch_size

        Args:
            clipped_sum_dict: Summed clipped gradients from clip_and_accumulate_gradients
            train_parameters: Target model parameters
            batch_size: Actual number of samples in the batch (handles partial batches)
        """
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")

        noise_std = self.noise_multiplier * self.max_grad_norm

        for p in train_parameters:
            if not p.requires_grad or id(p) not in clipped_sum_dict:
                p.grad = None
                continue

            summed_g = clipped_sum_dict[id(p)]

            # Generate calibrated Gaussian noise
            if noise_std > 0.0:
                gen = self.generator
                if gen is not None and hasattr(gen, "device"):
                    if gen.device.type != summed_g.device.type or (
                        gen.device.type == "cuda" and gen.device.index != summed_g.device.index
                    ):
                        gen = self._create_os_seeded_generator(device=summed_g.device)
                        self.generator = gen

                noise = torch.randn(
                    summed_g.shape,
                    device=summed_g.device,
                    dtype=summed_g.dtype,
                    generator=gen,
                ) * noise_std
                noisy_sum = summed_g + noise
            else:
                noisy_sum = summed_g

            # Average by actual batch size
            p.grad = noisy_sum / float(batch_size)
