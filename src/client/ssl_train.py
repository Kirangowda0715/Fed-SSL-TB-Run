"""
src/client/ssl_train.py
-----------------------
Local SSL training loop for each hospital using a Masked Autoencoder (MAE).

Key design decisions:
  - Encoder + decoder weights are returned to the server for federated MAE training
  - The prototype head is not trained or shared during Stage 1
  - FedProx proximal term added to encoder + decoder client loss when global_weights provided
  - AdamW optimizer with cosine annealing LR schedule
"""

import copy
import math
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm


def _fedprox_penalty(model: nn.Module, global_weights: Dict[str, Any], device: torch.device) -> torch.Tensor:
    """Return FedProx squared-distance penalty for federated encoder + decoder."""
    if not isinstance(global_weights, dict) or "encoder" not in global_weights or "decoder" not in global_weights:
        raise ValueError("Stage 1 FedProx requires global encoder and decoder weights.")

    penalty = torch.zeros((), device=device)
    for component_name, component in (
        ("encoder", model.encoder),
        ("decoder", model.decoder),
    ):
        reference_weights = global_weights[component_name]
        for name, param in component.named_parameters():
            if name not in reference_weights:
                raise ValueError(
                    f"Missing {component_name} parameter '{name}' in global FedProx weights."
                )
            reference = reference_weights[name].to(device=device, dtype=param.dtype)
            penalty = penalty + torch.sum((param - reference) ** 2)
    return penalty


def ssl_local_train(
    hospital_id: int,
    model: nn.Module,
    dataloader: DataLoader,
    config,
    global_weights: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    privacy_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Train the MAE locally on hospital data for one federated round.

    Args:
        hospital_id    : Hospital identifier (1..5) — used for logging
        model          : MaskedAutoencoder instance (encoder + decoder)
        dataloader     : DataLoader for this hospital's NIH shard
        config         : Loaded config (SimpleNamespace)
        global_weights : Nested encoder + decoder state dict from the global server model.
                         If provided, FedProx proximal regularization is applied.
                         If None, standard MAE loss only (FedAvg mode).
        device         : torch.device; defaults to CUDA if available
        privacy_state  : Optional prior cumulative privacy state dictionary for continuous accounting

    Returns:
        Dict with federated_weights, num_samples, epoch_losses, and optional privacy metadata.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.train()

    # ── Optimizer & scheduler ───────────────────────────────────────────
    train_parameters = list(model.encoder.parameters()) + list(model.decoder.parameters())
    optimizer = AdamW(
        train_parameters,
        lr=config.ssl.lr,
        weight_decay=0.05,
        betas=(0.9, 0.95),
    )
    num_epochs = config.ssl.epochs_per_round
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # ── Privacy setup ────────────────────────────────────────────────────
    privacy_cfg = getattr(config, "privacy", None)
    use_dp = bool(getattr(privacy_cfg, "enabled", False))

    # ── FedProx setup ────────────────────────────────────────────────────
    is_fedprox = (global_weights is not None) and (config.federated.aggregation == "fedprox")
    mu = getattr(config.federated, "fedprox_mu", 0.01)

    if use_dp and (is_fedprox or str(getattr(config.federated, "aggregation", "")).lower() == "fedprox"):
        raise ValueError(
            "DP + FedProx combination is explicitly rejected: adding deterministic proximal "
            "regularization into AdamW adaptive moment accumulators (v_t) distorts the proximal "
            "rate across parameters and violates DP calibration. Use 'federated.aggregation: fedavg' for DP pretraining."
        )

    dp_engine = None
    if use_dp:
        from src.client.dp_engine import DPEngine
        max_grad_norm = float(getattr(privacy_cfg, "max_grad_norm", 1.0))
        noise_multiplier = float(getattr(privacy_cfg, "noise_multiplier", 1.0))
        delta = float(getattr(privacy_cfg, "delta", 1e-5))
        sampling_scheme = str(getattr(privacy_cfg, "sampling_scheme", "shuffle_without_replacement"))
        dp_engine = DPEngine(
            max_grad_norm=max_grad_norm,
            noise_multiplier=noise_multiplier,
            delta=delta,
            sampling_scheme=sampling_scheme,
            device=device,
        )
        accountant_state = (
            privacy_state.get("accountant_state")
            if privacy_state is not None
            else None
        )
        if accountant_state is not None:
            dp_engine.accountant.load_state_dict(accountant_state)

    # ── Training loop ────────────────────────────────────────────────────
    epoch_losses: List[float] = []
    num_samples = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            dataloader,
            desc=f"  Hospital {hospital_id} | Epoch {epoch+1}/{num_epochs}",
            leave=False,
        )

        for batch in pbar:
            # NIHDataset returns a tensor or (view1, view2); generic loaders may return a tuple
            if isinstance(batch, (list, tuple)):
                imgs = batch[0].to(device)
            else:
                imgs = batch.to(device)

            if use_dp and dp_engine is not None:
                # ── Differentially Private Per-Example Path ───────────────────
                clipped_sum, batch_loss, _ = dp_engine.clip_and_accumulate_gradients(
                    model=model,
                    batch_inputs=imgs,
                    train_parameters=train_parameters,
                )
                dp_engine.add_noise_and_assign_gradients(
                    clipped_sum_dict=clipped_sum,
                    train_parameters=train_parameters,
                    batch_size=imgs.shape[0],
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # Record step in accountant
                ds_len = len(dataloader.dataset) if hasattr(dataloader, "dataset") else imgs.shape[0] * len(dataloader)
                q = float(imgs.shape[0]) / max(float(ds_len), 1.0)
                dp_engine.accountant.step(q=q, noise_multiplier=dp_engine.noise_multiplier, steps=1)
            else:
                # ── Canonical Non-Private Path (Unchanged) ────────────────────
                optimizer.zero_grad()

                # MAE forward pass → reconstruction loss
                loss, _, _ = model(imgs)

                # FedProx proximal term: μ/2 * ||w_local - w_global||²
                if is_fedprox and global_weights is not None:
                    proximal_term = _fedprox_penalty(model, global_weights, device)
                    loss = loss + (mu / 2.0) * proximal_term

                loss.backward()

                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(train_parameters, max_norm=1.0)

                optimizer.step()
                batch_loss = loss.item()

            epoch_loss += batch_loss
            num_batches += 1
            num_samples += imgs.shape[0]
            pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

        scheduler.step()
        mean_epoch_loss = epoch_loss / max(num_batches, 1)
        epoch_losses.append(mean_epoch_loss)

    # Stage 1 federates the complete MAE reconstruction path: encoder + decoder.
    # The prototype head is intentionally excluded from Stage 1 SSL.
    federated_weights = {
        "encoder": {k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()},
        "decoder": {k: v.detach().cpu().clone() for k, v in model.decoder.state_dict().items()},
    }

    print(
        f"  [Hospital {hospital_id}] SSL training done | "
        f"Samples: {num_samples} | "
        f"Final loss: {epoch_losses[-1]:.4f}"
    )

    result = {
        "federated_weights": federated_weights,
        # Keep this alias temporarily for callers/tests that inspect encoder weights.
        "encoder_weights": federated_weights["encoder"],
        "num_samples": num_samples,
        "epoch_losses": epoch_losses,
        "ssl_loss": epoch_losses[-1] if epoch_losses else float("nan"),
    }

    if use_dp and dp_engine is not None:
        eps = dp_engine.accountant.get_epsilon(dp_engine.delta)
        result["privacy"] = {
            "enabled": True,
            "max_grad_norm": dp_engine.max_grad_norm,
            "noise_multiplier": dp_engine.noise_multiplier,
            "delta": dp_engine.delta,
            "steps": dp_engine.accountant.steps,
            "epsilon": eps,
            "formal_epsilon_status": (
                "supported" if eps is not None else "unsupported_under_shuffle_without_replacement"
            ),
            "privacy_unit": "image",
            "accountant_state": dp_engine.accountant.state_dict(),
        }

    return result

