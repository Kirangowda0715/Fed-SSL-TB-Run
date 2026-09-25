"""
src/client/ssl_train.py
-----------------------
Local SSL training loop for each hospital using a Masked Autoencoder (MAE).

Key design decisions:
  - Only encoder weights are returned to the server (decoder stays local)
  - FedProx proximal term added to client loss when global_weights provided
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


def ssl_local_train(
    hospital_id: int,
    model: nn.Module,
    dataloader: DataLoader,
    config,
    global_weights: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Train the MAE locally on hospital data for one federated round.

    Args:
        hospital_id    : Hospital identifier (1..5) — used for logging
        model          : MaskedAutoencoder instance (encoder + decoder)
        dataloader     : DataLoader for this hospital's NIH shard
        config         : Loaded config (SimpleNamespace)
        global_weights : Encoder state_dict from the global server model.
                         If provided, FedProx proximal regularization is applied.
                         If None, standard MAE loss only (FedAvg mode).
        device         : torch.device; defaults to CUDA if available

    Returns:
        Dict with:
            'encoder_weights' : Encoder state_dict (to send to server)
            'num_samples'     : Number of training samples seen
            'epoch_losses'    : List of mean loss per epoch
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

    # ── FedProx setup ────────────────────────────────────────────────────
    is_fedprox = (global_weights is not None) and (config.federated.aggregation == "fedprox")
    mu = getattr(config.federated, "fedprox_mu", 0.01)


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

            optimizer.zero_grad()

            # MAE forward pass → reconstruction loss
            loss, _, _ = model(imgs)

            # FedProx proximal term: μ/2 * ||w_local - w_global||²
            # Computed over live encoder parameters to preserve the autograd computational graph
            if is_fedprox and global_weights is not None:
                enc_weights = global_weights.get("encoder", global_weights) if isinstance(global_weights, dict) else global_weights
                proximal_term = 0.0
                for name, param in model.encoder.named_parameters():
                    if name in enc_weights:
                        g_param = enc_weights[name].to(device)
                        proximal_term = proximal_term + torch.sum((param - g_param) ** 2)
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

    # Return only encoder weights — decoder stays local
    encoder_weights = model.get_encoder_weights()

    print(
        f"  [Hospital {hospital_id}] SSL training done | "
        f"Samples: {num_samples} | "
        f"Final loss: {epoch_losses[-1]:.4f}"
    )

    return {
        "encoder_weights": encoder_weights,
        "num_samples": num_samples,
        "epoch_losses": epoch_losses,
        "ssl_loss": epoch_losses[-1] if epoch_losses else float("nan"),
    }

