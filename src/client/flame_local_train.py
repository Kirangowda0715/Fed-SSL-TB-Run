"""Joint FLAME local update: MAE and prototypical learning in one optimizer step."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


def _images(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def joint_loss(mae_loss: torch.Tensor, proto_loss: torch.Tensor, alpha: float = 0.7) -> torch.Tensor:
    """Combine both objectives; gradients remain connected to shared modules."""
    return alpha * mae_loss + (1.0 - alpha) * proto_loss


def flame_local_train(
    hospital_id: int,
    model: nn.Module,
    unlabeled_loader: DataLoader,
    support_loader: DataLoader,
    config,
    global_weights: Optional[Dict[str, Dict[str, Any]]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run joint MAE + Proto training and return the complete local model."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    flame_config = getattr(config, "flame", None)
    alpha = float(getattr(flame_config, "alpha", 0.7))
    optimizer = AdamW(model.parameters(), lr=float(config.ssl.lr), weight_decay=0.05, betas=(0.9, 0.95))
    is_fedprox = config.federated.aggregation.lower() == "fedprox" and global_weights is not None
    mu = float(getattr(config.federated, "fedprox_mu", 0.01))
    global_parameters = {
        f"{component}.{name}": value.to(device)
        for component, state in (global_weights or {}).items()
        for name, value in state.items()
    }
    support_images, support_labels = next(iter(support_loader))
    support_images = support_images.to(device)
    support_labels = torch.as_tensor(support_labels, device=device).long()
    epoch_records = []

    for _ in range(int(config.ssl.epochs_per_round)):
        mae_total = proto_total = total_total = 0.0
        batches = 0
        for batch in unlabeled_loader:
            images = _images(batch).to(device)
            optimizer.zero_grad()
            mae_loss, _, _ = model(images)
            support_embeddings = model.encoder(support_images)
            prototypes = model.proto_head.compute_prototypes(
                support_embeddings, support_labels
            )
            # The local support episode is the labeled query set in this compact
            # implementation; this keeps every NIH iteration jointly supervised.
            query_embeddings = support_embeddings
            proto_loss, _ = model.proto_head.prototypical_loss(
                query_embeddings, support_labels, prototypes
            )
            total_loss = joint_loss(mae_loss, proto_loss, alpha)
            if is_fedprox:
                proximal = torch.zeros((), device=device)
                for name, parameter in model.named_parameters():
                    if name in global_parameters:
                        proximal = proximal + (parameter - global_parameters[name]).pow(2).sum()
                total_loss = total_loss + (mu / 2.0) * proximal
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            mae_total += float(mae_loss.detach())
            proto_total += float(proto_loss.detach())
            total_total += float(total_loss.detach())
            batches += 1

        divisor = max(batches, 1)
        epoch_records.append({"mae_loss": mae_total / divisor, "proto_loss": proto_total / divisor, "total_loss": total_total / divisor})

    final = epoch_records[-1]
    print(f"[Hospital {hospital_id}] MAE Loss: {final['mae_loss']:.4f} | Proto Loss: {final['proto_loss']:.4f} | Total Loss: {final['total_loss']:.4f}")
    return {
        "model_weights": model.get_federated_weights(),
        "num_samples": len(unlabeled_loader.dataset),
        "epoch_losses": epoch_records,
        **final,
    }
