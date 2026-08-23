"""
src/client/local_train.py
-------------------------
Few-shot fine-tuning using Prototypical Networks on the Shenzhen TB dataset.

Process:
  1. Freeze encoder (or use very low LR)
  2. Sample k-shot support set per class (k = config.finetuning.few_shot_k)
  3. Compute class prototypes from support embeddings
  4. Fine-tune prototypical head on remaining query samples
  5. Return fine-tuned model + validation metrics
"""

import copy
import random
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.models.proto_head import PrototypicalHead
from src.utils.metrics import evaluate


def finetune_local(
    hospital_id: int,
    encoder: nn.Module,
    shenzhen_loader: DataLoader,
    config,
    device: Optional[torch.device] = None,
) -> Tuple[PrototypicalHead, Dict[str, Any]]:
    """
    Few-shot fine-tuning of the prototypical head on Shenzhen TB data.

    Args:
        hospital_id     : Hospital ID (for logging)
        encoder         : Pre-trained encoder (ResNet50 / ViT-Small)
        shenzhen_loader : DataLoader for ShenzhenDataset
        config          : Loaded config SimpleNamespace
        device          : torch.device

    Returns:
        proto_head      : Fine-tuned PrototypicalHead
        metrics         : Dict with AUC, accuracy, sensitivity, specificity, F1
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = encoder.to(device)
    encoder.eval()  # Freeze encoder

    k = config.finetuning.few_shot_k
    num_classes = 2  # 0=Normal, 1=TB
    embed_dim = _get_embed_dim(encoder)

    # ── 1. Collect all embeddings + labels from Shenzhen ─────────────────
    all_embeddings, all_labels = _extract_embeddings(encoder, shenzhen_loader, device)

    if len(all_embeddings) == 0:
        print(f"  [Hospital {hospital_id}] No Shenzhen data — skipping fine-tuning.")
        proto_head = PrototypicalHead(embed_dim=embed_dim, num_classes=num_classes)
        return proto_head, {}

    # ── 2. Sample k-shot support set (k per class) ────────────────────────
    support_idx, query_idx = _sample_kshot(all_labels, k=k, num_classes=num_classes)

    support_emb = all_embeddings[support_idx]
    support_lbl = all_labels[support_idx]
    query_emb   = all_embeddings[query_idx]
    query_lbl   = all_labels[query_idx]

    # ── 3. Initialize prototypical head ──────────────────────────────────
    proto_head = PrototypicalHead(embed_dim=embed_dim, num_classes=num_classes).to(device)

    # Compute initial prototypes from support set
    with torch.no_grad():
        proto_head.compute_prototypes(support_emb, support_lbl)

    # ── 4. Fine-tune on query set ─────────────────────────────────────────
    optimizer = AdamW(proto_head.parameters(), lr=config.finetuning.lr, weight_decay=1e-4)
    num_epochs = config.finetuning.epochs

    proto_head.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        prototypes = proto_head.get_learnable_prototypes(support_emb, support_lbl)
        loss, probs = proto_head.prototypical_loss(query_emb, query_lbl, prototypes)
        if loss.requires_grad:
            loss.backward()
            optimizer.step()

        if (epoch + 1) % max(1, num_epochs // 3) == 0:
            acc = (probs.argmax(dim=-1) == query_lbl).float().mean().item()
            print(f"    Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

    # ── 5. Evaluate on query set ──────────────────────────────────────────
    proto_head.eval()
    with torch.no_grad():
        prototypes = proto_head.compute_prototypes(support_emb, support_lbl)
        _, probs = proto_head.predict(query_emb, prototypes)
        tb_probs = probs[:, 1].cpu().numpy()     # prob of TB class
        y_true   = query_lbl.cpu().numpy()

    metrics = evaluate(y_true, tb_probs)
    print(
        f"  [Hospital {hospital_id}] Fine-tuning done | "
        f"AUC={metrics['auc']:.4f} | "
        f"Sensitivity={metrics['sensitivity']:.4f} | "
        f"Specificity={metrics['specificity']:.4f}"
    )

    return proto_head, metrics


# ─── Evaluation on Montgomery (held-out test) ────────────────────────────────

def evaluate_on_montgomery(
    encoder: nn.Module,
    proto_head: PrototypicalHead,
    montgomery_loader: DataLoader,
    support_loader: DataLoader,
    config,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Run evaluation of the encoder + prototypical head on Montgomery test set.

    Args:
        encoder          : Trained encoder
        proto_head       : Fine-tuned PrototypicalHead
        montgomery_loader: DataLoader for MontgomeryDataset
        support_loader   : DataLoader for Shenzhen (to re-compute prototypes)
        config           : Config
        device           : torch.device

    Returns:
        metrics dict
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder.to(device).eval()
    proto_head.to(device).eval()

    # Re-compute prototypes from Shenzhen support set
    support_emb, support_lbl = _extract_embeddings(encoder, support_loader, device)
    k = config.finetuning.few_shot_k
    if len(support_emb) > 0:
        support_idx, _ = _sample_kshot(support_lbl, k=k, num_classes=2)
        support_emb = support_emb[support_idx]
        support_lbl = support_lbl[support_idx]
        with torch.no_grad():
            prototypes = proto_head.compute_prototypes(support_emb, support_lbl)
    else:
        prototypes = None

    # Embed Montgomery test set
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(montgomery_loader, desc="  Evaluating on Montgomery", leave=False):
            imgs = imgs.to(device)
            emb = _get_embedding(encoder, imgs)
            if prototypes is not None:
                _, probs = proto_head.predict(emb, prototypes)
            else:
                # Fallback: linear head
                logits = proto_head.linear_forward(emb)
                probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs[:, 1].cpu())
            all_labels.append(labels)

    y_pred = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy()

    return evaluate(y_true, y_pred)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _extract_embeddings(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run encoder over entire dataloader, return (embeddings, labels) tensors."""
    embeddings_list, labels_list = [], []
    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                imgs, labels = batch
            else:
                imgs = batch
                labels = torch.zeros(imgs.shape[0], dtype=torch.long)
            imgs = imgs.to(device)
            emb = _get_embedding(encoder, imgs)
            embeddings_list.append(emb.cpu())
            labels_list.append(labels if isinstance(labels, torch.Tensor) else torch.tensor(labels))

    if len(embeddings_list) == 0:
        return torch.zeros(0), torch.zeros(0, dtype=torch.long)

    return torch.cat(embeddings_list), torch.cat(labels_list).long()


def _get_embedding(encoder: nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    """Get embedding from encoder, handling both ResNet and ViT paths."""
    return encoder(imgs)


def _sample_kshot(
    labels: torch.Tensor,
    k: int,
    num_classes: int,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample k support examples per class, rest are query examples.

    Returns:
        support_indices : (k * num_classes,) index tensor
        query_indices   : remaining indices
    """
    torch.manual_seed(seed)
    support_idx = []
    all_idx = torch.arange(len(labels))

    for c in range(num_classes):
        class_idx = all_idx[labels == c].tolist()
        if len(class_idx) == 0:
            continue
        k_actual = min(k, len(class_idx))
        chosen = random.sample(class_idx, k_actual)
        support_idx.extend(chosen)

    support_idx = torch.tensor(support_idx)
    support_mask = torch.zeros(len(labels), dtype=torch.bool)
    support_mask[support_idx] = True
    query_idx = all_idx[~support_mask]

    return support_idx, query_idx


def _get_embed_dim(encoder: nn.Module) -> int:
    """Infer embed_dim from encoder module."""
    if hasattr(encoder, "embed_dim"):
        return encoder.embed_dim
    # Heuristic: run a dummy forward pass
    encoder.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        out = encoder(dummy)
    return out.shape[-1]
