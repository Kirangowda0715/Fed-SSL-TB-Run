"""
src/client/local_train.py
-------------------------
Few-shot fine-tuning using Prototypical Networks on the Shenzhen TB dataset.

Process:
  1. Split Shenzhen into k-shot support set + query set
  2. Each epoch: pass images through LIVE encoder → compute prototypes → prototypical loss
  3. Backprop through encoder to fine-tune representations
  4. Evaluate on query set with final prototypes
  5. Return fine-tuned model + validation metrics

Key fix (v2): The encoder must be unfrozen and embeddings must be computed
through the live computational graph each epoch. Pre-extracting embeddings
with torch.no_grad() and then trying to "train" on them produces zero gradients
because the computation graph is severed.
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
    Few-shot fine-tuning of the encoder using prototypical loss on Shenzhen TB data.

    The encoder is fine-tuned (not frozen) so that gradients flow from the
    prototypical loss back through the encoder, enabling actual learning.

    Args:
        hospital_id     : Hospital ID (for logging)
        encoder         : Pre-trained encoder (ViT-Tiny)
        shenzhen_loader : DataLoader for ShenzhenDataset
        config          : Loaded config SimpleNamespace
        device          : torch.device

    Returns:
        proto_head      : PrototypicalHead (with computed prototypes)
        metrics         : Dict with AUC, accuracy, sensitivity, specificity, F1
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = encoder.to(device)
    embed_dim = _get_embed_dim(encoder, device)
    num_classes = 2  # 0=Normal, 1=TB
    k = config.finetuning.few_shot_k

    # ── 1. Collect all images + labels from the DataLoader ────────────────
    all_images, all_labels = _collect_images_and_labels(shenzhen_loader)

    if len(all_images) == 0:
        print(f"  [Hospital {hospital_id}] No Shenzhen data — skipping fine-tuning.")
        proto_head = PrototypicalHead(embed_dim=embed_dim, num_classes=num_classes)
        return proto_head, {}

    # ── 2. Sample k-shot support set (k per class) ────────────────────────
    support_idx, query_idx = _sample_kshot(all_labels, k=k, num_classes=num_classes)

    support_images = all_images[support_idx].to(device)
    support_labels = all_labels[support_idx].to(device)
    query_images   = all_images[query_idx].to(device)
    query_labels   = all_labels[query_idx].to(device)

    # ── 3. Initialize prototypical head ──────────────────────────────────
    proto_head = PrototypicalHead(embed_dim=embed_dim, num_classes=num_classes).to(device)

    # ── 4. Fine-tune encoder on prototypical loss ─────────────────────────
    # Unfreeze encoder for fine-tuning with a small learning rate
    encoder.train()
    for param in encoder.parameters():
        param.requires_grad = True

    optimizer = AdamW(encoder.parameters(), lr=config.finetuning.lr, weight_decay=1e-4)
    num_epochs = config.finetuning.epochs

    # Mini-batch the query set for memory efficiency
    query_batch_size = min(64, len(query_images))

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Compute support embeddings through live encoder (with grad)
        support_emb = encoder(support_images)
        # Compute prototypes from live support embeddings (retains grad)
        prototypes = proto_head.get_learnable_prototypes(support_emb, support_labels)

        # Compute query embeddings through live encoder (with grad)
        # For large query sets, process in mini-batches and accumulate loss
        total_loss = torch.tensor(0.0, device=device)
        total_correct = 0
        total_count = 0

        for start in range(0, len(query_images), query_batch_size):
            end = min(start + query_batch_size, len(query_images))
            q_imgs = query_images[start:end]
            q_lbls = query_labels[start:end]

            q_emb = encoder(q_imgs)
            batch_loss, batch_probs = proto_head.prototypical_loss(q_emb, q_lbls, prototypes)
            total_loss = total_loss + batch_loss * len(q_lbls)
            total_correct += (batch_probs.argmax(dim=-1) == q_lbls).sum().item()
            total_count += len(q_lbls)

        mean_loss = total_loss / max(total_count, 1)
        mean_loss.backward()

        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % max(1, num_epochs // 3) == 0:
            acc = total_correct / max(total_count, 1)
            print(f"    Epoch {epoch+1}/{num_epochs} | Loss: {mean_loss.item():.4f} | Acc: {acc:.4f}")

    # ── 5. Evaluate on query set ──────────────────────────────────────────
    encoder.eval()
    proto_head.eval()
    with torch.no_grad():
        support_emb = encoder(support_images)
        prototypes = proto_head.compute_prototypes(support_emb, support_labels)

        # Evaluate on query set
        all_probs_list = []
        for start in range(0, len(query_images), query_batch_size):
            end = min(start + query_batch_size, len(query_images))
            q_emb = encoder(query_images[start:end])
            _, probs = proto_head.predict(q_emb, prototypes)
            all_probs_list.append(probs[:, 1].cpu())

        tb_probs = torch.cat(all_probs_list).numpy()
        y_true   = query_labels.cpu().numpy()

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
        support_emb_k = support_emb[support_idx].to(device)
        support_lbl_k = support_lbl[support_idx].to(device)
        with torch.no_grad():
            prototypes = proto_head.compute_prototypes(support_emb_k, support_lbl_k)
    else:
        prototypes = None

    # Embed Montgomery test set
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(montgomery_loader, desc="  Evaluating on Montgomery", leave=False):
            imgs = imgs.to(device)
            emb = encoder(imgs)  # Stay on device
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

def _collect_images_and_labels(
    loader: DataLoader,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect all images and labels from a DataLoader into single tensors."""
    images_list, labels_list = [], []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            imgs, labels = batch
        else:
            imgs = batch
            labels = torch.zeros(imgs.shape[0], dtype=torch.long)
        images_list.append(imgs)
        labels_list.append(labels if isinstance(labels, torch.Tensor) else torch.tensor(labels))

    if len(images_list) == 0:
        return torch.zeros(0), torch.zeros(0, dtype=torch.long)

    return torch.cat(images_list), torch.cat(labels_list).long()


def _extract_embeddings(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run encoder over entire dataloader, return (embeddings, labels) tensors on device."""
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
            emb = encoder(imgs)
            # Keep embeddings on the computation device (not CPU)
            embeddings_list.append(emb)
            labels_list.append(labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=device))

    if len(embeddings_list) == 0:
        return torch.zeros(0, device=device), torch.zeros(0, dtype=torch.long, device=device)

    return torch.cat(embeddings_list), torch.cat(labels_list).long()


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
    # Move labels to CPU for indexing
    labels_cpu = labels.cpu()
    torch.manual_seed(seed)
    support_idx = []
    all_idx = torch.arange(len(labels_cpu))

    for c in range(num_classes):
        class_idx = all_idx[labels_cpu == c].tolist()
        if len(class_idx) == 0:
            continue
        k_actual = min(k, len(class_idx))
        chosen = random.sample(class_idx, k_actual)
        support_idx.extend(chosen)

    support_idx = torch.tensor(support_idx)
    support_mask = torch.zeros(len(labels_cpu), dtype=torch.bool)
    support_mask[support_idx] = True
    query_idx = all_idx[~support_mask]

    return support_idx, query_idx


def _get_embed_dim(encoder: nn.Module, device: Optional[torch.device] = None) -> int:
    """Infer embed_dim from encoder module."""
    if hasattr(encoder, "embed_dim"):
        return encoder.embed_dim
    # Heuristic: run a dummy forward pass on the encoder's device
    if device is None:
        device = next(encoder.parameters()).device
    encoder.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        out = encoder(dummy)
    return out.shape[-1]
