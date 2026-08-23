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

Key fix (v3): Memory efficient data loading and gradient computation.
Instead of loading all images at once onto GPU and keeping all activations for a single backward pass,
we use standard PyTorch DataLoaders and perform backward/optimizer steps per batch.
This resolves CUDA Out-Of-Memory (OOM) errors.
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

    # ── 1. Fast label extraction ──────────────────────────────────────────
    dataset = shenzhen_loader.dataset
    all_labels = _get_dataset_labels(dataset)

    if len(all_labels) == 0:
        print(f"  [Hospital {hospital_id}] No Shenzhen data — skipping fine-tuning.")
        proto_head = PrototypicalHead(embed_dim=embed_dim, num_classes=num_classes)
        return proto_head, {}

    # ── 2. Sample k-shot support set (k per class) ────────────────────────
    support_idx, query_idx = _sample_kshot(all_labels, k=k, num_classes=num_classes)

    support_dataset = Subset(dataset, support_idx)
    query_dataset = Subset(dataset, query_idx)

    # Dataloaders for support and query
    q_batch_size = getattr(config.ssl, "batch_size", 16)
    
    query_loader = DataLoader(
        query_dataset,
        batch_size=q_batch_size,
        shuffle=True,
        num_workers=shenzhen_loader.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )

    # ── 3. Initialize prototypical head ──────────────────────────────────
    proto_head = PrototypicalHead(embed_dim=embed_dim, num_classes=num_classes).to(device)

    # ── 4. Fine-tune encoder on prototypical loss ─────────────────────────
    # Unfreeze encoder for fine-tuning
    encoder.train()
    for param in encoder.parameters():
        param.requires_grad = True

    optimizer = AdamW(encoder.parameters(), lr=config.finetuning.lr, weight_decay=1e-4)
    num_epochs = config.finetuning.epochs

    for epoch in range(num_epochs):
        # Retrieve support set images/labels (fresh augmentations per epoch)
        supp_loader = DataLoader(support_dataset, batch_size=len(support_dataset), shuffle=False)
        support_images, support_labels = next(iter(supp_loader))
        support_images = support_images.to(device)
        support_labels = support_labels.to(device)

        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for q_imgs, q_lbls in query_loader:
            q_imgs = q_imgs.to(device)
            q_lbls = q_lbls.to(device)

            # Compute support embeddings through live encoder (with grad)
            support_emb = encoder(support_images)
            # Compute prototypes from live support embeddings (retains grad)
            prototypes = proto_head.get_learnable_prototypes(support_emb, support_labels)

            # Compute query embeddings
            q_emb = encoder(q_imgs)

            # Prototypical loss
            loss, probs = proto_head.prototypical_loss(q_emb, q_lbls, prototypes)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate metrics
            total_loss += loss.item() * len(q_lbls)
            total_correct += (probs.argmax(dim=-1) == q_lbls).sum().item()
            total_count += len(q_lbls)

        if (epoch + 1) % max(1, num_epochs // 3) == 0:
            mean_loss = total_loss / max(total_count, 1)
            acc = total_correct / max(total_count, 1)
            print(f"    Epoch {epoch+1}/{num_epochs} | Loss: {mean_loss:.4f} | Acc: {acc:.4f}")

    # ── 5. Evaluate on query set ──────────────────────────────────────────
    encoder.eval()
    proto_head.eval()

    eval_query_loader = DataLoader(
        query_dataset,
        batch_size=q_batch_size,
        shuffle=False,
        num_workers=shenzhen_loader.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )

    all_probs_list = []
    all_labels_list = []
    
    with torch.no_grad():
        # Load support set one last time for evaluation
        supp_loader = DataLoader(support_dataset, batch_size=len(support_dataset), shuffle=False)
        support_images, support_labels = next(iter(supp_loader))
        support_images = support_images.to(device)
        support_labels = support_labels.to(device)

        support_emb = encoder(support_images)
        prototypes = proto_head.compute_prototypes(support_emb, support_labels)

        for q_imgs, q_lbls in eval_query_loader:
            q_imgs = q_imgs.to(device)
            q_emb = encoder(q_imgs)
            _, probs = proto_head.predict(q_emb, prototypes)
            all_probs_list.append(probs[:, 1].cpu())
            all_labels_list.append(q_lbls.cpu())

        tb_probs = torch.cat(all_probs_list).numpy()
        y_true   = torch.cat(all_labels_list).numpy()

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

def _get_dataset_labels(dataset) -> torch.Tensor:
    """Helper to extract labels from a dataset recursively and efficiently."""
    if isinstance(dataset, Subset):
        base_labels = _get_dataset_labels(dataset.dataset)
        return torch.tensor([base_labels[idx] for idx in dataset.indices])
    if hasattr(dataset, "get_labels"):
        return torch.tensor(dataset.get_labels())
    
    # Fallback if no helper method is present (loads all samples to inspect label)
    labels = []
    for i in range(len(dataset)):
        _, lbl = dataset[i]
        labels.append(lbl)
    return torch.tensor(labels)


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
