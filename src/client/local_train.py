"""Canonical FLAME-inspired few-shot adaptation and evaluation pipeline."""

import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from src.models.proto_head import PrototypicalHead
from src.utils.metrics import evaluate
from src.utils.reproducibility import seed_everything


NUM_CLASSES = 2


def _get_dataset_labels(dataset) -> torch.Tensor:
    if isinstance(dataset, Subset):
        labels = _get_dataset_labels(dataset.dataset)
        return labels[torch.as_tensor(dataset.indices)]
    if hasattr(dataset, "get_labels"):
        return torch.as_tensor(dataset.get_labels(), dtype=torch.long)
    return torch.tensor([dataset[index][1] for index in range(len(dataset))], dtype=torch.long)


def _sample_kshot(
    labels: torch.Tensor, k: int, num_classes: int = NUM_CLASSES, seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select exactly k random samples per class; return support and remainder."""
    if k < 1:
        raise ValueError("few_shot_k must be at least 1.")
    labels = torch.as_tensor(labels, dtype=torch.long).cpu()
    rng = random.Random(seed)
    support = []
    for class_id in range(num_classes):
        class_indices = (labels == class_id).nonzero(as_tuple=False).flatten().tolist()
        if len(class_indices) < k:
            raise ValueError(
                f"Class {class_id} has {len(class_indices)} samples; {k} are required."
            )
        support.extend(rng.sample(class_indices, k))
    support = sorted(support)
    support_set = set(support)
    query = [index for index in range(len(labels)) if index not in support_set]
    return torch.tensor(support, dtype=torch.long), torch.tensor(query, dtype=torch.long)


def _loader(dataset, indices, batch_size, source_loader=None, shuffle=False):
    kwargs = {
        "batch_size": max(1, min(batch_size, len(indices))) if len(indices) else batch_size,
        "shuffle": shuffle,
        "num_workers": getattr(source_loader, "num_workers", 0),
        "pin_memory": getattr(source_loader, "pin_memory", False),
    }
    return DataLoader(Subset(dataset, indices.tolist()), **kwargs)


def _batch_embeddings(encoder, loader, device, grad=False):
    embeddings, labels = [], []
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        for images, batch_labels in loader:
            embeddings.append(encoder(images.to(device)))
            labels.append(torch.as_tensor(batch_labels, device=device).long())
    if not embeddings:
        return torch.empty((0, 0), device=device), torch.empty(0, dtype=torch.long, device=device)
    return torch.cat(embeddings), torch.cat(labels)


def _support_records(dataset, indices):
    paths = [str(dataset.image_paths[int(index)]) for index in indices] if hasattr(dataset, "image_paths") else []
    study_ids = [dataset.get_study_id(int(index)) if hasattr(dataset, "get_study_id") else None for index in indices]
    return paths, study_ids


def _protocol_log(k, support_labels, query_count, projection_dim, freeze_encoder, epochs, seed):
    print("\nFew-Shot Protocol\n-----------------")
    print(f"K-shot per class : {k}")
    print(f"Support Normal   : {int((support_labels == 0).sum())}")
    print(f"Support TB       : {int((support_labels == 1).sum())}")
    print(f"Support Total    : {len(support_labels)}")
    print(f"Projection Dim   : {projection_dim}")
    print(f"Encoder Frozen   : {freeze_encoder}")
    print(f"Adaptation Epochs: {epochs}")
    print(f"Seed             : {seed}")
    print(f"Adaptation Query : {query_count}")


def finetune_local(
    hospital_id: int,
    encoder: nn.Module,
    shenzhen_loader: DataLoader,
    config,
    device: Optional[torch.device] = None,
) -> Tuple[PrototypicalHead, Dict[str, Any]]:
    """Adapt a copied encoder and projection head using Shenzhen support/query data."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(config.finetuning.seed)
    seed_everything(seed)
    dataset = shenzhen_loader.dataset
    labels = _get_dataset_labels(dataset)
    k = int(config.finetuning.few_shot_k)
    support_idx, query_idx = _sample_kshot(labels, k, seed=seed)
    embed_dim = getattr(config.model, "embed_dim", _get_embed_dim(encoder, device))
    projection_dim = int(config.finetuning.projection_dim)
    freeze_encoder = bool(config.finetuning.freeze_encoder)
    head = PrototypicalHead(embed_dim, NUM_CLASSES, projection_dim).to(device)
    encoder.to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad = not freeze_encoder
    train_parameters = list(head.parameters())
    if not freeze_encoder:
        train_parameters += list(encoder.parameters())
    optimizer = AdamW(train_parameters, lr=float(config.finetuning.lr), weight_decay=1e-4)
    batch_size = int(getattr(config.ssl, "batch_size", 16))
    support_loader = _loader(dataset, support_idx, len(support_idx), shenzhen_loader)
    query_loader = _loader(dataset, query_idx, batch_size, shenzhen_loader, shuffle=True)
    support_images, support_labels = next(iter(support_loader))
    support_images, support_labels = support_images.to(device), support_labels.to(device).long()
    _protocol_log(k, support_labels, len(query_idx), projection_dim, freeze_encoder, int(config.finetuning.epochs), seed)
    paths, study_ids = _support_records(dataset, support_idx)
    head.support_indices = support_idx.tolist()
    head.support_paths = paths
    head.support_study_ids = study_ids
    head.support_dataset = dataset
    head.support_labels = support_labels.detach().cpu()

    for epoch in range(int(config.finetuning.epochs)):
        if not freeze_encoder:
            encoder.train()
        else:
            encoder.eval()
        head.train()
        for query_images, query_labels in query_loader:
            support_embeddings = encoder(support_images)
            prototypes = head.get_learnable_prototypes(support_embeddings, support_labels)
            query_embeddings = encoder(query_images.to(device))
            loss, _ = head.prototypical_loss(query_embeddings, query_labels.to(device), prototypes)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(train_parameters, 1.0)
            optimizer.step()

    encoder.eval()
    head.eval()
    with torch.no_grad():
        support_embeddings = encoder(support_images)
        prototypes = head.compute_prototypes(support_embeddings, support_labels)
        probabilities, truth = [], []
        for query_images, query_labels in _loader(dataset, query_idx, batch_size, shenzhen_loader):
            _, probs = head.predict(encoder(query_images.to(device)), prototypes)
            probabilities.append(probs[:, 1].cpu())
            truth.append(query_labels)
    metrics = evaluate(torch.cat(truth).numpy(), torch.cat(probabilities).numpy()) if truth else {}
    metrics["support_paths"] = paths
    metrics["support_study_ids"] = study_ids
    return head, metrics


def evaluate_on_montgomery(
    encoder: nn.Module,
    proto_head: PrototypicalHead,
    montgomery_loader: DataLoader,
    support_loader: Optional[DataLoader] = None,
    config=None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Evaluate only on Montgomery using the already-selected Shenzhen support set."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not hasattr(proto_head, "support_dataset"):
        raise ValueError("No canonical Shenzhen support set is attached to the prototype head.")
    if hasattr(montgomery_loader.dataset, "image_paths") and hasattr(proto_head.support_dataset, "image_paths"):
        support_paths = {str(proto_head.support_dataset.image_paths[index].resolve()) for index in proto_head.support_indices}
        test_paths = {str(path.resolve()) for path in montgomery_loader.dataset.image_paths}
        if support_paths & test_paths:
            raise ValueError("Shenzhen support and Montgomery test images overlap.")
    encoder.to(device).eval()
    proto_head.to(device).eval()
    support_idx = torch.tensor(proto_head.support_indices, dtype=torch.long)
    support_loader = _loader(proto_head.support_dataset, support_idx, len(support_idx))
    support_images, support_labels = next(iter(support_loader))
    with torch.no_grad():
        prototypes = proto_head.compute_prototypes(
            encoder(support_images.to(device)), support_labels.to(device)
        )
        probabilities, truth = [], []
        for images, labels in montgomery_loader:
            _, probs = proto_head.predict(encoder(images.to(device)), prototypes)
            probabilities.append(probs[:, 1].cpu())
            truth.append(torch.as_tensor(labels).long())
    if not truth:
        raise ValueError("Montgomery test set is empty.")
    metrics = evaluate(torch.cat(truth).numpy(), torch.cat(probabilities).numpy())
    print("\nMontgomery Test")
    for name in ("auc", "accuracy", "sensitivity", "specificity", "f1", "balanced_accuracy"):
        if name in metrics:
            print(f"{name.replace('_', ' ').title():18s}: {metrics[name]:.4f}")
    return metrics


def _get_embed_dim(encoder, device):
    with torch.no_grad():
        output = encoder(torch.zeros(1, 3, 224, 224, device=device))
    return int(output.shape[-1])
