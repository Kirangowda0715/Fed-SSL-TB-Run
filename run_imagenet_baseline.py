"""Evaluate an ImageNet-pretrained ViT-Tiny with the canonical K-shot prototype protocol.

This is a controlled baseline against the federated MAE encoder:
- same Shenzhen K-shot support selection
- same direct mean-prototype classifier
- same deterministic evaluation transforms
- no adaptation training
- Montgomery remains held out for evaluation
"""

import argparse
import json
import random
from pathlib import Path

import timm
import torch
from torch.utils.data import DataLoader, Subset

from src.datasets.loader import (
    ShenzhenDataset,
    MontgomeryDataset,
    get_eval_transform,
)
from src.utils.config import load_config
from src.utils.metrics import evaluate


METRIC_NAMES = (
    "auc",
    "accuracy",
    "sensitivity",
    "specificity",
    "f1",
    "balanced_accuracy",
)


def sample_kshot(labels, k, seed):
    rng = random.Random(seed)
    support = []

    for cls in (0, 1):
        idx = (labels == cls).nonzero(as_tuple=False).flatten().tolist()
        if len(idx) < k:
            raise ValueError(f"Class {cls} has {len(idx)} samples; {k} required.")
        support.extend(rng.sample(idx, k))

    support = sorted(support)
    support_set = set(support)
    query = [i for i in range(len(labels)) if i not in support_set]
    return torch.tensor(support), torch.tensor(query)


def embeddings(encoder, loader, device):
    out, labels = [], []

    with torch.no_grad():
        for images, y in loader:
            out.append(encoder(images.to(device)))
            labels.append(torch.as_tensor(y).long())

    return torch.cat(out), torch.cat(labels)


def prototype_probabilities(query_embeddings, prototypes):
    q = torch.nn.functional.normalize(query_embeddings, dim=-1)
    p = torch.nn.functional.normalize(prototypes, dim=-1)
    logits = q @ p.T * 10.0
    return torch.softmax(logits, dim=-1)[:, 1]


def printable_metrics(metrics):
    return {name: float(metrics[name]) for name in METRIC_NAMES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/current_stage1_3round.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[ImageNet Baseline] Device: {device}")
    print("[ImageNet Baseline] Encoder: ImageNet-pretrained ViT-Tiny")
    print(f"[ImageNet Baseline] K-shot per class: {cfg.finetuning.few_shot_k}")
    print("[ImageNet Baseline] Learned projection: False")
    print("[ImageNet Baseline] Adaptation epochs: 0")

    encoder = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=True,
        num_classes=0,
    ).to(device).eval()

    transform = get_eval_transform(cfg.data.image_size)

    shenzhen = ShenzhenDataset(
        cfg.data.shenzhen_path,
        transform=transform,
        image_size=cfg.data.image_size,
    )
    montgomery = MontgomeryDataset(
        cfg.data.montgomery_path,
        transform=transform,
        image_size=cfg.data.image_size,
    )

    labels = torch.as_tensor(shenzhen.get_labels(), dtype=torch.long)
    support_idx, query_idx = sample_kshot(
        labels,
        int(cfg.finetuning.few_shot_k),
        int(cfg.finetuning.seed),
    )

    support_loader = DataLoader(
        Subset(shenzhen, support_idx.tolist()),
        batch_size=len(support_idx),
        shuffle=False,
    )
    query_loader = DataLoader(
        Subset(shenzhen, query_idx.tolist()),
        batch_size=32,
        shuffle=False,
    )
    mont_loader = DataLoader(montgomery, batch_size=32, shuffle=False)

    support_emb, support_labels = embeddings(encoder, support_loader, device)
    prototypes = torch.stack([
        support_emb[support_labels == cls].mean(0)
        for cls in (0, 1)
    ])

    q_emb, q_labels = embeddings(encoder, query_loader, device)
    q_probs = prototype_probabilities(q_emb, prototypes).cpu().numpy()
    shenzhen_metrics = printable_metrics(
        evaluate(q_labels.numpy(), q_probs)
    )

    m_emb, m_labels = embeddings(encoder, mont_loader, device)
    m_probs = prototype_probabilities(m_emb, prototypes).cpu().numpy()
    montgomery_metrics = printable_metrics(
        evaluate(m_labels.numpy(), m_probs)
    )

    print("\nImageNet ViT-Tiny Direct Prototype Baseline")
    print("-------------------------------------------")
    print(f"Support Normal : {int((support_labels == 0).sum())}")
    print(f"Support TB     : {int((support_labels == 1).sum())}")

    for title, metrics in (
        ("Shenzhen Query", shenzhen_metrics),
        ("Montgomery", montgomery_metrics),
    ):
        print(f"\n{title}:")
        for name in METRIC_NAMES:
            print(f"  {name:20s}: {metrics[name]:.4f}")

    result = {
        "encoder": "vit_tiny_patch16_224",
        "pretrained": "ImageNet",
        "few_shot_k": int(cfg.finetuning.few_shot_k),
        "seed": int(cfg.finetuning.seed),
        "learned_projection": False,
        "adaptation_epochs": 0,
        "shenzhen_query_metrics": shenzhen_metrics,
        "montgomery_metrics": montgomery_metrics,
        "support_indices": support_idx.tolist(),
    }

    out = Path(cfg.logging.log_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"imagenet_proto_k{cfg.finetuning.few_shot_k}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n[ImageNet Baseline] Results saved -> {path}")


if __name__ == "__main__":
    main()
