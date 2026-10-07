"""Diagnose frozen Stage-1 encoder class separability without training.

Loads an existing Stage-1 checkpoint and evaluates raw encoder embeddings on
Shenzhen and Montgomery. No weights are modified or saved.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.loader import ShenzhenDataset, MontgomeryDataset, get_eval_transform
from src.models.mae import build_mae
from src.utils.config import load_config
from src.utils.metrics import evaluate


def extract_embeddings(encoder, loader, device):
    encoder.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for images, batch_labels, *_ in loader:
            images = images.to(device, non_blocking=True)
            z = encoder(images)
            embeddings.append(z.detach().cpu())
            labels.append(torch.as_tensor(batch_labels).cpu())
    return torch.cat(embeddings).numpy(), torch.cat(labels).numpy()


def cosine_prototype_metrics(embeddings, labels):
    """Classify each sample using leave-one-out class mean prototypes."""
    z = torch.as_tensor(embeddings, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    scores = torch.zeros(len(y), dtype=torch.float32)

    for i in range(len(y)):
        same = (y == y[i])
        same[i] = False
        other = y != y[i]
        if not same.any() or not other.any():
            continue

        proto_same = z[same].mean(dim=0)
        proto_other = z[other].mean(dim=0)
        zi = z[i]
        sim_same = torch.nn.functional.cosine_similarity(
            zi.unsqueeze(0), proto_same.unsqueeze(0)
        ).item()
        sim_other = torch.nn.functional.cosine_similarity(
            zi.unsqueeze(0), proto_other.unsqueeze(0)
        ).item()
        scores[i] = 0.5 + 0.5 * (sim_same - sim_other)

    return evaluate(labels, scores.numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Encoder Diagnostic] Device: {device}")
    print(f"[Encoder Diagnostic] Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "encoder_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain encoder_state_dict.")

    model = build_mae(
        backbone=config.model.backbone,
        embed_dim=config.model.embed_dim,
        mask_ratio=config.model.mask_ratio,
        decoder_depth=config.model.decoder_depth,
        image_size=config.data.image_size,
        projection_dim=config.finetuning.projection_dim,
    )
    model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder = model.encoder.to(device)

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    transform = get_eval_transform(config.data.image_size)
    shenzhen = ShenzhenDataset(
        config.data.shenzhen_path,
        transform=transform,
        image_size=config.data.image_size,
    )
    montgomery = MontgomeryDataset(
        config.data.montgomery_path,
        transform=transform,
        image_size=config.data.image_size,
    )

    shenzhen_loader = DataLoader(shenzhen, batch_size=32, shuffle=False)
    montgomery_loader = DataLoader(montgomery, batch_size=32, shuffle=False)

    print(f"[Encoder Diagnostic] Shenzhen: {len(shenzhen)} images")
    print(f"[Encoder Diagnostic] Montgomery: {len(montgomery)} images")

    shenzhen_z, shenzhen_y = extract_embeddings(encoder, shenzhen_loader, device)
    montgomery_z, montgomery_y = extract_embeddings(encoder, montgomery_loader, device)

    results = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_round": int(checkpoint.get("round", -1)),
        "shenzhen": {
            "num_images": int(len(shenzhen_y)),
            "embedding_dim": int(shenzhen_z.shape[1]),
            "label_counts": {
                "normal": int((shenzhen_y == 0).sum()),
                "tuberculosis": int((shenzhen_y == 1).sum()),
            },
            "leave_one_out_cosine_prototype": cosine_prototype_metrics(
                shenzhen_z, shenzhen_y
            ),
        },
        "montgomery": {
            "num_images": int(len(montgomery_y)),
            "embedding_dim": int(montgomery_z.shape[1]),
            "label_counts": {
                "normal": int((montgomery_y == 0).sum()),
                "tuberculosis": int((montgomery_y == 1).sum()),
            },
            "leave_one_out_cosine_prototype": cosine_prototype_metrics(
                montgomery_z, montgomery_y
            ),
        },
    }

    output_dir = Path(config.logging.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"encoder_diagnostic_round{checkpoint.get('round', 0):03d}.json"
    output_path.write_text(json.dumps(results, indent=2, default=lambda x: x.tolist()), encoding="utf-8")

    for name in ("shenzhen", "montgomery"):
        metrics = results[name]["leave_one_out_cosine_prototype"]
        print(f"\n{name.title()} — frozen encoder, leave-one-out cosine prototypes")
        for key in ("auc", "accuracy", "sensitivity", "specificity", "f1", "balanced_accuracy"):
            print(f"  {key:20s}: {metrics[key]:.4f}")

    print(f"\n[Encoder Diagnostic] Results saved -> {output_path}")


if __name__ == "__main__":
    main()
