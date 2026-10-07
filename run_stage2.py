"""Run the corrected Stage 2 K-shot experiment from an existing Stage 1 checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.client.local_train import finetune_local, evaluate_on_montgomery
from src.datasets.loader import ShenzhenDataset, MontgomeryDataset, get_eval_transform
from src.models.mae import build_mae
from src.utils.config import load_config


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
    print(f"[Stage 2] Device: {device}")
    print(f"[Stage 2] Checkpoint: {checkpoint_path}")
    print(f"[Stage 2] K-shot per class: {config.finetuning.few_shot_k}")

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
    # The checkpoint is loaded on CPU; Stage 2 may run on CUDA.
    encoder = model.encoder.to(device)

    # Use deterministic preprocessing for both support/query and external test.
    # This makes the first research run reproducible and avoids random
    # augmentation being applied during evaluation.
    eval_transform = get_eval_transform(config.data.image_size)

    shenzhen = ShenzhenDataset(
        config.data.shenzhen_path,
        transform=eval_transform,
        image_size=config.data.image_size,
    )
    montgomery = MontgomeryDataset(
        config.data.montgomery_path,
        transform=eval_transform,
        image_size=config.data.image_size,
    )

    if len(shenzhen) == 0:
        raise ValueError("Shenzhen dataset is empty.")
    if len(montgomery) == 0:
        raise ValueError("Montgomery dataset is empty.")

    shenzhen_loader = DataLoader(shenzhen, batch_size=len(shenzhen), shuffle=False)
    montgomery_loader = DataLoader(montgomery, batch_size=32, shuffle=False)

    proto_head, shenzhen_metrics = finetune_local(
        hospital_id=1,
        encoder=encoder,
        shenzhen_loader=shenzhen_loader,
        config=config,
        device=device,
    )

    montgomery_metrics = evaluate_on_montgomery(
        encoder=encoder,
        proto_head=proto_head,
        montgomery_loader=montgomery_loader,
        config=config,
        device=device,
    )

    results = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_round": int(checkpoint.get("round", -1)),
        "few_shot_k": int(config.finetuning.few_shot_k),
        "seed": int(config.finetuning.seed),
        "freeze_encoder": bool(config.finetuning.freeze_encoder),
        "projection_dim": int(config.finetuning.projection_dim),
        "shenzhen_query_metrics": {
            k: float(v) for k, v in shenzhen_metrics.items()
            if isinstance(v, (int, float)) and k != "support_paths"
        },
        "montgomery_metrics": {
            k: float(v) for k, v in montgomery_metrics.items()
            if isinstance(v, (int, float))
        },
        "support_paths": shenzhen_metrics.get("support_paths", []),
        "support_study_ids": shenzhen_metrics.get("support_study_ids", []),
    }

    output_dir = Path(config.logging.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"stage2_k{config.finetuning.few_shot_k}_round{checkpoint.get('round', 0):03d}.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n[Stage 2] Results saved ->", output_path)
    print("\nShenzhen Query:")
    for name, value in results["shenzhen_query_metrics"].items():
        print(f"  {name:20s}: {value:.4f}")
    print("\nMontgomery:")
    for name, value in results["montgomery_metrics"].items():
        print(f"  {name:20s}: {value:.4f}")


if __name__ == "__main__":
    main()
