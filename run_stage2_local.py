import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.reproducibility import seed_everything
from src.datasets.loader import ShenzhenDataset, MontgomeryDataset, get_eval_transform
from src.models.encoder import get_encoder
from src.client.local_train import finetune_local, evaluate_on_montgomery


def main():
    config = load_config()
    seed_everything(int(config.finetuning.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(config.logging.checkpoint_dir) / "flame_round_099.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("round") != 99:
        raise ValueError(f"Expected round 99, got {checkpoint.get('round')}")

    encoder = get_encoder(config.model.backbone, config.model.embed_dim)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.to(device)

    transform = get_eval_transform(config.data.image_size)
    shenzhen = ShenzhenDataset(
        config.data.shenzhen_path, transform=transform,
        image_size=config.data.image_size
    )
    montgomery = MontgomeryDataset(
        config.data.montgomery_path, transform=transform,
        image_size=config.data.image_size
    )

    if not len(shenzhen) or not len(montgomery):
        raise ValueError("Shenzhen or Montgomery dataset is empty.")

    batch_size = int(config.ssl.batch_size)
    shenzhen_loader = DataLoader(
        shenzhen, batch_size=batch_size, shuffle=False, num_workers=0
    )
    montgomery_loader = DataLoader(
        montgomery, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print("Checkpoint:", checkpoint_path)
    print("Device:", device)
    print("Shenzhen images:", len(shenzhen))
    print("Montgomery images:", len(montgomery))

    head, adaptation_metrics = finetune_local(
        0, encoder, shenzhen_loader, config, device
    )
    test_metrics = evaluate_on_montgomery(
        encoder, head, montgomery_loader, config=config, device=device
    )

    output = {
        "experiment": "stage2_k20_round099",
        "checkpoint": str(checkpoint_path),
        "checkpoint_round": int(checkpoint["round"]),
        "few_shot_k_per_class": int(config.finetuning.few_shot_k),
        "seed": int(config.finetuning.seed),
        "freeze_encoder": bool(config.finetuning.freeze_encoder),
        "projection_dim": int(config.finetuning.projection_dim),
        "adaptation_epochs": int(config.finetuning.epochs),
        "image_size": int(config.data.image_size),
        "batch_size": batch_size,
        "device": str(device),
        "shenzhen_image_count": len(shenzhen),
        "montgomery_image_count": len(montgomery),
        "shenzhen_adaptation_metrics": adaptation_metrics,
        "montgomery_test_metrics": test_metrics,
    }

    output_path = Path(
        "experiments/main_experiment/metrics/stage2_k20_round099.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print("\nSaved results to:", output_path.resolve())
    print("Montgomery metrics:", test_metrics)


if __name__ == "__main__":
    main()
