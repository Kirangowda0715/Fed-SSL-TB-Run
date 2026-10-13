import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.reproducibility import seed_everything
from src.datasets.loader import ShenzhenDataset, get_eval_transform
from src.models.encoder import get_encoder
from src.client.local_train import finetune_local


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
        config.data.shenzhen_path,
        transform=transform,
        image_size=config.data.image_size,
    )

    if len(shenzhen) == 0:
        raise ValueError("Shenzhen dataset is empty.")

    loader = DataLoader(
        shenzhen,
        batch_size=int(config.ssl.batch_size),
        shuffle=False,
        num_workers=0,
    )

    print("Checkpoint:", checkpoint_path)
    print("Device:", device)
    print("Shenzhen images:", len(shenzhen))

    _, metrics = finetune_local(0, encoder, loader, config, device)

    print("\n=== SHENZHEN QUERY TEST ===")
    for key, value in metrics.items():
        if key not in ("support_paths", "support_study_ids"):
            print(f"{key}: {value}")

    output = {
        "experiment": "shenzhen_repeat_round099_k20",
        "checkpoint_round": 99,
        "few_shot_k_per_class": int(config.finetuning.few_shot_k),
        "seed": int(config.finetuning.seed),
        "freeze_encoder": bool(config.finetuning.freeze_encoder),
        "projection_dim": int(config.finetuning.projection_dim),
        "adaptation_epochs": int(config.finetuning.epochs),
        "image_count": len(shenzhen),
        "query_count": len(shenzhen) - 2 * int(config.finetuning.few_shot_k),
        "metrics": metrics,
    }

    output_path = Path(
        "experiments/main_experiment/metrics/shenzhen_repeat_k20_round099.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    print("\nSaved:", output_path.resolve())


if __name__ == "__main__":
    main()
