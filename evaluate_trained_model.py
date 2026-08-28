"""Evaluate the latest federated encoder with the canonical few-shot protocol."""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.client.local_train import finetune_local, evaluate_on_montgomery
from src.datasets.loader import ShenzhenDataset, MontgomeryDataset, get_eval_transform
from src.models.encoder import get_encoder
from src.utils.config import load_config
from src.utils.reproducibility import seed_everything


def main():
    config = load_config()
    seed_everything(int(config.finetuning.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(config.logging.checkpoint_dir)
    checkpoints = sorted(
        list(checkpoint_dir.glob("encoder_round_*.pt")) +
        list(checkpoint_dir.glob("flame_round_*.pt"))
    )
    checkpoint = checkpoint_dir / "best_encoder.pt"
    if not checkpoint.exists():
        if not checkpoints:
            raise FileNotFoundError(f"No encoder checkpoint found in {checkpoint_dir}")
        checkpoint = checkpoints[-1]
    encoder = get_encoder(config.model.backbone, config.model.embed_dim)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    encoder.load_state_dict(state.get("encoder_state_dict", state))
    encoder.to(device)
    transform = get_eval_transform(config.data.image_size)
    shenzhen = ShenzhenDataset(config.data.shenzhen_path, transform=transform, image_size=config.data.image_size)
    montgomery = MontgomeryDataset(config.data.montgomery_path, transform=transform, image_size=config.data.image_size)
    shenzhen_loader = DataLoader(shenzhen, batch_size=config.ssl.batch_size, shuffle=False)
    montgomery_loader = DataLoader(montgomery, batch_size=config.ssl.batch_size, shuffle=False)
    head, adaptation_metrics = finetune_local(0, encoder, shenzhen_loader, config, device)
    test_metrics = evaluate_on_montgomery(encoder, head, montgomery_loader, config=config, device=device)
    print(f"Adaptation metrics: {adaptation_metrics}")
    print(f"Montgomery metrics: {test_metrics}")


if __name__ == "__main__":
    main()
