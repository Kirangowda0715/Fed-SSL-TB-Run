"""Inspect the canonical few-shot support episode and projected prototypes."""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.client.local_train import finetune_local
from src.datasets.loader import ShenzhenDataset, get_eval_transform
from src.models.encoder import get_encoder
from src.utils.config import load_config

config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = get_encoder(config.model.backbone, config.model.embed_dim).to(device)
checkpoint_dir = Path(config.logging.checkpoint_dir)
checkpoints = sorted(checkpoint_dir.glob("encoder_round_*.pt"))
if checkpoints:
    state = torch.load(checkpoints[-1], map_location=device, weights_only=False)
    encoder.load_state_dict(state.get("encoder_state_dict", state))
transform = get_eval_transform(config.data.image_size)
dataset = ShenzhenDataset(config.data.shenzhen_path, transform=transform, image_size=config.data.image_size)
loader = DataLoader(dataset, batch_size=config.ssl.batch_size, shuffle=False)
head, metrics = finetune_local(0, encoder, loader, config, device)
print(f"Support paths: {head.support_paths}")
print(f"Support study IDs: {head.support_study_ids}")
print(f"Projected prototype shape: {tuple(head.prototypes.shape)}")
print(f"Adaptation metrics: {metrics}")
