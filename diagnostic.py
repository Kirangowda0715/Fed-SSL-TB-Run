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

import argparse

parser = argparse.ArgumentParser(description="Inspect few-shot episode and prototypes")
parser.add_argument("--config", default=None, help="Path to config YAML")
parser.add_argument("--checkpoint", default=None, help="Path to checkpoint .pt file")
args = parser.parse_args()

config = load_config(args.config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = None
if args.checkpoint:
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        rel = Path(__file__).resolve().parent / args.checkpoint.replace("/kaggle/working/Fed-SSL-TB/", "")
        if rel.exists():
            checkpoint = rel
else:
    checkpoint_dir = Path(config.logging.checkpoint_dir)
    if not checkpoint_dir.exists():
        rel = Path(__file__).resolve().parent / "experiments" / "checkpoints" / checkpoint_dir.name
        if rel.exists():
            checkpoint_dir = rel
    checkpoints = sorted(
        list(checkpoint_dir.glob("flame_round_*.pt")) + list(checkpoint_dir.glob("encoder_round_*.pt")),
        key=lambda p: int(p.stem.split("_")[-1])
    )
    checkpoint = checkpoints[-1] if checkpoints else None

backbone = config.model.backbone
embed_dim = config.model.embed_dim

if checkpoint and checkpoint.exists():
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    saved_cfg = state.get("config", {})
    backbone = saved_cfg.get("backbone", backbone)
    embed_dim = saved_cfg.get("embed_dim", embed_dim)
    print(f"[Diagnostic] Loading checkpoint: {checkpoint} | backbone={backbone}, embed_dim={embed_dim}")
    encoder = get_encoder(backbone, embed_dim).to(device)
    encoder.load_state_dict(state.get("encoder_state_dict", state))
else:
    print(f"[Diagnostic] No checkpoint loaded | backbone={backbone}, embed_dim={embed_dim}")
    encoder = get_encoder(backbone, embed_dim).to(device)
transform = get_eval_transform(config.data.image_size)
dataset = ShenzhenDataset(config.data.shenzhen_path, transform=transform, image_size=config.data.image_size)
loader = DataLoader(dataset, batch_size=config.ssl.batch_size, shuffle=False)
head, metrics = finetune_local(0, encoder, loader, config, device)
print(f"Support paths: {head.support_paths}")
print(f"Support study IDs: {head.support_study_ids}")
print(f"Projected prototype shape: {tuple(head.prototypes.shape)}")
print(f"Adaptation metrics: {metrics}")
