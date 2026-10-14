"""Compatibility entry point for the canonical FLAME-inspired few-shot adaptation.

This is not a linear probe and never writes ``linear_head.pt``.
"""

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
    import argparse
    parser = argparse.ArgumentParser(description="Few-shot adaptation entry point")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint .pt file")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.finetuning.seed))
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
        checkpoint = checkpoint_dir / "best_encoder.pt"
        if not checkpoint.exists():
            checkpoints = sorted(
                list(checkpoint_dir.glob("flame_round_*.pt")) + list(checkpoint_dir.glob("encoder_round_*.pt")),
                key=lambda p: int(p.stem.split("_")[-1])
            )
            checkpoint = checkpoints[-1] if checkpoints else checkpoint

    backbone = config.model.backbone
    embed_dim = config.model.embed_dim

    if checkpoint and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        saved_cfg = state.get("config", {})
        backbone = saved_cfg.get("backbone", backbone)
        embed_dim = saved_cfg.get("embed_dim", embed_dim)
        print(f"[FineTune] Loading checkpoint: {checkpoint} | backbone={backbone}, embed_dim={embed_dim}")
        encoder = get_encoder(backbone, embed_dim)
        encoder.load_state_dict(state.get("encoder_state_dict", state))
    else:
        print(f"[FineTune] Initializing randomly initialized encoder | backbone={backbone}, embed_dim={embed_dim}")
        encoder = get_encoder(backbone, embed_dim)
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
