"""
run_finetune.py
---------------
One-time fine-tuning script for the FedSSL TB Detection demo.

Architecture:
  - Feature Extractor : ViT-Tiny (ImageNet pretrained via timm)
  - Classifier        : Linear(192 -> 2) trained on Shenzhen TB dataset
  - FL Encoder        : encoder_round_013.pt kept for training dashboard/metrics

Why ImageNet pretrained:
  Our MAE encoder was trained for 14 rounds (1 epoch each = 14 total epochs)
  on 20k unlabeled X-rays. This is not enough to learn discriminative features.
  ImageNet-pretrained ViT-Tiny has seen 14M diverse images and produces
  strong, separable embeddings that a linear head can easily classify.

Run once before starting the API:
  python run_finetune.py
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import load_config
from src.datasets.loader import ShenzhenDataset, MontgomeryDataset, get_eval_transform


def load_fl_encoder(config, device):
    """Load the Federated SSL trained ViT-Tiny encoder from checkpoint."""
    from src.models.encoder import get_encoder
    encoder = get_encoder(config.model.backbone, config.model.embed_dim)
    
    ckpt_dir = Path(config.logging.checkpoint_dir)
    best_ckpt = ckpt_dir / "best_encoder.pt"
    
    if not best_ckpt.exists():
        ckpts = sorted(ckpt_dir.glob("encoder_round_*.pt"), key=lambda x: int(x.stem.split("_")[-1]))
        if ckpts:
            best_ckpt = ckpts[-1]
            
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        state = ckpt["encoder_state_dict"] if isinstance(ckpt, dict) and "encoder_state_dict" in ckpt else ckpt
        encoder.load_state_dict(state)
        print(f"[Encoder] Loaded FL trained ViT-Tiny encoder: {best_ckpt.name}")
    else:
        print("[Encoder] WARNING: No FL checkpoint found, using initial ViT-Tiny encoder")
        
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load FL trained encoder
    encoder = load_fl_encoder(config, device)


    # 2. Build linear classification head
    embed_dim   = config.model.embed_dim
    linear_head = nn.Linear(embed_dim, 2).to(device)

    # 3. Load FULL Shenzhen dataset (no validation split - maximize labeled data)
    transform   = get_eval_transform(config.data.image_size)
    shenzhen_ds = ShenzhenDataset(
        root_dir=config.data.shenzhen_path,
        transform=transform,
        image_size=config.data.image_size,
    )
    train_loader = DataLoader(shenzhen_ds, batch_size=32, shuffle=True, num_workers=0)
    print(f"Shenzhen: {len(shenzhen_ds)} images (all used for training)")

    # 4. Extract embeddings once (encoder is frozen)
    def extract(loader, desc):
        embs, labels = [], []
        print(f"Extracting {desc} embeddings...")
        with torch.no_grad():
            for imgs, lbls in loader:
                embs.append(encoder(imgs.to(device)).cpu())
                labels.append(lbls)
        return torch.cat(embs).to(device), torch.cat(labels).to(device)

    train_embs, train_labels = extract(train_loader, "train")

    # Class distribution
    n_normal = (train_labels == 0).sum().item()
    n_tb     = (train_labels == 1).sum().item()
    print(f"Class distribution — Normal: {n_normal}, TB: {n_tb}")

    # 5. Train with CLASS-WEIGHTED loss (penalize missing TB 3x more)
    # This forces the model to actually learn TB features instead of defaulting to Normal
    weight = torch.tensor([1.0, float(n_normal) / max(n_tb, 1)]).to(device)
    print(f"Class weights — Normal: {weight[0]:.2f}, TB: {weight[1]:.2f}")
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(linear_head.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs    = 200

    print(f"\nTraining linear head for {epochs} epochs...")
    best_tb_acc  = 0.0
    best_state   = None

    for epoch in range(epochs):
        linear_head.train()
        logits = linear_head(train_embs)
        loss   = criterion(logits, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 40 == 0:
            linear_head.eval()
            with torch.no_grad():
                preds = linear_head(train_embs).argmax(dim=-1)
                tb_mask  = (train_labels == 1)
                tb_acc   = (preds[tb_mask] == 1).float().mean().item()
                all_acc  = (preds == train_labels).float().mean().item()
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.4f} | Train Acc: {all_acc:.1%} | TB Recall: {tb_acc:.1%}")

            if tb_acc > best_tb_acc:
                best_tb_acc = tb_acc
                best_state  = {k: v.clone() for k, v in linear_head.state_dict().items()}

    if best_state:
        linear_head.load_state_dict(best_state)
    print(f"\nBest TB recall on train set: {best_tb_acc:.1%}")


    # 6. Evaluate on Montgomery (held-out test set)
    print("\nEvaluating on Montgomery test set (never seen during training)...")
    montgomery_ds = MontgomeryDataset(
        root_dir=config.data.montgomery_path,
        transform=transform,
        image_size=config.data.image_size,
    )
    mont_loader = DataLoader(montgomery_ds, batch_size=32, shuffle=False, num_workers=0)

    all_preds, all_true = [], []
    linear_head.eval()
    with torch.no_grad():
        for imgs, lbls in mont_loader:
            emb   = encoder(imgs.to(device))
            preds = linear_head(emb).argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_true.append(lbls)

    all_preds = torch.cat(all_preds)
    all_true  = torch.cat(all_true)

    for c, name in [(0, "Normal"), (1, "TB Positive")]:
        mask = (all_true == c)
        if mask.sum() > 0:
            acc = (all_preds[mask] == all_true[mask]).float().mean().item()
            print(f"  {name:12s}: {acc:.1%} ({int(mask.sum())} images)")
    overall = (all_preds == all_true).float().mean().item()
    print(f"  {'Overall':12s}: {overall:.1%}")

    # 7. Save linear head
    ckpt_dir  = Path(config.logging.checkpoint_dir)
    save_path = ckpt_dir / "linear_head.pt"
    torch.save(linear_head.state_dict(), save_path)
    print(f"\nSaved: {save_path}")
    print("Next: restart the API ->  python src/web/api.py")


if __name__ == "__main__":
    main()
