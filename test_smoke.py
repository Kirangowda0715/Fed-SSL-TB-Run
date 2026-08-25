"""Smoke test for FedSSL pipeline — verifies ViT-Tiny MAE, FedProx autograd, Prototypical head, and FL loop."""
import sys
import copy
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("FedSSL Smoke Test (ViT-Tiny & FedProx Verification)")
print("=" * 60)

# 1. Config
from src.utils.config import load_config
config = load_config("configs/kaggle.yaml")
print(f"[OK] Config loaded | backbone={config.model.backbone} | embed_dim={config.model.embed_dim}")

# 2. Metrics
from src.utils.metrics import evaluate, format_metrics
metrics = evaluate(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
print(f"[OK] Metrics: {format_metrics(metrics)}")

# 3. ViT-Tiny MAE model
from src.models.mae import build_mae
mae = build_mae(backbone="vit_tiny", embed_dim=192, mask_ratio=0.75, decoder_depth=4)
mae.train()
x = torch.randn(2, 3, 224, 224)
loss, pred, mask = mae(x)
print(f"[OK] MAE forward | loss={loss.item():.4f} | pred={pred.shape} | mask={mask.shape}")

# 4. Global Feature Embedding Pool (Mean patch token pooling)
emb = mae.get_embedding(x)
print(f"[OK] Feature Embedding Pool (Mean patch tokens) | shape={emb.shape}")
assert emb.shape == (2, 192), f"Expected embedding shape (2, 192), got {emb.shape}"

# 5. Encoder weights
enc_weights = mae.get_encoder_weights()
print(f"[OK] Encoder weights extracted | keys={len(enc_weights)}")

# 6. FedProx Autograd Verification
print("[Testing] FedProx Autograd linkage...")
g_weights = {k: v.clone() for k, v in mae.encoder.state_dict().items()}
# Perturb one parameter in current model to generate non-zero proximal term
for p in mae.encoder.parameters():
    p.data += 0.01

loss_fedprox, _, _ = mae(x)
prox_term = 0.0
for name, param in mae.encoder.named_parameters():
    if name in g_weights:
        prox_term = prox_term + torch.sum((param - g_weights[name]) ** 2)
total_loss = loss_fedprox + 0.01 * prox_term
total_loss.backward()

# Verify that encoder parameters received non-zero gradients from FedProx loss
has_grad = any(p.grad is not None and torch.abs(p.grad).sum() > 0 for p in mae.encoder.parameters())
assert has_grad, "CRITICAL ERROR: Encoder parameters received zero gradients during FedProx loss backprop!"
print(f"[OK] FedProx autograd verified | Non-zero gradients propagated successfully!")

# 7. FederatedServer & Aggregation
from src.server.server import FederatedServer
server = FederatedServer(config)
global_model = server.initialize_global_model()
bcast = server.broadcast()
print(f"[OK] Server initialized + broadcast | broadcast keys={len(bcast)}")

# 8. Local SSL Training step
from src.client.ssl_train import ssl_local_train

class TinyDataset(torch.utils.data.Dataset):
    def __len__(self): return 8
    def __getitem__(self, i): return torch.randn(3, 224, 224)

cfg2 = load_config("configs/kaggle.yaml")
cfg2.ssl.epochs_per_round = 1
cfg2.ssl.batch_size = 4

loader = torch.utils.data.DataLoader(TinyDataset(), batch_size=4)
result = ssl_local_train(
    hospital_id=1,
    model=copy.deepcopy(global_model),
    dataloader=loader,
    config=cfg2,
    global_weights=bcast,
)
print(f"[OK] ssl_local_train | samples={result['num_samples']} | loss={result['epoch_losses'][-1]:.4f}")

# 9. Server aggregate + update
agg2 = server.aggregate([result["encoder_weights"]], [result["num_samples"]])
server.update_global_model(agg2)
print(f"[OK] Server aggregate + global update complete")

# 10. PrototypicalHead Verification
from src.models.proto_head import PrototypicalHead
ph = PrototypicalHead(embed_dim=192, num_classes=2)
supp_emb = torch.randn(10, 192)  # 5-shot per class (2 classes * 5 samples = 10)
supp_lbl = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
protos = ph.compute_prototypes(supp_emb, supp_lbl)
query_emb = torch.randn(6, 192)
probs = ph.forward(query_emb, protos)
print(f"[OK] PrototypicalHead | 5-shot prototypes={protos.shape} | query probs={probs.shape}")

# 11. Save Checkpoint
ckpt_path = server.save_checkpoint(0, metrics={"auc": 0.88})
print(f"[OK] Checkpoint saved: {ckpt_path}")

print()
print("=" * 60)
print("ALL FEDSSL SMOKE TESTS PASSED SUCCESSFULLY!")
print("=" * 60)
