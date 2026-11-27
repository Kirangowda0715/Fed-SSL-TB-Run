"""
Evaluate Multiple K-Shots upon the Final Chosen Prototypical Head Architecture:
- Backbone: Frozen ViT-Tiny (192-dim, FLAME Round 99 Checkpoint)
- Projection Head: 2-Layer MLP (192 -> 128 -> 128, GELU, 41,216 params)
- Optimization: 50 epochs LOO prototypical adaptation (AdamW, lr=0.001, wd=1e-4)
- Distance Metric: Hyperspherical Cosine Similarity with temperature scaling (tau=10.0)
- Evaluates K in [3, 5, 10, 15, 20, 30] for:
  1. Single Head (Seed 42)
  2. Multi-Seed Mean +- Std (5 Seeds: 42, 101, 202, 303, 404)
  3. 5-Head Bagging Ensemble
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.loader import ShenzhenDataset, MontgomeryDataset, get_eval_transform
from src.models.encoder import get_encoder
from src.models.proto_head import PrototypicalHead
from src.utils.metrics import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing on device: {device}")

# 1. Load Frozen ViT-Tiny Encoder
checkpoint_path = Path("experiments/stage2_round099_checkpoint/flame_round_099.pt")
ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
encoder = get_encoder("vit_tiny", embed_dim=192).to(device)
encoder.load_state_dict(ckpt["encoder_state_dict"])
encoder.eval()

# 2. Extract and Cache Embeddings
transform = get_eval_transform(224)
sh_ds = ShenzhenDataset("data/raw/Shenzhen", transform=transform, image_size=224)
mo_ds = MontgomeryDataset("data/raw/Montgomery", transform=transform, image_size=224)

def cache_latents(ds):
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    all_z, all_y = [], []
    with torch.no_grad():
        for imgs, y in loader:
            z = encoder(imgs.to(device))
            all_z.append(z.cpu())
            all_y.append(torch.as_tensor(y).long())
    return torch.cat(all_z), torch.cat(all_y)

print("Caching Shenzhen & Montgomery latents...")
sh_z, sh_y = cache_latents(sh_ds)
mo_z, mo_y = cache_latents(mo_ds)
print(f"Shenzhen cached: {sh_z.shape}, Montgomery cached: {mo_z.shape}")

def sample_kshot(labels, k, seed):
    rng = np.random.RandomState(seed)
    support = []
    for cls in (0, 1):
        idx = (labels == cls).nonzero(as_tuple=False).flatten().numpy()
        support.extend(rng.choice(idx, k, replace=False))
    support = sorted(support)
    supp_set = set(support)
    query = [i for i in range(len(labels)) if i not in supp_set]
    return torch.tensor(support, dtype=torch.long), torch.tensor(query, dtype=torch.long)

def train_head_loo(support_z, support_y, epochs=50, lr=0.001, seed=42):
    torch.manual_seed(seed)
    head = PrototypicalHead(embed_dim=192, num_classes=2, projection_dim=128).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sz = support_z.to(device)
    sy = support_y.to(device)
    n = len(sy)
    head.train()
    for _ in range(epochs):
        losses = []
        for i in range(n):
            mask = torch.ones(n, dtype=torch.bool, device=device)
            mask[i] = False
            p = head.get_learnable_prototypes(sz[mask], sy[mask])
            loss, _ = head.prototypical_loss(sz[i:i+1], sy[i:i+1], p)
            losses.append(loss)
        total_loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
    head.eval()
    return head

def predict_probs(head, support_z, support_y, query_z, tau=10.0):
    head.eval()
    with torch.no_grad():
        sz = head.projection(support_z.to(device))
        qz = head.projection(query_z.to(device))
        p0 = sz[support_y.to(device) == 0].mean(dim=0)
        p1 = sz[support_y.to(device) == 1].mean(dim=0)
        protos = torch.stack([p0, p1])
        
        qn = F.normalize(qz, p=2, dim=-1)
        pn = F.normalize(protos, p=2, dim=-1)
        logits = torch.mm(qn, pn.t()) * tau
        probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return probs

k_values = [3, 5, 10, 15, 20, 30]
seeds = [42, 101, 202, 303, 404]

results_single = []
results_multi = []
results_ensemble = []

print("\n" + "="*85)
print("EVALUATING MULTIPLE K-SHOTS ON THE FINAL CHOSEN PROTOTYPICAL HEAD (50 Epochs LOO)")
print("="*85)
header = f"{'K (Shots)':<10} | {'Mode':<18} | {'Shenzhen AUC':<14} | {'Accuracy':<10} | {'Sensitivity':<12} | {'Specificity':<12} | {'Mont. AUC':<10}"
print(header)
print("-" * 88)

for k in k_values:
    # 1. Evaluate Single Head (Seed 42)
    s_idx, q_idx = sample_kshot(sh_y, k=k, seed=42)
    head_42 = train_head_loo(sh_z[s_idx], sh_y[s_idx], epochs=50, lr=0.001, seed=42)
    p_sh_42 = predict_probs(head_42, sh_z[s_idx], sh_y[s_idx], sh_z[q_idx])
    p_mo_42 = predict_probs(head_42, sh_z[s_idx], sh_y[s_idx], mo_z)
    
    m_sh_42 = evaluate(sh_y[q_idx].numpy(), p_sh_42)
    m_mo_42 = evaluate(mo_y.numpy(), p_mo_42)
    results_single.append({
        "k": k, "seed": 42,
        "sh_auc": m_sh_42["auc"], "sh_acc": m_sh_42["accuracy"],
        "sh_sens": m_sh_42["sensitivity"], "sh_spec": m_sh_42["specificity"],
        "sh_f1": m_sh_42["f1"], "mo_auc": m_mo_42["auc"]
    })
    print(f"K = {k:<6} | {'Single Head (s42)':<18} | {m_sh_42['auc']*100:>12.2f}% | {m_sh_42['accuracy']*100:>8.2f}% | {m_sh_42['sensitivity']*100:>10.2f}% | {m_sh_42['specificity']*100:>10.2f}% | {m_mo_42['auc']*100:>8.2f}%")
    
    # 2. Multi-Seed Runs and Ensemble
    seed_sh_aucs, seed_sh_accs, seed_sh_sens, seed_sh_specs = [], [], [], []
    seed_mo_aucs = []
    
    # For common ensemble query set
    all_supp_indices = set()
    ensemble_heads_data = []
    for s in seeds:
        si, qi = sample_kshot(sh_y, k=k, seed=s)
        all_supp_indices.update(si.tolist())
        h = train_head_loo(sh_z[si], sh_y[si], epochs=50, lr=0.001, seed=s)
        p_sh = predict_probs(h, sh_z[si], sh_y[si], sh_z[qi])
        p_mo = predict_probs(h, sh_z[si], sh_y[si], mo_z)
        
        m_sh = evaluate(sh_y[qi].numpy(), p_sh)
        m_mo = evaluate(mo_y.numpy(), p_mo)
        seed_sh_aucs.append(m_sh["auc"])
        seed_sh_accs.append(m_sh["accuracy"])
        seed_sh_sens.append(m_sh["sensitivity"])
        seed_sh_specs.append(m_sh["specificity"])
        seed_mo_aucs.append(m_mo["auc"])
        
        ensemble_heads_data.append((h, sh_z[si], sh_y[si]))
        
    results_multi.append({
        "k": k,
        "sh_auc_mean": np.mean(seed_sh_aucs), "sh_auc_std": np.std(seed_sh_aucs),
        "sh_acc_mean": np.mean(seed_sh_accs), "sh_acc_std": np.std(seed_sh_accs),
        "sh_sens_mean": np.mean(seed_sh_sens), "sh_spec_mean": np.mean(seed_sh_specs),
        "mo_auc_mean": np.mean(seed_mo_aucs),
    })
    print(f"K = {k:<6} | {'5-Seed Mean+-Std':<18} | {np.mean(seed_sh_aucs)*100:>5.2f}+-{np.std(seed_sh_aucs)*100:<4.2f}% | {np.mean(seed_sh_accs)*100:>5.2f}+-{np.std(seed_sh_accs)*100:<4.2f}% | {np.mean(seed_sh_sens)*100:>8.2f}% | {np.mean(seed_sh_specs)*100:>8.2f}% | {np.mean(seed_mo_aucs)*100:>8.2f}%")

    # 3. 5-Head Ensemble on disjoint holdout
    common_q_idx = torch.tensor([i for i in range(len(sh_y)) if i not in all_supp_indices], dtype=torch.long)
    ens_probs_sh = []
    ens_probs_mo = []
    for h, sz, sy in ensemble_heads_data:
        ens_probs_sh.append(predict_probs(h, sz, sy, sh_z[common_q_idx]))
        ens_probs_mo.append(predict_probs(h, sz, sy, mo_z))
    
    avg_p_sh = np.mean(ens_probs_sh, axis=0)
    avg_p_mo = np.mean(ens_probs_mo, axis=0)
    
    m_ens_sh = evaluate(sh_y[common_q_idx].numpy(), avg_p_sh)
    m_ens_mo = evaluate(mo_y.numpy(), avg_p_mo)
    results_ensemble.append({
        "k": k,
        "holdout_n": len(common_q_idx),
        "sh_auc": m_ens_sh["auc"],
        "sh_acc": m_ens_sh["accuracy"],
        "sh_sens": m_ens_sh["sensitivity"],
        "sh_spec": m_ens_sh["specificity"],
        "mo_auc": m_ens_mo["auc"],
    })
    print(f"K = {k:<6} | {'5-Head Ensemble':<18} | {m_ens_sh['auc']*100:>12.2f}% | {m_ens_sh['accuracy']*100:>8.2f}% | {m_ens_sh['sensitivity']*100:>10.2f}% | {m_ens_sh['specificity']*100:>10.2f}% | {m_ens_mo['auc']*100:>8.2f}%")
    print("-" * 88)

# Save JSON results
out_data = {
    "architecture": "2-Layer MLP (192 -> 128 -> 128, GELU, LayerNorm)",
    "training": "50 Epochs LOO Prototypical Adaptation, AdamW lr=0.001, wd=1e-4",
    "metric": "Cosine similarity with tau=10.0",
    "single_head_seed_42": results_single,
    "multi_seed_5episodes": results_multi,
    "ensemble_5heads": results_ensemble,
}

out_path = Path("experiments/ablation_studies/prototypical_head/4_k_shot_scaling/final_head_kshot_scaling.json")
with open(out_path, "w") as f:
    json.dump(out_data, f, indent=2)

print(f"\nSaved results to {out_path}")
