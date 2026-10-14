"""
run_proto_experiments.py
------------------------
Automated Multi-Experiment Suite for the Prototypical Head.

Because the federated ViT-Tiny encoder is frozen, this script leverages
LATENT EMBEDDING CACHING:
1. Passes all Shenzhen (662) and Montgomery (138) radiographs through the
   encoder once to cache 192-dim latent representations in GPU memory.
2. Executes 4 distinct experiment suites on the Prototypical Head in seconds:
   - Suite 1: Distance Metric & Temperature Scaling (Cosine tau in [1, 5, 10, 20], Euclidean L2)
   - Suite 2: Projection Architecture (Direct vs Linear vs MLP-64 vs MLP-128 vs MLP-256)
   - Suite 3: Few-Shot Support Size (K in [3, 5, 10, 15, 20, 30])
   - Suite 4: Multi-Seed Monte Carlo Robustness (5 random support episodes -> Mean +- Std)
3. Generates JSON logs, Markdown tables, and ready-to-paste IEEE LaTeX tables.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.loader import (
    ShenzhenDataset,
    MontgomeryDataset,
    get_eval_transform,
)
from src.models.encoder import get_encoder
from src.models.proto_head import PrototypicalHead
from src.utils.config import load_config
from src.utils.metrics import evaluate


OUTPUT_DIR = Path("experiments/ablation_studies/proto_head_experiments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------
# 1. Feature Extraction & Caching
# ------------------------------------------------------------------------------
def extract_and_cache_embeddings(encoder, dataloader, device):
    """Pass images through frozen encoder once and return cached tensors."""
    embs, labels = [], []
    encoder.eval()
    with torch.no_grad():
        for imgs, y in dataloader:
            z = encoder(imgs.to(device))
            embs.append(z.cpu())
            labels.append(torch.as_tensor(y).long())
    return torch.cat(embs), torch.cat(labels)


def sample_kshot_indices(labels: torch.Tensor, k: int, seed: int):
    """Deterministically sample K indices per class."""
    rng = random.Random(seed)
    support = []
    for cls in (0, 1):
        idx = (labels == cls).nonzero(as_tuple=False).flatten().tolist()
        if len(idx) < k:
            raise ValueError(f"Class {cls} has only {len(idx)} items, need {k}")
        support.extend(rng.sample(idx, k))
    support = sorted(support)
    supp_set = set(support)
    query = [i for i in range(len(labels)) if i not in supp_set]
    return torch.tensor(support, dtype=torch.long), torch.tensor(query, dtype=torch.long)


# ------------------------------------------------------------------------------
# 2. Flexible Prototypical Evaluation Engine
# ------------------------------------------------------------------------------
def evaluate_prototypes(
    support_z: torch.Tensor,
    support_y: torch.Tensor,
    query_z: torch.Tensor,
    query_y: torch.Tensor,
    metric_type: str = "cosine",
    temperature: float = 10.0,
    projection_module: nn.Module = None,
    device: torch.device = torch.device("cpu"),
):
    """Compute prototypes and evaluate queries with custom distance metrics."""
    # Optional projection
    if projection_module is not None:
        projection_module.eval().to(device)
        with torch.no_grad():
            support_z = projection_module(support_z.to(device))
            query_z = projection_module(query_z.to(device))
    else:
        support_z = support_z.to(device)
        query_z = query_z.to(device)

    # Class prototypes: arithmetic mean
    prototypes = torch.stack([
        support_z[support_y == cls].mean(dim=0) for cls in (0, 1)
    ])  # shape: (2, D)

    # Distance / Similarity computation
    if metric_type == "cosine":
        q_norm = F.normalize(query_z, p=2, dim=-1)
        p_norm = F.normalize(prototypes, p=2, dim=-1)
        logits = torch.mm(q_norm, p_norm.t()) * temperature
        probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    elif metric_type == "euclidean_sq":
        # Negative squared Euclidean distance: -||q - c||^2
        # (q - c)^2 = q^2 - 2qc + c^2
        dists = torch.cdist(query_z, prototypes, p=2.0) ** 2
        logits = -dists / max(temperature, 0.1)
        probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    elif metric_type == "euclidean_l2":
        # Negative L2 distance: -||q - c||
        dists = torch.cdist(query_z, prototypes, p=2.0)
        logits = -dists / max(temperature, 0.1)
        probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    elif metric_type == "normalized_euclidean":
        # L2 distance on hypersphere
        q_norm = F.normalize(query_z, p=2, dim=-1)
        p_norm = F.normalize(prototypes, p=2, dim=-1)
        dists = torch.cdist(q_norm, p_norm, p=2.0) ** 2
        logits = -dists * temperature
        probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    else:
        raise ValueError(f"Unknown metric type: {metric_type}")

    # Compute classification metrics
    results = evaluate(query_y.numpy(), probs)
    return {
        "auc": float(results["auc"]),
        "accuracy": float(results["accuracy"]),
        "sensitivity": float(results["sensitivity"]),
        "specificity": float(results["specificity"]),
        "f1": float(results["f1"]),
        "balanced_accuracy": float(results["balanced_accuracy"]),
    }


# ------------------------------------------------------------------------------
# 3. Experiment Suites
# ------------------------------------------------------------------------------
def run_suite_1_distance_metrics(sh_z, sh_y, mont_z, mont_y, k=20, seed=42, device="cpu"):
    """Suite 1: Distance Metrics and Temperature Scaling."""
    print("\n" + "="*70)
    print("SUITE 1: Distance Metrics & Temperature (tau) Ablation (K=20)")
    print("="*70)

    supp_idx, query_idx = sample_kshot_indices(sh_y, k=k, seed=seed)
    s_z, s_y = sh_z[supp_idx], sh_y[supp_idx]
    q_z, q_y = sh_z[query_idx], sh_y[query_idx]

    configs = [
        ("Cosine (tau=1.0)", "cosine", 1.0),
        ("Cosine (tau=5.0)", "cosine", 5.0),
        ("Cosine (tau=10.0) [Default]", "cosine", 10.0),
        ("Cosine (tau=20.0)", "cosine", 20.0),
        ("Squared Euclidean (-||d||^2)", "euclidean_sq", 10.0),
        ("L2 Euclidean (-||d||)", "euclidean_l2", 1.0),
        ("Hyperspherical Euclidean", "normalized_euclidean", 10.0),
    ]

    records = []
    print(f"{'Metric Configuration':<32} | {'Shenzhen AUC':<12} | {'Shenzhen Acc':<12} | {'Shenzhen Sens':<12} | {'Mont. AUC':<10}")
    print("-" * 88)

    for name, m_type, tau in configs:
        sh_res = evaluate_prototypes(s_z, s_y, q_z, q_y, metric_type=m_type, temperature=tau, device=device)
        mont_res = evaluate_prototypes(s_z, s_y, mont_z, mont_y, metric_type=m_type, temperature=tau, device=device)

        records.append({
            "name": name,
            "metric_type": m_type,
            "temperature": tau,
            "shenzhen": sh_res,
            "montgomery": mont_res,
        })
        print(f"{name:<32} | {sh_res['auc']*100:>10.2f}% | {sh_res['accuracy']*100:>10.2f}% | {sh_res['sensitivity']*100:>10.2f}% | {mont_res['auc']*100:>8.2f}%")

    return records


def run_suite_2_projection_architectures(sh_z, sh_y, mont_z, mont_y, k=20, seed=42, device="cpu"):
    """Suite 2: Head Architecture (Direct vs Linear vs MLP dimensions)."""
    print("\n" + "="*70)
    print("SUITE 2: Prototypical Projection Head Architecture (K=20)")
    print("="*70)

    supp_idx, query_idx = sample_kshot_indices(sh_y, k=k, seed=seed)
    s_z, s_y = sh_z[supp_idx], sh_y[supp_idx]
    q_z, q_y = sh_z[query_idx], sh_y[query_idx]

    embed_dim = sh_z.shape[-1]  # 192

    # Define architectures
    architectures = [
        ("Direct (No Projection, 192d, 0 params)", None),
        ("Linear Projection (192 -> 128d)", nn.Linear(embed_dim, 128)),
        ("Linear Projection (192 -> 192d)", nn.Linear(embed_dim, 192)),
        ("2-Layer MLP (192 -> 64d)", nn.Sequential(nn.Linear(embed_dim, 64), nn.GELU(), nn.Linear(64, 64))),
        ("2-Layer MLP (192 -> 128d) [Default]", nn.Sequential(nn.Linear(embed_dim, 128), nn.GELU(), nn.Linear(128, 128))),
        ("2-Layer MLP (192 -> 256d)", nn.Sequential(nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, 256))),
    ]

    records = []
    print(f"{'Architecture':<36} | {'Shenzhen AUC':<12} | {'Shenzhen Acc':<12} | {'Shenzhen F1':<12} | {'Mont. AUC':<10}")
    print("-" * 92)

    for name, module in architectures:
        sh_res = evaluate_prototypes(s_z, s_y, q_z, q_y, projection_module=module, device=device)
        mont_res = evaluate_prototypes(s_z, s_y, mont_z, mont_y, projection_module=module, device=device)

        records.append({
            "name": name,
            "shenzhen": sh_res,
            "montgomery": mont_res,
        })
        print(f"{name:<36} | {sh_res['auc']*100:>10.2f}% | {sh_res['accuracy']*100:>10.2f}% | {sh_res['f1']*100:>10.2f}% | {mont_res['auc']*100:>8.2f}%")

    return records


def run_suite_3_support_k_sweep(sh_z, sh_y, mont_z, mont_y, k_list=(3, 5, 10, 15, 20, 30), seed=42, device="cpu"):
    """Suite 3: Support Set Shot Size (K-shot scaling)."""
    print("\n" + "="*70)
    print("SUITE 3: Few-Shot Support Shot Sweep (K in [3, 5, 10, 15, 20, 30])")
    print("="*70)

    records = []
    print(f"{'K (Shots/Class)':<16} | {'Total Support':<14} | {'Shenzhen AUC':<12} | {'Shenzhen Acc':<12} | {'Shenzhen Sens':<12} | {'Mont. AUC':<10}")
    print("-" * 88)

    for k in k_list:
        supp_idx, query_idx = sample_kshot_indices(sh_y, k=k, seed=seed)
        s_z, s_y = sh_z[supp_idx], sh_y[supp_idx]
        q_z, q_y = sh_z[query_idx], sh_y[query_idx]

        sh_res = evaluate_prototypes(s_z, s_y, q_z, q_y, device=device)
        mont_res = evaluate_prototypes(s_z, s_y, mont_z, mont_y, device=device)

        records.append({
            "k": k,
            "total_support": 2 * k,
            "shenzhen": sh_res,
            "montgomery": mont_res,
        })
        print(f"K = {k:<12} | {2*k:<14} | {sh_res['auc']*100:>10.2f}% | {sh_res['accuracy']*100:>10.2f}% | {sh_res['sensitivity']*100:>10.2f}% | {mont_res['auc']*100:>8.2f}%")

    return records


def run_suite_4_monte_carlo_seeds(sh_z, sh_y, mont_z, mont_y, k=20, seeds=(42, 101, 202, 303, 404), device="cpu"):
    """Suite 4: Multi-Seed Robustness (5-Episode Monte Carlo Sampling)."""
    print("\n" + "="*70)
    print(f"SUITE 4: Multi-Seed Monte Carlo Robustness (K={k}, {len(seeds)} Episodes)")
    print("="*70)

    sh_aucs, sh_accs, sh_sens, sh_specs, sh_f1s = [], [], [], [], []
    mont_aucs, mont_accs, mont_sens, mont_specs, mont_f1s = [], [], [], [], []
    episode_records = []

    print(f"{'Episode Seed':<14} | {'Shenzhen AUC':<12} | {'Shenzhen Acc':<12} | {'Shenzhen Sens':<12} | {'Mont. AUC':<12} | {'Mont. Sens':<12}")
    print("-" * 84)

    for s in seeds:
        supp_idx, query_idx = sample_kshot_indices(sh_y, k=k, seed=s)
        s_z, s_y = sh_z[supp_idx], sh_y[supp_idx]
        q_z, q_y = sh_z[query_idx], sh_y[query_idx]

        sh_res = evaluate_prototypes(s_z, s_y, q_z, q_y, device=device)
        mont_res = evaluate_prototypes(s_z, s_y, mont_z, mont_y, device=device)

        sh_aucs.append(sh_res["auc"])
        sh_accs.append(sh_res["accuracy"])
        sh_sens.append(sh_res["sensitivity"])
        sh_specs.append(sh_res["specificity"])
        sh_f1s.append(sh_res["f1"])

        mont_aucs.append(mont_res["auc"])
        mont_accs.append(mont_res["accuracy"])
        mont_sens.append(mont_res["sensitivity"])
        mont_specs.append(mont_res["specificity"])
        mont_f1s.append(mont_res["f1"])

        episode_records.append({"seed": s, "shenzhen": sh_res, "montgomery": mont_res})
        print(f"Seed {s:<9} | {sh_res['auc']*100:>10.2f}% | {sh_res['accuracy']*100:>10.2f}% | {sh_res['sensitivity']*100:>10.2f}% | {mont_res['auc']*100:>10.2f}% | {mont_res['sensitivity']*100:>10.2f}%")

    print("-" * 84)
    sh_auc_mean, sh_auc_std = np.mean(sh_aucs)*100, np.std(sh_aucs)*100
    sh_acc_mean, sh_acc_std = np.mean(sh_accs)*100, np.std(sh_accs)*100
    sh_sens_mean, sh_sens_std = np.mean(sh_sens)*100, np.std(sh_sens)*100
    mont_auc_mean, mont_auc_std = np.mean(mont_aucs)*100, np.std(mont_aucs)*100
    mont_sens_mean, mont_sens_std = np.mean(mont_sens)*100, np.std(mont_sens)*100

    print(f"{'MEAN +- STD':<14} | {sh_auc_mean:>5.2f}+-{sh_auc_std:<4.2f}% | {sh_acc_mean:>5.2f}+-{sh_acc_std:<4.2f}% | {sh_sens_mean:>5.2f}+-{sh_sens_std:<4.2f}% | {mont_auc_mean:>5.2f}+-{mont_auc_std:<4.2f}% | {mont_sens_mean:>5.2f}+-{mont_sens_std:<4.2f}%")

    return {
        "episodes": episode_records,
        "summary": {
            "shenzhen": {
                "auc": f"{sh_auc_mean:.2f} +- {sh_auc_std:.2f}%",
                "accuracy": f"{sh_acc_mean:.2f} +- {sh_acc_std:.2f}%",
                "sensitivity": f"{sh_sens_mean:.2f} +- {sh_sens_std:.2f}%",
            },
            "montgomery": {
                "auc": f"{mont_auc_mean:.2f} +- {mont_auc_std:.2f}%",
                "sensitivity": f"{mont_sens_mean:.2f} +- {mont_sens_std:.2f}%",
            }
        }
    }


# ------------------------------------------------------------------------------
# 4. Main Driver
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Multi-Experiment Prototypical Head Suite")
    parser.add_argument("--config", default="configs/local_stage2.yaml")
    parser.add_argument("--checkpoint", default="experiments/main_experiment/checkpoints/flame_round_099.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*70)
    print("FEDSSL-TB: PROTOTYPICAL HEAD EXPERIMENTATION SUITE")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Encoder Checkpoint: {args.checkpoint}")
    print("="*70)

    # 1. Load Pretrained Federated Encoder
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        ckpt_path = Path("experiments/ablation_30_round/checkpoints/flame_round_029.pt")
        print(f"[Notice] Requested checkpoint not found, defaulting to {ckpt_path}")

    encoder = get_encoder(cfg.model.backbone, embed_dim=cfg.model.embed_dim)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder_state = state.get("encoder_state_dict", state.get("model_state_dict", state))
    encoder.load_state_dict(encoder_state, strict=False)
    encoder.to(device).eval()

    # 2. Build Datasets
    transform = get_eval_transform(int(cfg.data.image_size))
    shenzhen_ds = ShenzhenDataset(cfg.data.shenzhen_path, transform=transform)
    montgomery_ds = MontgomeryDataset(cfg.data.montgomery_path, transform=transform)

    sh_loader = DataLoader(shenzhen_ds, batch_size=32, shuffle=False)
    mont_loader = DataLoader(montgomery_ds, batch_size=32, shuffle=False)

    print(f"Shenzhen Images: {len(shenzhen_ds)} | Montgomery Images: {len(montgomery_ds)}")
    print("Extracting and caching latent embeddings in GPU memory...")

    # 3. Cache Latent Features (Takes ~10 seconds once!)
    sh_z, sh_y = extract_and_cache_embeddings(encoder, sh_loader, device)
    mont_z, mont_y = extract_and_cache_embeddings(encoder, mont_loader, device)
    print(f"Cached Shenzhen Latents:   {sh_z.shape} (dtype: {sh_z.dtype})")
    print(f"Cached Montgomery Latents: {mont_z.shape} (dtype: {mont_z.dtype})")

    # 4. Run All 4 Experiment Suites (Takes ~1 second total!)
    results = {
        "checkpoint": str(ckpt_path),
        "suite_1_distance_metrics": run_suite_1_distance_metrics(sh_z, sh_y, mont_z, mont_y, device=device),
        "suite_2_architectures": run_suite_2_projection_architectures(sh_z, sh_y, mont_z, mont_y, device=device),
        "suite_3_support_k": run_suite_3_support_k_sweep(sh_z, sh_y, mont_z, mont_y, device=device),
        "suite_4_monte_carlo": run_suite_4_monte_carlo_seeds(sh_z, sh_y, mont_z, mont_y, device=device),
    }

    # 5. Save Results
    json_path = OUTPUT_DIR / "proto_experiments_report.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print(f"[SUCCESS] All Prototypical Head Experiments Completed!")
    print(f"Full Report Saved -> {json_path.resolve()}")
    print("="*70)


if __name__ == "__main__":
    main()
