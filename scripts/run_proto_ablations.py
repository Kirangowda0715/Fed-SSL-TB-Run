"""
scripts/run_proto_ablations.py
------------------------------
Reproducible Ablation Study Engine for the Prototypical Head.

Supports modular execution of specific experiment suites:
  --suite distance      : Metric Geometries (Cosine, L2 Euclidean, Sq Euclidean, L1 Manhattan)
  --suite architecture  : Head Depth & Dimension (Direct, Linear, MLP-64, MLP-128, MLP-256)
  --suite aggregation   : Prototype Centroid Formulation (Arithmetic Mean vs Coordinate Median)
  --suite kshot         : Support Set Shot Scaling (K in [3, 5, 10, 15, 20, 30])
  --suite seeds         : 10-Episode Monte Carlo Robustness (Mean +- Std)
  --suite all           : Runs all suites in sequence

Every experiment records:
  - results.json (complete query metrics for Shenzhen and Montgomery)
  - support_indices.json (exact image indices sampled for 100% deterministic reproducibility)
  - run_metadata.json (checkpoint, git commit, seed, device, environment)
  - summary.md (formatted comparison table)
"""

import argparse
import datetime
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
from src.utils.config import load_config
from src.utils.metrics import evaluate

BASE_DIR = Path("experiments/ablation_studies/prototypical_head")


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def extract_cached_embeddings(encoder, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    embs, labels = [], []
    encoder.eval()
    with torch.no_grad():
        for imgs, y in loader:
            z = encoder(imgs.to(device))
            embs.append(z.cpu())
            labels.append(torch.as_tensor(y).long())
    return torch.cat(embs), torch.cat(labels)


def sample_kshot(labels: torch.Tensor, k: int, seed: int):
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


def compute_class_prototypes(
    support_z: torch.Tensor,
    support_y: torch.Tensor,
    method: str = "mean",
) -> torch.Tensor:
    prototypes = []
    for cls in (0, 1):
        cls_embs = support_z[support_y == cls]
        if method == "mean":
            prototypes.append(cls_embs.mean(dim=0))
        elif method == "median":
            # Coordinate-wise median: resistant to outlier radiographs
            med, _ = torch.median(cls_embs, dim=0)
            prototypes.append(med)
        else:
            raise ValueError(f"Unknown prototype aggregation method: {method}")
    return torch.stack(prototypes)


def evaluate_query_metric(
    support_z: torch.Tensor,
    support_y: torch.Tensor,
    query_z: torch.Tensor,
    query_y: torch.Tensor,
    metric_type: str = "cosine",
    temperature: float = 10.0,
    aggregation: str = "mean",
    projection_module: nn.Module = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    if projection_module is not None:
        projection_module.eval().to(device)
        with torch.no_grad():
            support_z = projection_module(support_z.to(device))
            query_z = projection_module(query_z.to(device))
    else:
        support_z = support_z.to(device)
        query_z = query_z.to(device)

    support_y = support_y.to(device)
    prototypes = compute_class_prototypes(support_z, support_y, method=aggregation)

    if metric_type == "cosine":
        q_norm = F.normalize(query_z, p=2, dim=-1)
        p_norm = F.normalize(prototypes, p=2, dim=-1)
        logits = torch.mm(q_norm, p_norm.t()) * temperature
        probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    elif metric_type == "euclidean_sq":
        dists = torch.cdist(query_z, prototypes, p=2.0) ** 2
        logits = -dists / max(temperature, 0.1)
        probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    elif metric_type == "euclidean_l2":
        dists = torch.cdist(query_z, prototypes, p=2.0)
        logits = -dists / max(temperature, 0.1)
        probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    elif metric_type == "manhattan_l1":
        dists = torch.cdist(query_z, prototypes, p=1.0)
        logits = -dists / max(temperature, 0.1)
        probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    elif metric_type == "normalized_euclidean":
        q_norm = F.normalize(query_z, p=2, dim=-1)
        p_norm = F.normalize(prototypes, p=2, dim=-1)
        dists = torch.cdist(q_norm, p_norm, p=2.0) ** 2
        logits = -dists * temperature
        probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    else:
        raise ValueError(f"Unknown metric type: {metric_type}")

    eval_out = evaluate(query_y.numpy(), probs)
    return {
        "auc": float(eval_out["auc"]),
        "accuracy": float(eval_out["accuracy"]),
        "sensitivity": float(eval_out["sensitivity"]),
        "specificity": float(eval_out["specificity"]),
        "f1": float(eval_out["f1"]),
        "balanced_accuracy": float(eval_out["balanced_accuracy"]),
    }


def save_reproducibility_artifacts(
    out_dir: Path,
    suite_name: str,
    results_data: Dict[str, Any],
    support_indices: Dict[str, Any],
    checkpoint_path: str,
    seed: int,
    summary_markdown: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. results.json
    with open(out_dir / "results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    # 2. support_indices.json
    with open(out_dir / "support_indices.json", "w") as f:
        json.dump(support_indices, f, indent=2)

    # 3. run_metadata.json
    metadata = {
        "suite": suite_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "checkpoint": str(checkpoint_path),
        "seed": seed,
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # 4. summary.md
    with open(out_dir / "summary.md", "w") as f:
        f.write(summary_markdown)

    print(f"\n[Saved Artifacts] -> {out_dir.resolve()}")


# ==============================================================================
# Suite 1: Distance Metrics & Geometry Ablation
# ==============================================================================
def run_suite_distance(sh_z, sh_y, mont_z, mont_y, checkpoint_path, k=20, seed=42, device="cpu"):
    out_dir = BASE_DIR / "1_distance_metrics"
    print("\n" + "="*75)
    print("RUNNING ABLATION: 1. Distance Metrics & Temperature Geometries")
    print(f"Target Directory: {out_dir}")
    print("="*75)

    supp_idx, query_idx = sample_kshot(sh_y, k=k, seed=seed)
    s_z, s_y = sh_z[supp_idx], sh_y[supp_idx]
    q_z, q_y = sh_z[query_idx], sh_y[query_idx]

    configs = [
        ("Cosine (tau=1.0)", "cosine", 1.0),
        ("Cosine (tau=5.0)", "cosine", 5.0),
        ("Cosine (tau=10.0) [Baseline]", "cosine", 10.0),
        ("Cosine (tau=20.0)", "cosine", 20.0),
        ("Squared Euclidean (-||d||^2)", "euclidean_sq", 10.0),
        ("L2 Euclidean (-||d||)", "euclidean_l2", 1.0),
        ("Manhattan Distance (L1)", "manhattan_l1", 10.0),
        ("Hyperspherical Euclidean", "normalized_euclidean", 10.0),
    ]

    records = []
    md_lines = [
        "# Ablation Study: Distance Metrics & Temperature Geometries",
        f"- **Checkpoint**: `{checkpoint_path}`",
        f"- **Few-Shot Shots ($K$)**: `{k}` per class",
        f"- **Random Seed**: `{seed}`",
        f"- **Date**: `{datetime.date.today()}`\n",
        "| Metric Configuration | Shenzhen AUC | Shenzhen Acc | Shenzhen Sens | Shenzhen Spec | Mont. AUC | Mont. Sens |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    print(f"{'Configuration':<32} | {'Shenzhen AUC':<12} | {'Shenzhen Acc':<12} | {'Shenzhen Sens':<12} | {'Mont. AUC':<10}")
    print("-" * 88)

    for name, m_type, tau in configs:
        sh_res = evaluate_query_metric(s_z, s_y, q_z, q_y, metric_type=m_type, temperature=tau, device=device)
        mont_res = evaluate_query_metric(s_z, s_y, mont_z, mont_y, metric_type=m_type, temperature=tau, device=device)

        records.append({
            "name": name,
            "metric_type": m_type,
            "temperature": tau,
            "shenzhen": sh_res,
            "montgomery": mont_res,
        })

        print(f"{name:<32} | {sh_res['auc']*100:>10.2f}% | {sh_res['accuracy']*100:>10.2f}% | {sh_res['sensitivity']*100:>10.2f}% | {mont_res['auc']*100:>8.2f}%")
        md_lines.append(f"| **{name}** | {sh_res['auc']*100:.2f}% | {sh_res['accuracy']*100:.2f}% | {sh_res['sensitivity']*100:.2f}% | {sh_res['specificity']*100:.2f}% | {mont_res['auc']*100:.2f}% | {mont_res['sensitivity']*100:.2f}% |")

    save_reproducibility_artifacts(
        out_dir=out_dir,
        suite_name="distance_metrics",
        results_data={"configurations": records},
        support_indices={"k": k, "seed": seed, "support_indices": supp_idx.tolist()},
        checkpoint_path=checkpoint_path,
        seed=seed,
        summary_markdown="\n".join(md_lines),
    )


# ==============================================================================
# Suite 2: Head Architecture Ablation
# ==============================================================================
def run_suite_architecture(sh_z, sh_y, mont_z, mont_y, checkpoint_path, k=20, seed=42, device="cpu"):
    out_dir = BASE_DIR / "2_architectures"
    print("\n" + "="*75)
    print("RUNNING ABLATION: 2. Projection Head Architecture & Latent Dimension")
    print(f"Target Directory: {out_dir}")
    print("="*75)

    supp_idx, query_idx = sample_kshot(sh_y, k=k, seed=seed)
    s_z, s_y = sh_z[supp_idx], sh_y[supp_idx]
    q_z, q_y = sh_z[query_idx], sh_y[query_idx]
    embed_dim = sh_z.shape[-1]

    configs = [
        ("Direct (None, 192d, 0 params)", None),
        ("Linear Projection (192 -> 128d)", nn.Linear(embed_dim, 128)),
        ("Linear Projection (192 -> 192d)", nn.Linear(embed_dim, 192)),
        ("2-Layer MLP (192 -> GELU -> 64d)", nn.Sequential(nn.Linear(embed_dim, 64), nn.GELU(), nn.Linear(64, 64))),
        ("2-Layer MLP (192 -> GELU -> 128d) [Default]", nn.Sequential(nn.Linear(embed_dim, 128), nn.GELU(), nn.Linear(128, 128))),
        ("2-Layer MLP (192 -> GELU -> 256d)", nn.Sequential(nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, 256))),
    ]

    records = []
    md_lines = [
        "# Ablation Study: Projection Head Architecture",
        f"- **Checkpoint**: `{checkpoint_path}`",
        f"- **Few-Shot Shots ($K$)**: `{k}` per class",
        f"- **Random Seed**: `{seed}`\n",
        "| Architecture | Shenzhen AUC | Shenzhen Acc | Shenzhen F1 | Mont. AUC | Mont. Acc |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]

    print(f"{'Architecture':<36} | {'Shenzhen AUC':<12} | {'Shenzhen Acc':<12} | {'Shenzhen F1':<12} | {'Mont. AUC':<10}")
    print("-" * 92)

    for name, module in configs:
        sh_res = evaluate_query_metric(s_z, s_y, q_z, q_y, projection_module=module, device=device)
        mont_res = evaluate_query_metric(s_z, s_y, mont_z, mont_y, projection_module=module, device=device)

        records.append({"name": name, "shenzhen": sh_res, "montgomery": mont_res})
        print(f"{name:<36} | {sh_res['auc']*100:>10.2f}% | {sh_res['accuracy']*100:>10.2f}% | {sh_res['f1']*100:>10.2f}% | {mont_res['auc']*100:>8.2f}%")
        md_lines.append(f"| **{name}** | {sh_res['auc']*100:.2f}% | {sh_res['accuracy']*100:.2f}% | {sh_res['f1']*100:.2f}% | {mont_res['auc']*100:.2f}% | {mont_res['accuracy']*100:.2f}% |")

    save_reproducibility_artifacts(
        out_dir=out_dir,
        suite_name="architectures",
        results_data={"configurations": records},
        support_indices={"k": k, "seed": seed, "support_indices": supp_idx.tolist()},
        checkpoint_path=checkpoint_path,
        seed=seed,
        summary_markdown="\n".join(md_lines),
    )


# ==============================================================================
# Suite 3: Prototype Formulation (Mean vs Median)
# ==============================================================================
def run_suite_aggregation(sh_z, sh_y, mont_z, mont_y, checkpoint_path, k=20, seed=42, device="cpu"):
    out_dir = BASE_DIR / "3_prototype_aggregation"
    print("\n" + "="*75)
    print("RUNNING ABLATION: 3. Prototype Formulation (Mean vs Coordinate Median)")
    print(f"Target Directory: {out_dir}")
    print("="*75)

    supp_idx, query_idx = sample_kshot(sh_y, k=k, seed=seed)
    s_z, s_y = sh_z[supp_idx], sh_y[supp_idx]
    q_z, q_y = sh_z[query_idx], sh_y[query_idx]

    methods = [
        ("Arithmetic Mean Prototype (Standard)", "mean"),
        ("Coordinate Median Prototype (Outlier Robust)", "median"),
    ]

    records = []
    md_lines = [
        "# Ablation Study: Prototype Centroid Formulation",
        f"- **Checkpoint**: `{checkpoint_path}`",
        f"- **Few-Shot Shots ($K$)**: `{k}` per class",
        f"- **Random Seed**: `{seed}`\n",
        "| Prototype Aggregation | Shenzhen AUC | Shenzhen Acc | Shenzhen Sens | Mont. AUC | Mont. Sens |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]

    print(f"{'Method':<40} | {'Shenzhen AUC':<12} | {'Shenzhen Acc':<12} | {'Shenzhen Sens':<12} | {'Mont. AUC':<10}")
    print("-" * 92)

    for name, method in methods:
        sh_res = evaluate_query_metric(s_z, s_y, q_z, q_y, aggregation=method, device=device)
        mont_res = evaluate_query_metric(s_z, s_y, mont_z, mont_y, aggregation=method, device=device)

        records.append({"method": name, "shenzhen": sh_res, "montgomery": mont_res})
        print(f"{name:<40} | {sh_res['auc']*100:>10.2f}% | {sh_res['accuracy']*100:>10.2f}% | {sh_res['sensitivity']*100:>10.2f}% | {mont_res['auc']*100:>8.2f}%")
        md_lines.append(f"| **{name}** | {sh_res['auc']*100:.2f}% | {sh_res['accuracy']*100:.2f}% | {sh_res['sensitivity']*100:.2f}% | {mont_res['auc']*100:.2f}% | {mont_res['sensitivity']*100:.2f}% |")

    save_reproducibility_artifacts(
        out_dir=out_dir,
        suite_name="prototype_aggregation",
        results_data={"configurations": records},
        support_indices={"k": k, "seed": seed, "support_indices": supp_idx.tolist()},
        checkpoint_path=checkpoint_path,
        seed=seed,
        summary_markdown="\n".join(md_lines),
    )


# ==============================================================================
# Suite 4: Support Set Size (K-Shot Scaling)
# ==============================================================================
def run_suite_kshot(sh_z, sh_y, mont_z, mont_y, checkpoint_path, k_list=(3, 5, 10, 15, 20, 30), seed=42, device="cpu"):
    out_dir = BASE_DIR / "4_k_shot_scaling"
    print("\n" + "="*75)
    print("RUNNING ABLATION: 4. Few-Shot Support Set Scaling (K Sweep)")
    print(f"Target Directory: {out_dir}")
    print("="*75)

    records = []
    support_dict = {}
    md_lines = [
        "# Ablation Study: Few-Shot Support Shot Scaling ($K$)",
        f"- **Checkpoint**: `{checkpoint_path}`",
        f"- **Random Seed**: `{seed}`\n",
        "| $K$ Shots/Class | Total Support Scans | Shenzhen AUC | Shenzhen Acc | Shenzhen Sens | Mont. AUC | Mont. Sens |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    print(f"{'K (Shots/Class)':<16} | {'Total Support':<14} | {'Shenzhen AUC':<12} | {'Shenzhen Acc':<12} | {'Shenzhen Sens':<12} | {'Mont. AUC':<10}")
    print("-" * 88)

    for k in k_list:
        supp_idx, query_idx = sample_kshot(sh_y, k=k, seed=seed)
        support_dict[f"k_{k}"] = supp_idx.tolist()
        s_z, s_y = sh_z[supp_idx], sh_y[supp_idx]
        q_z, q_y = sh_z[query_idx], sh_y[query_idx]

        sh_res = evaluate_query_metric(s_z, s_y, q_z, q_y, device=device)
        mont_res = evaluate_query_metric(s_z, s_y, mont_z, mont_y, device=device)

        records.append({"k": k, "total_support": 2*k, "shenzhen": sh_res, "montgomery": mont_res})
        print(f"K = {k:<12} | {2*k:<14} | {sh_res['auc']*100:>10.2f}% | {sh_res['accuracy']*100:>10.2f}% | {sh_res['sensitivity']*100:>10.2f}% | {mont_res['auc']*100:>8.2f}%")
        md_lines.append(f"| **{k}** | {2*k} | {sh_res['auc']*100:.2f}% | {sh_res['accuracy']*100:.2f}% | {sh_res['sensitivity']*100:.2f}% | {mont_res['auc']*100:.2f}% | {mont_res['sensitivity']*100:.2f}% |")

    save_reproducibility_artifacts(
        out_dir=out_dir,
        suite_name="k_shot_scaling",
        results_data={"k_shot_progression": records},
        support_indices=support_dict,
        checkpoint_path=checkpoint_path,
        seed=seed,
        summary_markdown="\n".join(md_lines),
    )


# ==============================================================================
# Suite 5: Multi-Seed Robustness (Monte Carlo Episodes)
# ==============================================================================
def run_suite_seeds(sh_z, sh_y, mont_z, mont_y, checkpoint_path, k=20, seeds=(42, 101, 202, 303, 404, 505, 606, 707, 808, 909), device="cpu"):
    out_dir = BASE_DIR / "5_seed_robustness"
    print("\n" + "="*75)
    print(f"RUNNING ABLATION: 5. Multi-Seed Monte Carlo Robustness ({len(seeds)} Random Episodes)")
    print(f"Target Directory: {out_dir}")
    print("="*75)

    episodes = []
    sh_aucs, sh_accs, sh_sens = [], [], []
    mont_aucs, mont_sens = [], []
    supp_dict = {}

    md_lines = [
        "# Ablation Study: Multi-Seed Monte Carlo Robustness",
        f"- **Checkpoint**: `{checkpoint_path}`",
        f"- **Few-Shot Shots ($K$)**: `{k}` per class",
        f"- **Episodes Evaluated**: `{len(seeds)}` independent random samplings\n",
        "| Episode Seed | Shenzhen AUC | Shenzhen Acc | Shenzhen Sens | Mont. AUC | Mont. Sens |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    print(f"{'Episode Seed':<14} | {'Shenzhen AUC':<12} | {'Shenzhen Acc':<12} | {'Shenzhen Sens':<12} | {'Mont. AUC':<12} | {'Mont. Sens':<12}")
    print("-" * 84)

    for s in seeds:
        supp_idx, query_idx = sample_kshot(sh_y, k=k, seed=s)
        supp_dict[f"seed_{s}"] = supp_idx.tolist()
        s_z, s_y = sh_z[supp_idx], sh_y[supp_idx]
        q_z, q_y = sh_z[query_idx], sh_y[query_idx]

        sh_res = evaluate_query_metric(s_z, s_y, q_z, q_y, device=device)
        mont_res = evaluate_query_metric(s_z, s_y, mont_z, mont_y, device=device)

        sh_aucs.append(sh_res["auc"])
        sh_accs.append(sh_res["accuracy"])
        sh_sens.append(sh_res["sensitivity"])
        mont_aucs.append(mont_res["auc"])
        mont_sens.append(mont_res["sensitivity"])

        episodes.append({"seed": s, "shenzhen": sh_res, "montgomery": mont_res})
        print(f"Seed {s:<9} | {sh_res['auc']*100:>10.2f}% | {sh_res['accuracy']*100:>10.2f}% | {sh_res['sensitivity']*100:>10.2f}% | {mont_res['auc']*100:>10.2f}% | {mont_res['sensitivity']*100:>10.2f}%")
        md_lines.append(f"| Seed {s} | {sh_res['auc']*100:.2f}% | {sh_res['accuracy']*100:.2f}% | {sh_res['sensitivity']*100:.2f}% | {mont_res['auc']*100:.2f}% | {mont_res['sensitivity']*100:.2f}% |")

    # Mean +- Std
    sh_auc_m, sh_auc_s = np.mean(sh_aucs)*100, np.std(sh_aucs)*100
    sh_acc_m, sh_acc_s = np.mean(sh_accs)*100, np.std(sh_accs)*100
    sh_sens_m, sh_sens_s = np.mean(sh_sens)*100, np.std(sh_sens)*100
    mont_auc_m, mont_auc_s = np.mean(mont_aucs)*100, np.std(mont_aucs)*100
    mont_sens_m, mont_sens_s = np.mean(mont_sens)*100, np.std(mont_sens)*100

    print("-" * 84)
    print(f"{'MEAN +- STD':<14} | {sh_auc_m:>5.2f}+-{sh_auc_s:<4.2f}% | {sh_acc_m:>5.2f}+-{sh_acc_s:<4.2f}% | {sh_sens_m:>5.2f}+-{sh_sens_s:<4.2f}% | {mont_auc_m:>5.2f}+-{mont_auc_s:<4.2f}% | {mont_sens_m:>5.2f}+-{mont_sens_s:<4.2f}%")
    md_lines.append(f"| **MEAN $\\pm$ STD** | **{sh_auc_m:.2f} $\\pm$ {sh_auc_s:.2f}%** | **{sh_acc_m:.2f} $\\pm$ {sh_acc_s:.2f}%** | **{sh_sens_m:.2f} $\\pm$ {sh_sens_s:.2f}%** | **{mont_auc_m:.2f} $\\pm$ {mont_auc_s:.2f}%** | **{mont_sens_m:.2f} $\\pm$ {mont_sens_s:.2f}%** |")

    save_reproducibility_artifacts(
        out_dir=out_dir,
        suite_name="seed_robustness",
        results_data={
            "episodes": episodes,
            "summary": {
                "shenzhen_auc": f"{sh_auc_m:.2f} +- {sh_auc_s:.2f}%",
                "shenzhen_accuracy": f"{sh_acc_m:.2f} +- {sh_acc_s:.2f}%",
                "shenzhen_sensitivity": f"{sh_sens_m:.2f} +- {sh_sens_s:.2f}%",
                "montgomery_auc": f"{mont_auc_m:.2f} +- {mont_auc_s:.2f}%",
                "montgomery_sensitivity": f"{mont_sens_m:.2f} +- {mont_sens_s:.2f}%",
            }
        },
        support_indices=supp_dict,
        checkpoint_path=checkpoint_path,
        seed=seeds[0],
        summary_markdown="\n".join(md_lines),
    )


# ==============================================================================
# Driver
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Prototypical Head Reproducible Ablation Runner")
    parser.add_argument("--suite", choices=["distance", "architecture", "aggregation", "kshot", "seeds", "all"], default="distance",
                        help="Which ablation suite to execute (default: distance)")
    parser.add_argument("--config", default="configs/local_stage2.yaml")
    parser.add_argument("--checkpoint", default="experiments/main_experiment/checkpoints/flame_round_099.pt")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        ckpt_path = Path("experiments/ablation_30_round/checkpoints/flame_round_029.pt")
        print(f"[Notice] Requested checkpoint not found; fallback to {ckpt_path}")

    # Load frozen encoder
    encoder = get_encoder(cfg.model.backbone, embed_dim=cfg.model.embed_dim)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder_state = state.get("encoder_state_dict", state.get("model_state_dict", state))
    encoder.load_state_dict(encoder_state, strict=False)
    encoder.to(device).eval()

    # Load datasets
    transform = get_eval_transform(int(cfg.data.image_size))
    shenzhen_ds = ShenzhenDataset(cfg.data.shenzhen_path, transform=transform)
    montgomery_ds = MontgomeryDataset(cfg.data.montgomery_path, transform=transform)

    sh_loader = DataLoader(shenzhen_ds, batch_size=32, shuffle=False)
    mont_loader = DataLoader(montgomery_ds, batch_size=32, shuffle=False)

    print("\n" + "="*75)
    print(f"EXTRACTING CACHED LATENT EMBEDDINGS (Device: {device})")
    print(f"Shenzhen: {len(shenzhen_ds)} images | Montgomery: {len(montgomery_ds)} images")
    print("="*75)

    sh_z, sh_y = extract_cached_embeddings(encoder, sh_loader, device)
    mont_z, mont_y = extract_cached_embeddings(encoder, mont_loader, device)
    print(f"Cached Shenzhen Latents:   {sh_z.shape}")
    print(f"Cached Montgomery Latents: {mont_z.shape}")

    # Execute selected suite(s)
    if args.suite in ("distance", "all"):
        run_suite_distance(sh_z, sh_y, mont_z, mont_y, str(ckpt_path), k=args.k, seed=args.seed, device=device)

    if args.suite in ("architecture", "all"):
        run_suite_architecture(sh_z, sh_y, mont_z, mont_y, str(ckpt_path), k=args.k, seed=args.seed, device=device)

    if args.suite in ("aggregation", "all"):
        run_suite_aggregation(sh_z, sh_y, mont_z, mont_y, str(ckpt_path), k=args.k, seed=args.seed, device=device)

    if args.suite in ("kshot", "all"):
        run_suite_kshot(sh_z, sh_y, mont_z, mont_y, str(ckpt_path), seed=args.seed, device=device)

    if args.suite in ("seeds", "all"):
        run_suite_seeds(sh_z, sh_y, mont_z, mont_y, str(ckpt_path), k=args.k, device=device)

    print("\n" + "="*75)
    print("[COMPLETED] All requested ablations executed and saved!")
    print(f"Base Directory: {BASE_DIR.resolve()}")
    print("="*75 + "\n")


if __name__ == "__main__":
    main()
