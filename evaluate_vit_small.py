"""
evaluate_vit_small.py
---------------------
Comprehensive Prototypical Head Evaluation Suite for ViT-Small (Round 55).

Evaluates BOTH:
  1. Within-Domain Performance on Shenzhen (622 held-out test images)
  2. Cross-Hospital Transfer Performance on Montgomery (138 held-out images)

Suites evaluated per dataset:
  - Direct Metric & Temperature Sweep (Euclidean L2, Cosine tau in [1, 5, 10, 20], Centered Cosine)
  - Few-Shot Support Scaling (K = 3, 5, 10, 15, 20, 30)
  - Stage 2 Prototypical Head Fine-Tuning (384d -> 128d Projection space)
  - Multi-Support Bagging Ensemble (M=5 and M=10 bags)
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.proto_head import PrototypicalHead
from src.utils.metrics import evaluate, format_metrics


RESULTS_DIR = Path("experiments/vit_small/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = RESULTS_DIR / "cached_embeddings.pt"


def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def sample_stratified_support(
    labels: torch.Tensor, k: int, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample K support examples per class deterministically."""
    g = torch.Generator().manual_seed(seed)
    c0_idx = (labels == 0).nonzero(as_tuple=True)[0]
    c1_idx = (labels == 1).nonzero(as_tuple=True)[0]

    sel_c0 = c0_idx[torch.randperm(len(c0_idx), generator=g)[:k]]
    sel_c1 = c1_idx[torch.randperm(len(c1_idx), generator=g)[:k]]
    supp_idx = torch.cat([sel_c0, sel_c1])

    mask = torch.ones(len(labels), dtype=torch.bool)
    mask[supp_idx] = False
    query_idx = mask.nonzero(as_tuple=True)[0]
    return supp_idx, query_idx


# ------------------------------------------------------------------------------
# 1. Direct Distance Metrics
# ------------------------------------------------------------------------------
def evaluate_direct_metric(
    supp_z: torch.Tensor,
    supp_y: torch.Tensor,
    test_z: torch.Tensor,
    test_y: torch.Tensor,
) -> Dict[str, Any]:
    y_test = test_y.numpy()
    results = {}

    # Euclidean L2
    p0 = supp_z[supp_y == 0].mean(dim=0)
    p1 = supp_z[supp_y == 1].mean(dim=0)
    d0 = torch.cdist(test_z, p0.unsqueeze(0)).squeeze(-1)
    d1 = torch.cdist(test_z, p1.unsqueeze(0)).squeeze(-1)
    logits_l2 = torch.stack([-d0, -d1], dim=1)
    probs_l2 = F.softmax(logits_l2, dim=-1)[:, 1].numpy()
    results["Euclidean L2"] = evaluate(y_test, probs_l2)

    # Cosine Distance with Temperature
    supp_norm = F.normalize(supp_z, p=2, dim=-1)
    test_norm = F.normalize(test_z, p=2, dim=-1)
    cp0 = F.normalize(supp_norm[supp_y == 0].mean(dim=0), p=2, dim=0)
    cp1 = F.normalize(supp_norm[supp_y == 1].mean(dim=0), p=2, dim=0)
    protos_cos = torch.stack([cp0, cp1])

    for tau in [1.0, 5.0, 10.0, 20.0]:
        sims = torch.mm(test_norm, protos_cos.t()) * tau
        probs_cos = F.softmax(sims, dim=-1)[:, 1].numpy()
        results[f"Cosine (tau={tau})"] = evaluate(y_test, probs_cos)

    # Centered Cosine Distance (Mean Subtraction)
    mu = supp_z.mean(dim=0, keepdim=True)
    c_supp_norm = F.normalize(supp_z - mu, p=2, dim=-1)
    c_test_norm = F.normalize(test_z - mu, p=2, dim=-1)
    ccp0 = F.normalize(c_supp_norm[supp_y == 0].mean(dim=0), p=2, dim=0)
    ccp1 = F.normalize(c_supp_norm[supp_y == 1].mean(dim=0), p=2, dim=0)
    c_protos = torch.stack([ccp0, ccp1])
    c_sims = torch.mm(c_test_norm, c_protos.t()) * 10.0
    c_probs = F.softmax(c_sims, dim=-1)[:, 1].numpy()
    results["Centered Cosine (tau=10.0)"] = evaluate(y_test, c_probs)

    return results


# ------------------------------------------------------------------------------
# 2. Few-Shot Support Scaling (K)
# ------------------------------------------------------------------------------
def evaluate_k_shot_scaling(
    pool_z: torch.Tensor,
    pool_y: torch.Tensor,
    test_z: torch.Tensor,
    test_y: torch.Tensor,
    k_values: List[int] = [3, 5, 10, 15, 20, 30],
    seed: int = 42,
) -> Dict[str, Any]:
    y_test = test_y.numpy()
    results = {}

    for k in k_values:
        supp_idx, _ = sample_stratified_support(pool_y, k=k, seed=seed)
        supp_z, supp_y = pool_z[supp_idx], pool_y[supp_idx]
        mu = supp_z.mean(dim=0, keepdim=True)
        c_supp_norm = F.normalize(supp_z - mu, p=2, dim=-1)
        c_test_norm = F.normalize(test_z - mu, p=2, dim=-1)
        cp0 = F.normalize(c_supp_norm[supp_y == 0].mean(dim=0), p=2, dim=0)
        cp1 = F.normalize(c_supp_norm[supp_y == 1].mean(dim=0), p=2, dim=0)
        protos = torch.stack([cp0, cp1])
        sims = torch.mm(c_test_norm, protos.t()) * 10.0
        probs = F.softmax(sims, dim=-1)[:, 1].numpy()
        results[f"K={k}"] = evaluate(y_test, probs)

    return results


# ------------------------------------------------------------------------------
# 3. Stage 2 Head Fine-Tuning
# ------------------------------------------------------------------------------
def train_and_evaluate_head(
    supp_z: torch.Tensor,
    supp_y: torch.Tensor,
    val_z: torch.Tensor,
    val_y: torch.Tensor,
    test_z: torch.Tensor,
    test_y: torch.Tensor,
    seed: int = 42,
    epochs: int = 60,
    lr: float = 1e-3,
    projection_dim: int = 128,
    device: str = "cpu",
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    s_z = supp_z.to(device)
    s_y = supp_y.to(device)
    v_z = val_z.to(device)
    v_y = val_y.to(device)
    t_z = test_z.to(device)
    y_test = test_y.numpy()

    head = PrototypicalHead(
        embed_dim=supp_z.shape[-1],
        num_classes=2,
        projection_dim=projection_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        head.train()
        optimizer.zero_grad()
        protos = head.get_learnable_prototypes(s_z, s_y)
        loss, _ = head.prototypical_loss(s_z, s_y, prototypes=protos)
        loss.backward()
        optimizer.step()

        head.eval()
        with torch.no_grad():
            eval_protos = head.compute_prototypes(s_z, s_y)
            val_loss, _ = head.prototypical_loss(v_z, v_y, prototypes=eval_protos)
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_state = copy.deepcopy(head.state_dict())

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        final_protos = head.compute_prototypes(s_z, s_y)
        _, probs = head.predict(t_z, prototypes=final_protos)
    probs_tb = probs[:, 1].cpu().numpy()
    return evaluate(y_test, probs_tb)


# ------------------------------------------------------------------------------
# 4. Multi-Support Bagging Ensemble
# ------------------------------------------------------------------------------
def evaluate_bagging_ensemble(
    pool_z: torch.Tensor,
    pool_y: torch.Tensor,
    test_z: torch.Tensor,
    test_y: torch.Tensor,
    k: int = 20,
    num_bags: int = 10,
    base_seed: int = 100,
) -> Dict[str, Any]:
    y_test = test_y.numpy()
    all_bag_probs = []

    for b in range(num_bags):
        seed = base_seed + b * 13
        supp_idx, _ = sample_stratified_support(pool_y, k=k, seed=seed)
        supp_z, supp_y = pool_z[supp_idx], pool_y[supp_idx]
        mu = supp_z.mean(dim=0, keepdim=True)
        c_supp_norm = F.normalize(supp_z - mu, p=2, dim=-1)
        c_test_norm = F.normalize(test_z - mu, p=2, dim=-1)
        cp0 = F.normalize(c_supp_norm[supp_y == 0].mean(dim=0), p=2, dim=0)
        cp1 = F.normalize(c_supp_norm[supp_y == 1].mean(dim=0), p=2, dim=0)
        protos = torch.stack([cp0, cp1])
        sims = torch.mm(c_test_norm, protos.t()) * 10.0
        probs = F.softmax(sims, dim=-1)[:, 1].numpy()
        all_bag_probs.append(probs)

    ensemble_probs = np.mean(all_bag_probs, axis=0)
    return evaluate(y_test, ensemble_probs)


# ------------------------------------------------------------------------------
# Complete Suite Runner for a given Test Set
# ------------------------------------------------------------------------------
def run_evaluation_suite(
    pool_z: torch.Tensor,
    pool_y: torch.Tensor,
    test_z: torch.Tensor,
    test_y: torch.Tensor,
    target_name: str,
    k: int = 20,
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, Any]:
    print(f"\n{'='*75}")
    print(f"  EVALUATION TARGET: {target_name.upper()} ({len(test_y)} test images)")
    print(f"{'='*75}")

    supp_idx, val_idx = sample_stratified_support(pool_y, k=k, seed=seed)
    supp_z, supp_y = pool_z[supp_idx], pool_y[supp_idx]
    val_z, val_y = pool_z[val_idx], pool_y[val_idx]

    # Suite 1: Metrics
    print("\n[Suite 1] Distance Metric & Temperature Comparison...")
    direct = evaluate_direct_metric(supp_z, supp_y, test_z, test_y)
    for name, m in direct.items():
        print(f"  {name:<26} -> {format_metrics(m)}")

    # Suite 2: K-shot scaling
    print(f"\n[Suite 2] Few-Shot Support Scaling (K = 3 to 30)...")
    k_scaling = evaluate_k_shot_scaling(pool_z, pool_y, test_z, test_y, seed=seed)
    for name, m in k_scaling.items():
        print(f"  {name:<26} -> {format_metrics(m)}")

    # Suite 3: Stage 2 Head Fine-Tuning
    print(f"\n[Suite 3] Stage 2 Head Fine-Tuning (384d -> 128d Projection)...")
    adapted = train_and_evaluate_head(supp_z, supp_y, val_z, val_y, test_z, test_y, seed=seed, device=device)
    print(f"  {'Stage 2 Adapted Head':<26} -> {format_metrics(adapted)}")

    # Suite 4: Bagging Ensemble
    print(f"\n[Suite 4] Multi-Support Bagging Ensemble (M=5 and M=10 bags)...")
    bagging_5 = evaluate_bagging_ensemble(pool_z, pool_y, test_z, test_y, k=k, num_bags=5)
    bagging_10 = evaluate_bagging_ensemble(pool_z, pool_y, test_z, test_y, k=k, num_bags=10)
    print(f"  {'Bagging (M=5 bags)':<26} -> {format_metrics(bagging_5)}")
    print(f"  {'Bagging (M=10 bags)':<26} -> {format_metrics(bagging_10)}")

    return {
        "suite_1_direct_metrics": direct,
        "suite_2_k_shot_scaling": k_scaling,
        "suite_3_stage2_adaptation": adapted,
        "suite_4_bagging": {
            "m_5": bagging_5,
            "m_10": bagging_10,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="ViT-Small Prototypical Head Dual-Dataset Evaluation")
    parser.add_argument("--cache", default=str(CACHE_FILE), help="Path to cached embeddings")
    parser.add_argument("--k", type=int, default=20, help="Few-shot support size K")
    args = parser.parse_args()

    cache_path = Path(args.cache)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file {cache_path} not found.")

    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    sh_z, sh_y = data["sh_z"], data["sh_y"]
    mont_z, mont_y = data["mont_z"], data["mont_y"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "#" * 75)
    print("  ViT-Small (Round 55) Dual-Dataset Evaluation Suite")
    print(f"  Shenzhen Dataset  : {sh_z.shape} (Normal={sum(sh_y == 0)}, TB={sum(sh_y == 1)})")
    print(f"  Montgomery Dataset: {mont_z.shape} (Normal={sum(mont_y == 0)}, TB={sum(mont_y == 1)})")
    print("#" * 75)

    # 1. Intra-domain Shenzhen Split (20% held-out test = 132 images, 80% support pool = 530 images)
    g = torch.Generator().manual_seed(42)
    c0_idx = (sh_y == 0).nonzero(as_tuple=True)[0]
    c1_idx = (sh_y == 1).nonzero(as_tuple=True)[0]
    p_c0 = c0_idx[torch.randperm(len(c0_idx), generator=g)]
    p_c1 = c1_idx[torch.randperm(len(c1_idx), generator=g)]
    n0, n1 = int(0.2 * len(c0_idx)), int(0.2 * len(c1_idx))
    sh_test_idx = torch.cat([p_c0[:n0], p_c1[:n1]])
    sh_pool_idx = torch.cat([p_c0[n0:], p_c1[n1:]])

    sh_eval_results = run_evaluation_suite(
        pool_z=sh_z[sh_pool_idx],
        pool_y=sh_y[sh_pool_idx],
        test_z=sh_z[sh_test_idx],
        test_y=sh_y[sh_test_idx],
        target_name="Shenzhen (Within-Domain Held-Out Query)",
        k=args.k,
        device=device,
    )

    # 2. Cross-domain Montgomery Split (Shenzhen full 662 support pool -> Montgomery 138 test)
    mont_eval_results = run_evaluation_suite(
        pool_z=sh_z,
        pool_y=sh_y,
        test_z=mont_z,
        test_y=mont_y,
        target_name="Montgomery (Cross-Domain External Test)",
        k=args.k,
        device=device,
    )

    # Compile JSON report
    full_report = {
        "architecture": "vit_small",
        "embed_dim": 384,
        "round": 55,
        "shenzhen_within_domain": sh_eval_results,
        "montgomery_cross_domain": mont_eval_results,
    }
    json_path = RESULTS_DIR / "evaluation_report.json"
    with open(json_path, "w") as f:
        json.dump(make_serializable(full_report), f, indent=2)

    # Compile Comprehensive Markdown Report
    md_path = RESULTS_DIR / "results.md"
    md_content = f"""# ViT-Small (Round 55) Dual-Dataset Prototypical Head Evaluation Report

- **Backbone**: ViT-Small (`vit_small_patch16_224`, `embed_dim=384`)
- **Stage 1 FL Checkpoint**: `flame_round_055.pt` (Federated MAE, Round 55)

---

## 1. Within-Domain Performance: Shenzhen Test Set (132 Held-Out Images)

Support set sampled from Shenzhen training pool ($K=20$).

| Strategy / Metric | AUC | Accuracy | Balanced Acc | Sensitivity | Specificity | F1-Score |
|---|---|---|---|---|---|---|
| Direct Cosine (\\tau=10.0) | {sh_eval_results['suite_1_direct_metrics']['Cosine (tau=10.0)']['auc']:.4f} | {sh_eval_results['suite_1_direct_metrics']['Cosine (tau=10.0)']['accuracy']:.4f} | {sh_eval_results['suite_1_direct_metrics']['Cosine (tau=10.0)']['balanced_accuracy']:.4f} | {sh_eval_results['suite_1_direct_metrics']['Cosine (tau=10.0)']['sensitivity']:.4f} | {sh_eval_results['suite_1_direct_metrics']['Cosine (tau=10.0)']['specificity']:.4f} | {sh_eval_results['suite_1_direct_metrics']['Cosine (tau=10.0)']['f1']:.4f} |
| Centered Cosine (\\tau=10.0) | {sh_eval_results['suite_1_direct_metrics']['Centered Cosine (tau=10.0)']['auc']:.4f} | {sh_eval_results['suite_1_direct_metrics']['Centered Cosine (tau=10.0)']['accuracy']:.4f} | {sh_eval_results['suite_1_direct_metrics']['Centered Cosine (tau=10.0)']['balanced_accuracy']:.4f} | {sh_eval_results['suite_1_direct_metrics']['Centered Cosine (tau=10.0)']['sensitivity']:.4f} | {sh_eval_results['suite_1_direct_metrics']['Centered Cosine (tau=10.0)']['specificity']:.4f} | {sh_eval_results['suite_1_direct_metrics']['Centered Cosine (tau=10.0)']['f1']:.4f} |
| Stage 2 Adapted Head | {sh_eval_results['suite_3_stage2_adaptation']['auc']:.4f} | {sh_eval_results['suite_3_stage2_adaptation']['accuracy']:.4f} | {sh_eval_results['suite_3_stage2_adaptation']['balanced_accuracy']:.4f} | {sh_eval_results['suite_3_stage2_adaptation']['sensitivity']:.4f} | {sh_eval_results['suite_3_stage2_adaptation']['specificity']:.4f} | {sh_eval_results['suite_3_stage2_adaptation']['f1']:.4f} |
| Multi-Support Bagging (M=10 bags) | **{sh_eval_results['suite_4_bagging']['m_10']['auc']:.4f}** | **{sh_eval_results['suite_4_bagging']['m_10']['accuracy']:.4f}** | **{sh_eval_results['suite_4_bagging']['m_10']['balanced_accuracy']:.4f}** | **{sh_eval_results['suite_4_bagging']['m_10']['sensitivity']:.4f}** | **{sh_eval_results['suite_4_bagging']['m_10']['specificity']:.4f}** | **{sh_eval_results['suite_4_bagging']['m_10']['f1']:.4f}** |

---

## 2. Cross-Hospital Transfer Performance: Montgomery Test Set (138 Images)

Support set sampled from Shenzhen ($K=20$), tested across domain on Montgomery.

| Strategy / Metric | AUC | Accuracy | Balanced Acc | Sensitivity | Specificity | F1-Score |
|---|---|---|---|---|---|---|
| Direct Cosine (\\tau=10.0) | {mont_eval_results['suite_1_direct_metrics']['Cosine (tau=10.0)']['auc']:.4f} | {mont_eval_results['suite_1_direct_metrics']['Cosine (tau=10.0)']['accuracy']:.4f} | {mont_eval_results['suite_1_direct_metrics']['Cosine (tau=10.0)']['balanced_accuracy']:.4f} | {mont_eval_results['suite_1_direct_metrics']['Cosine (tau=10.0)']['sensitivity']:.4f} | {mont_eval_results['suite_1_direct_metrics']['Cosine (tau=10.0)']['specificity']:.4f} | {mont_eval_results['suite_1_direct_metrics']['Cosine (tau=10.0)']['f1']:.4f} |
| Centered Cosine (\\tau=10.0) | {mont_eval_results['suite_1_direct_metrics']['Centered Cosine (tau=10.0)']['auc']:.4f} | {mont_eval_results['suite_1_direct_metrics']['Centered Cosine (tau=10.0)']['accuracy']:.4f} | {mont_eval_results['suite_1_direct_metrics']['Centered Cosine (tau=10.0)']['balanced_accuracy']:.4f} | {mont_eval_results['suite_1_direct_metrics']['Centered Cosine (tau=10.0)']['sensitivity']:.4f} | {mont_eval_results['suite_1_direct_metrics']['Centered Cosine (tau=10.0)']['specificity']:.4f} | {mont_eval_results['suite_1_direct_metrics']['Centered Cosine (tau=10.0)']['f1']:.4f} |
| Stage 2 Adapted Head | {mont_eval_results['suite_3_stage2_adaptation']['auc']:.4f} | {mont_eval_results['suite_3_stage2_adaptation']['accuracy']:.4f} | {mont_eval_results['suite_3_stage2_adaptation']['balanced_accuracy']:.4f} | {mont_eval_results['suite_3_stage2_adaptation']['sensitivity']:.4f} | {mont_eval_results['suite_3_stage2_adaptation']['specificity']:.4f} | {mont_eval_results['suite_3_stage2_adaptation']['f1']:.4f} |
| Multi-Support Bagging (M=10 bags) | {mont_eval_results['suite_4_bagging']['m_10']['auc']:.4f} | {mont_eval_results['suite_4_bagging']['m_10']['accuracy']:.4f} | {mont_eval_results['suite_4_bagging']['m_10']['balanced_accuracy']:.4f} | {mont_eval_results['suite_4_bagging']['m_10']['sensitivity']:.4f} | {mont_eval_results['suite_4_bagging']['m_10']['specificity']:.4f} | {mont_eval_results['suite_4_bagging']['m_10']['f1']:.4f} |
"""
    with open(md_path, "w") as f:
        f.write(md_content)

    print("\n" + "=" * 75)
    print("  [SUCCESS] Dual-Dataset Evaluation Complete!")
    print(f"  JSON Report -> {json_path.resolve()}")
    print(f"  Markdown    -> {md_path.resolve()}")
    print("=" * 75)


if __name__ == "__main__":
    main()
