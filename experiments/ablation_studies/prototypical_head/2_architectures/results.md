# Ablation Study 2: Projection Head Architecture & Latent Dimension

## 📌 Executive Summary
This experiment evaluates whether adding a trainable or non-trainable projection head between the Vision Transformer encoder and the prototype classifier improves representation clustering, and what latent bottleneck dimension ($D$) is optimal.

---

## 🔬 Experimental Setup
- **Encoder Backbone**: ViT-Tiny (5.7M parameters) from Round 99 checkpoint (`flame_round_099.pt`).
- **Support Cohort**: Shenzhen Hospital ($K = 20$ shots per class, 40 total images).
- **Tested Architectures**:
  1. **Direct (None)**: Raw 192-dim encoder embeddings, 0 learnable parameters.
  2. **Linear (192 $\to$ 128d)**: Single linear bottleneck projection.
  3. **Linear (192 $\to$ 192d)**: Single full-rank linear rotation.
  4. **MLP (192 $\to$ GELU $\to$ 64d)**: Tight non-linear bottleneck.
  5. **MLP (192 $\to$ GELU $\to$ 128d)**: Standard non-linear projection.
  6. **MLP (192 $\to$ GELU $\to$ 256d)**: Overcomplete non-linear expansion.
- **Random Seed**: `42`.

---

## 📊 Comprehensive Results Table

| Architecture | Bottleneck Dimension ($D$) | Parameter Count | Shenzhen ROC-AUC | Shenzhen Accuracy | Shenzhen F1-Score | Montgomery ROC-AUC | Montgomery Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Direct (None)** | 192 | **0 params** | 67.62% | 65.76% | 66.56% | 53.06% | 38.41% |
| **Linear Projection** | 128 | 24,704 | 66.73% | 65.76% | 66.14% | 51.53% | 37.68% |
| **Linear Projection** | 192 | 37,056 | 67.71% | 65.76% | 66.46% | 52.95% | 39.13% |
| **2-Layer MLP** | 64 | 16,512 | 67.66% | 65.43% | 65.93% | **53.92%** | 38.41% |
| **2-Layer MLP [Default]** | 128 | 41,216 | **68.06%** | **65.92%** | **66.77%** | 53.53% | 39.13% |
| **2-Layer MLP** | 256 | 115,200 | 65.81% | 64.79% | 65.94% | 52.35% | 39.86% |

---

## 💡 Key Findings & Scientific Insights

1. **MLP-128 Provides the Optimal Balance (68.06% AUC)**:
   - The 2-layer MLP with bottleneck dimension $D=128$ achieves the highest in-domain AUC (68.06%) and accuracy (65.92%) among all tested projection designs.
   - The non-linear GELU activation allows the projection space to fold pathological feature clusters into tighter spheres.

2. **The Danger of Overparameterization ($D=256$)**:
   - Expanding the projection head to $D=256$ (115,200 parameters) caused performance to drop to **65.81% AUC** (-2.25% drop).
   - *Why?* With only 40 support images, an overcomplete latent space increases representation dispersion, pulling prototypes away from the true population distribution.

3. **Direct Baseline Viability**:
   - The zero-parameter direct projection (67.62% AUC) is surprisingly strong and within 0.44% of the MLP head, confirming that the self-supervised federated MAE encoder already learned strong linear cluster representations during Stage 1 pretraining.

---

## 🔁 Exact Reproduction Command

To reproduce this exact experiment:
```bash
python scripts/run_proto_ablations.py --suite architecture --k 20 --seed 42
```

Associated files in this folder:
- Machine-readable metrics: [`results.json`](results.json)
- Deterministic support indices: [`support_indices.json`](support_indices.json)
- Run environment & Git metadata: [`run_metadata.json`](run_metadata.json)
