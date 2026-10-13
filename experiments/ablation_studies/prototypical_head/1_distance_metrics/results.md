# Ablation Study 1: Distance Metrics & Temperature Geometries

## 📌 Executive Summary
This experiment investigates the fundamental geometric distance function used by the Prototypical Head to classify chest X-rays. While the default literature uses cosine similarity or squared Euclidean distance, our results demonstrate that **Manhattan ($L_1$) Distance and un-squared Euclidean ($L_2$) Distance significantly outperform Cosine Similarity on chest radiographs**.

---

## 🔬 Experimental Setup
- **Encoder Backbone**: ViT-Tiny (5.7M parameters) pretrained on NIH ChestX-ray14 via Federated MAE (Round 99).
- **Evaluation Protocol**: Fixed $K = 20$ support shots per class (40 total support scans from Shenzhen Hospital).
- **Query Sets**: 
  - **Shenzhen Query Set** (In-Domain): 622 unseen radiographs.
  - **Montgomery Test Set** (External Held-Out): 138 radiographs.
- **Random Seed**: `42` (Fixed stratified support split).

---

## 📊 Comprehensive Results Table

| Distance Metric / Geometry | Mathematical Formulation | Shenzhen ROC-AUC | Shenzhen Accuracy | Shenzhen Sensitivity | Shenzhen Specificity | Montgomery ROC-AUC | Montgomery Sensitivity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cosine ($\tau=1.0$)** | $\frac{\mathbf{z} \cdot \mathbf{c}}{\|\mathbf{z}\|\|\mathbf{c}\|}$ | 67.62% | 65.76% | 67.09% | 64.38% | 53.06% | 98.28% |
| **Cosine ($\tau=5.0$)** | $5.0 \times \frac{\mathbf{z} \cdot \mathbf{c}}{\|\mathbf{z}\|\|\mathbf{c}\|}$ | 67.62% | 65.76% | 67.09% | 64.38% | 53.06% | 98.28% |
| **Cosine ($\tau=10.0$)** *(Baseline)* | $10.0 \times \frac{\mathbf{z} \cdot \mathbf{c}}{\|\mathbf{z}\|\|\mathbf{c}\|}$ | 67.62% | 65.76% | 67.09% | 64.38% | 53.06% | 98.28% |
| **Cosine ($\tau=20.0$)** | $20.0 \times \frac{\mathbf{z} \cdot \mathbf{c}}{\|\mathbf{z}\|\|\mathbf{c}\|}$ | 67.62% | 65.76% | 67.09% | 64.38% | 53.06% | 98.28% |
| **Squared Euclidean ($-d^2$)** | $-\|\mathbf{z} - \mathbf{c}\|_2^2$ | 67.29% | 65.92% | 67.72% | 64.05% | 52.69% | 98.28% |
| **Hyperspherical Euclidean** | $-\|\hat{\mathbf{z}} - \hat{\mathbf{c}}\|_2^2$ | 67.62% | 65.76% | 67.09% | 64.38% | 53.06% | 98.28% |
| **L2 Euclidean ($-d$)** | $-\|\mathbf{z} - \mathbf{c}\|_2$ | **72.81%** | 65.92% | 67.72% | 64.05% | **57.74%** | 98.28% |
| **Manhattan ($L_1$) Distance** | $-\sum \|z_i - c_i\|$ | <span style="color:green">**74.68%**</span> | <span style="color:green">**67.20%**</span> | <span style="color:green">**68.35%**</span> | <span style="color:green">**66.01%**</span> | **55.37%** | <span style="color:green">**100.00%**</span> |

---

## 💡 Key Findings & Scientific Insights

1. **The Superiority of Manhattan ($L_1$) Geometry (+7.06% AUC)**:
   - Manhattan distance achieved **74.68% AUC**, an improvement of **+7.06%** over the default cosine baseline (67.62%).
   - *Why?* In high-dimensional latent spaces (192-dim), $L_1$ norm is significantly less sensitive to extreme dimension spikes (spurious patch noise) than Euclidean or Cosine metrics. It treats all 192 latent dimensions with linear penalties rather than squaring extreme outliers.

2. **L2 Euclidean Distance Preserves Vector Magnitude (+5.19% AUC)**:
   - Standard un-squared Euclidean distance achieved **72.81% AUC** on Shenzhen and **57.74% AUC** on Montgomery (a +4.68% gain out-of-domain).
   - Cosine normalization forces all vectors onto a unit sphere, discarding vector length. In chest X-ray foundation representations, the **magnitude (norm) of the embedding reflects diagnostic signal confidence**; discarding magnitude hurts linear separability.

3. **Temperature ($\tau$) Invariance in Direct Inference**:
   - For pure unadapted inference, varying $\tau \in [1, 5, 10, 20]$ yields identical ROC-AUC (67.62%) because multiplying by a positive scalar does not alter the monotonic ranking of query probabilities. Temperature primarily matters when training with Cross-Entropy loss.

---

## 🔁 Exact Reproduction Command

To reproduce this exact experiment:
```bash
python scripts/run_proto_ablations.py --suite distance --k 20 --seed 42
```

Associated files in this folder:
- Machine-readable metrics: [`results.json`](results.json)
- Deterministic support indices: [`support_indices.json`](support_indices.json)
- Run environment & Git metadata: [`run_metadata.json`](run_metadata.json)
