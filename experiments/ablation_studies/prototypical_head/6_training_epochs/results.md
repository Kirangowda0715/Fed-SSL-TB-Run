# Ablation Study 6: Prototypical Adaptation Training Epochs Sweep

## 📌 Executive Summary
In Stage 2 of the FedSSL-TB pipeline, the Prototypical Head's 2-layer projection MLP (41,216 parameters) is adapted to local hospital data using **Leave-One-Out (LOO) Support Episodes**. This experiment answers the critical question: **"How many training epochs are required to achieve peak diagnostic accuracy, and at what point does the head begin overfitting to the few-shot support set?"**

We evaluate performance across an 8-point progression: **$E \in \{0, 10, 25, 50, 75, 100, 150, 200\}$ epochs**.

---

## 🔬 Experimental Setup
- **Encoder Backbone**: ViT-Tiny (Round 99, `flame_round_099.pt`, frozen).
- **Prototypical Head**: 2-Layer MLP ($192 \to \text{GELU} \to 128 \to 128$, 41,216 learnable parameters).
- **Support Cohort**: Shenzhen Hospital ($K = 20$ shots per class, 40 total reference scans).
- **Optimizer**: AdamW ($\text{LR} = 0.001$, $\text{weight decay} = 10^{-4}$, gradient clip norm $= 1.0$).
- **Query Evaluation Sets**: 
  - **Shenzhen In-Domain Query Set**: 622 unseen radiographs.
  - **Montgomery External Stress Test**: 138 radiographs.
- **Random Seed**: `42`.

---

## 📊 Comprehensive Results Table (Epoch Progression)

| Adaptation Epochs | Shenzhen ROC-AUC | Shenzhen Accuracy | Shenzhen Sensitivity | Shenzhen Specificity | Montgomery ROC-AUC | Montgomery Sensitivity | Clinical Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0 Epochs** *(Unadapted)* | 68.93% | 66.56% | 68.35% | 64.71% | 54.20% | 98.28% | Baseline representation |
| **10 Epochs** | 71.37% | 67.20% | 67.09% | 67.32% | 57.18% | 100.00% | Initial boundary alignment |
| **25 Epochs** | 78.57% | 72.19% | 71.52% | 72.88% | **57.97%** | 100.00% | Rapid cluster contraction |
| **50 Epochs** | <span style="color:green">**84.70%**</span> | 77.49% | 74.05% | 81.05% | 55.86% | 79.31% | <span style="color:green">**Peak ROC-AUC (+15.77% gain)**</span> |
| **75 Epochs** | 84.04% | <span style="color:green">**79.26%**</span> | 75.63% | <span style="color:green">**83.01%**</span> | 54.27% | 50.00% | <span style="color:green">**Peak Accuracy & Specificity**</span> |
| **100 Epochs** *(Default)* | 81.26% | 77.49% | **78.48%** | 76.47% | 51.29% | 60.34% | Standard benchmark match |
| **150 Epochs** | 79.94% | 74.76% | 71.20% | 78.43% | 48.78% | 82.76% | Onset of support overfitting |
| **200 Epochs** | 79.74% | 72.99% | 72.78% | 73.20% | 47.77% | 65.52% | Severe cross-site degradation |

---

## 💡 Key Findings & Scientific Insights

```
                   EPOCH PROGRESSION TRAJECTORY
  AUC (%)
   85% ─┐                      ┌── Peak AUC: 84.70% (50 Epochs)
        │                     ┌┴┐
   80% ─┤                   ┌─┘ └──┐
        │                 ┌─┘      └── Overfitting Plateau
   75% ─┤               ┌─┘            └── 79.74% (200 Epochs)
        │             ┌─┘
   70% ─┤           ┌─┘
        │   ┌───────┘ (Rapid Convergence Phase)
   65% ─┴───┴───────┴───────┴───────┴───────┴───────┴───────►
        0   10      25      50      75     100     150     200 Epochs
```

1. **The Golden Convergence Phase (0 $\to$ 50 Epochs: +15.77% AUC Boost)**:
   - Within the first 50 epochs, in-domain ROC-AUC climbs dramatically from **68.93% $\to$ 84.70%**.
   - *Why?* Leave-one-out episodic training acts as meta-learning on the support set. It tunes the MLP weights to actively contract intra-class variance (pulling healthy lungs toward the Normal center and infected lungs toward the TB center) while widening the inter-class decision boundary margin.

2. **The True Optimal Sweet Spot is 50 to 75 Epochs**:
   - **Peak ROC-AUC is reached at 50 Epochs (84.70%)**.
   - **Peak Accuracy is reached at 75 Epochs (79.26%)**, with specificity reaching **83.01%** (reducing false positive alarms on healthy lungs).
   - Both checkpoints comfortably outperform the default 100-epoch configuration (81.26% AUC).

3. **The Overfitting Plateau (100+ Epochs)**:
   - When training continues beyond 75–100 epochs, performance begins to degrade:
     - In-domain Shenzhen AUC drops from **84.70% $\to$ 79.74%** (-4.96% drop).
     - External Montgomery AUC drops from **57.97% $\to$ 47.77%** (-10.20% collapse!).
   - *Why?* With only 40 support images, the 41k-parameter projection head begins **memorizing the idiosyncratic high-frequency features** of those specific 40 reference scans rather than learning general radiographic pathology.

4. **Clinical Deployment Recommendation**:
   - In production or edge hospital deployments, early stopping should be set to **50–60 epochs**, which cuts adaptation compute time by half while delivering **+3.4% higher AUC (84.7% vs 81.3%)** and superior cross-site stability.

---

## 🔁 Exact Reproduction Command

To reproduce this exact experiment:
```bash
python scripts/run_proto_ablations.py --suite epochs --k 20 --seed 42
```

Associated files in this folder:
- Machine-readable metrics: [`results.json`](results.json)
- Deterministic support indices: [`support_indices.json`](support_indices.json)
- Run environment & Git metadata: [`run_metadata.json`](run_metadata.json)
