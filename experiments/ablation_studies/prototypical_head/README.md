# Prototypical Head Ablation Studies

This directory contains systematic, reproducible ablation experiments evaluating architectural, geometric, and sample-size variations of the Prototypical Head for Tuberculosis detection.

Execution shortcuts and command documentation: [`COMMANDS.md`](COMMANDS.md) | Windows Batch Runner: [`run_all_ablations.bat`](run_all_ablations.bat).

---

## 📁 Reproducible Ablation Manifest

| Study Directory | Description | Detailed Findings & Report | Key Metric Takeaway |
| :--- | :--- | :---: | :--- |
| [`1_distance_metrics/`](1_distance_metrics/) | Cosine vs. L2 Euclidean vs. Manhattan ($L_1$) vs. Hyperspherical | [**`1_distance_metrics/results.md`**](1_distance_metrics/results.md) | **Manhattan ($L_1$) achieves 74.68% AUC** and **L2 Euclidean achieves 72.81% AUC**, outperforming Cosine (67.62%). |
| [`2_architectures/`](2_architectures/) | Direct (0 params) vs. Linear (192d, 128d) vs. MLP (64d, 128d, 256d) | [**`2_architectures/results.md`**](2_architectures/results.md) | **MLP-128 (68.06% AUC)** and Direct (67.62% AUC) provide best stability; expanding to 256d overfits. |
| [`3_prototype_aggregation/`](3_prototype_aggregation/) | Arithmetic Mean vs. Coordinate Median Centroid | [**`3_prototype_aggregation/results.md`**](3_prototype_aggregation/results.md) | **Coordinate Median (68.01% AUC)** beats Mean (67.62% AUC), demonstrating outlier resilience. |
| [`4_k_shot_scaling/`](4_k_shot_scaling/) | Support Set Shot Sweep ($K \in [3, 5, 10, 15, 20, 30]$) | [**`4_k_shot_scaling/results.md`**](4_k_shot_scaling/results.md) | Peak unadapted sample efficiency occurs at **$K=10$ (71.35% AUC, 79.75% Sensitivity)**. |
| [`5_seed_robustness/`](5_seed_robustness/) | 10-Episode Monte Carlo Sampling (10 random seeds) | [**`5_seed_robustness/results.md`**](5_seed_robustness/results.md) | **$67.89 \pm 5.99\%$ AUC** on Shenzhen across 10 independent random sampling episodes. |
| [`6_training_epochs/`](6_training_epochs/) | Adaptation Training Epochs ($E \in \{0, 10, 25, 50, 75, 100, 150, 200\}$) | [**`6_training_epochs/results.md`**](6_training_epochs/results.md) | **Peak ROC-AUC reaches 84.70% at 50 Epochs** (+15.77% gain); beyond 100 epochs, head overfits. |
| [`7_head_optimization_and_ensembling/`](7_head_optimization_and_ensembling/) | Temperature scaling, LR sweep, and Multi-Support Bagging ($M \in \{1, 3, 5\}$) | [**`7_head_optimization_and_ensembling/results.md`**](7_head_optimization_and_ensembling/results.md) | **5-Head Support Bagging breaks 80% accuracy** (**80.29% Accuracy**, **84.97% AUC**, **84.39% Specificity**). |

---

## 🔄 Exact Reproducibility Instructions

Every ablation subdirectory stores:
1. `results.md`: Complete human-readable analysis, hypothesis, findings, and comparison table.
2. `results.json`: Full machine-readable metrics (AUC, Accuracy, Sensitivity, Specificity, F1, Balanced Accuracy) for both Shenzhen and Montgomery.
3. `support_indices.json`: The exact list of image indices used for support sampling (guaranteeing deterministic support reproduction).
4. `run_metadata.json`: Exact git commit hash, PyTorch version, CUDA GPU device name, and execution timestamp.

### Quick Terminal Commands
```bash
# Re-run Distance Metric suite only
python scripts/run_proto_ablations.py --suite distance

# Re-run Head Architecture suite only
python scripts/run_proto_ablations.py --suite architecture

# Re-run Prototype Aggregation suite only
python scripts/run_proto_ablations.py --suite aggregation

# Re-run K-Shot sweep only
python scripts/run_proto_ablations.py --suite kshot

# Re-run Multi-Seed Monte Carlo suite only
python scripts/run_proto_ablations.py --suite seeds

# Re-run Training Epochs progression suite only
python scripts/run_proto_ablations.py --suite epochs

# Re-run Head Optimization & Multi-Support Ensembling suite only
python scripts/run_proto_ablations.py --suite ensembling

# Re-run ALL suites in sequence
python scripts/run_proto_ablations.py --suite all
```
See [`COMMANDS.md`](COMMANDS.md) for advanced customization flags.
