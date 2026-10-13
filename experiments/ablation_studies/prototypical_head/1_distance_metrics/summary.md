# Ablation Study: Distance Metrics & Temperature Geometries
- **Checkpoint**: `experiments\main_experiment\checkpoints\flame_round_099.pt`
- **Few-Shot Shots ($K$)**: `20` per class
- **Random Seed**: `42`
- **Date**: `2026-10-13`

| Metric Configuration | Shenzhen AUC | Shenzhen Acc | Shenzhen Sens | Shenzhen Spec | Mont. AUC | Mont. Sens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cosine (tau=1.0)** | 67.62% | 65.76% | 67.09% | 64.38% | 53.06% | 98.28% |
| **Cosine (tau=5.0)** | 67.62% | 65.76% | 67.09% | 64.38% | 53.06% | 98.28% |
| **Cosine (tau=10.0) [Baseline]** | 67.62% | 65.76% | 67.09% | 64.38% | 53.06% | 98.28% |
| **Cosine (tau=20.0)** | 67.62% | 65.76% | 67.09% | 64.38% | 53.06% | 98.28% |
| **Squared Euclidean (-||d||^2)** | 67.29% | 65.92% | 67.72% | 64.05% | 52.69% | 98.28% |
| **L2 Euclidean (-||d||)** | 72.81% | 65.92% | 67.72% | 64.05% | 57.74% | 98.28% |
| **Manhattan Distance (L1)** | 74.68% | 67.20% | 68.35% | 66.01% | 55.37% | 100.00% |
| **Hyperspherical Euclidean** | 67.62% | 65.76% | 67.09% | 64.38% | 53.06% | 98.28% |