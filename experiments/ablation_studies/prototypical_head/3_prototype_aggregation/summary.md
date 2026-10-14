# Ablation Study: Prototype Centroid Formulation
- **Checkpoint**: `experiments\main_experiment\checkpoints\flame_round_099.pt`
- **Few-Shot Shots ($K$)**: `20` per class
- **Random Seed**: `42`

| Prototype Aggregation | Shenzhen AUC | Shenzhen Acc | Shenzhen Sens | Mont. AUC | Mont. Sens |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Arithmetic Mean Prototype (Standard)** | 67.62% | 65.76% | 67.09% | 53.06% | 98.28% |
| **Coordinate Median Prototype (Outlier Robust)** | 68.01% | 65.27% | 66.77% | 53.60% | 98.28% |