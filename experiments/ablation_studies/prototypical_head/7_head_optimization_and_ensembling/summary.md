# Ablation Study 7: Head Hyperparameter Optimization & Multi-Support Ensembling
- **Checkpoint**: `experiments\main_experiment\checkpoints\flame_round_099.pt`
- **Few-Shot Shots ($K$)**: `20` per class
- **Adaptation Epochs**: `50`

### Part A: Softmax Temperature Scaling ($	au$)
| Temperature | Shenzhen ROC-AUC | Shenzhen Accuracy | Shenzhen Sensitivity | Shenzhen Specificity | Montgomery ROC-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2.0** | 84.70% | 77.49% | 74.05% | 81.05% | 55.86% |
| **5.0** | 84.70% | 77.49% | 74.05% | 81.05% | 55.86% |
| **8.0** | 84.70% | 77.49% | 74.05% | 81.05% | 55.86% |
| **10.0** | 84.70% | 77.49% | 74.05% | 81.05% | 55.86% |
| **12.0** | 84.70% | 77.49% | 74.05% | 81.05% | 55.86% |
| **15.0** | 84.70% | 77.49% | 74.05% | 81.05% | 55.86% |
| **20.0** | 84.70% | 77.49% | 74.05% | 81.05% | 55.85% |

### Part B: Learning Rate Sweep (AdamW, 50 Epochs)
| Learning Rate | Shenzhen ROC-AUC | Shenzhen Accuracy | Shenzhen Sensitivity | Shenzhen Specificity | Montgomery ROC-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0003** | 77.16% | 71.38% | 70.25% | 72.55% | 58.58% |
| **0.0005** | 80.79% | 74.12% | 72.15% | 76.14% | 57.77% |
| **0.001** | 84.70% | 77.49% | 74.05% | 81.05% | 55.86% |
| **0.002** | 84.66% | 78.78% | 75.00% | 82.68% | 55.13% |
| **0.003** | 84.27% | 78.14% | 73.42% | 83.01% | 54.96% |

### Part C: Multi-Support Bagging Ensemble
| Ensemble Configuration | Shenzhen ROC-AUC | Shenzhen Accuracy | Shenzhen Sensitivity | Shenzhen Specificity | Montgomery ROC-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Head(s)** | 83.22% | 76.59% | 71.60% | 81.86% | 55.86% |
| **3 Head(s)** | 84.01% | 77.41% | 73.60% | 81.43% | 55.37% |
| **5 Head(s)** | 84.97% | 80.29% | 76.40% | 84.39% | 55.65% |