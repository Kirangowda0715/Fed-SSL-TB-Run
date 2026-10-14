# Ablation Study 6: Prototypical Adaptation Training Epochs Sweep
- **Checkpoint**: `experiments\main_experiment\checkpoints\flame_round_099.pt`
- **Few-Shot Shots ($K$)**: `20` per class
- **Learning Rate**: `0.001` (AdamW)
- **Random Seed**: `42`

| Epochs | Shenzhen ROC-AUC | Shenzhen Accuracy | Shenzhen Sensitivity | Shenzhen Specificity | Montgomery ROC-AUC | Montgomery Sensitivity |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 68.93% | 66.56% | 68.35% | 64.71% | 54.20% | 98.28% |
| **10** | 71.37% | 67.20% | 67.09% | 67.32% | 57.18% | 100.00% |
| **25** | 78.57% | 72.19% | 71.52% | 72.88% | 57.97% | 100.00% |
| **50** | 84.70% | 77.49% | 74.05% | 81.05% | 55.86% | 79.31% |
| **75** | 84.04% | 79.26% | 75.63% | 83.01% | 54.27% | 50.00% |
| **100** | 81.26% | 77.49% | 78.48% | 76.47% | 51.29% | 60.34% |
| **150** | 79.94% | 74.76% | 71.20% | 78.43% | 48.78% | 82.76% |
| **200** | 79.74% | 72.99% | 72.78% | 73.20% | 47.77% | 65.52% |