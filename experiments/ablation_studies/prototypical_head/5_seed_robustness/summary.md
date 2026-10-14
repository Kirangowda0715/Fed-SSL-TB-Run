# Ablation Study: Multi-Seed Monte Carlo Robustness
- **Checkpoint**: `experiments\main_experiment\checkpoints\flame_round_099.pt`
- **Few-Shot Shots ($K$)**: `20` per class
- **Episodes Evaluated**: `10` independent random samplings

| Episode Seed | Shenzhen AUC | Shenzhen Acc | Shenzhen Sens | Mont. AUC | Mont. Sens |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Seed 42 | 67.62% | 65.76% | 67.09% | 53.06% | 98.28% |
| Seed 101 | 71.26% | 66.72% | 68.99% | 54.50% | 98.28% |
| Seed 202 | 74.80% | 69.29% | 60.44% | 58.53% | 86.21% |
| Seed 303 | 68.91% | 65.11% | 59.49% | 53.19% | 96.55% |
| Seed 404 | 67.98% | 66.72% | 70.57% | 52.20% | 98.28% |
| Seed 505 | 50.95% | 51.29% | 49.05% | 50.43% | 5.17% |
| Seed 606 | 67.95% | 67.36% | 69.94% | 51.70% | 100.00% |
| Seed 707 | 69.34% | 68.81% | 68.99% | 52.63% | 100.00% |
| Seed 808 | 69.45% | 68.65% | 68.67% | 52.50% | 98.28% |
| Seed 909 | 70.60% | 64.63% | 74.68% | 58.19% | 36.21% |
| **MEAN $\pm$ STD** | **67.89 $\pm$ 5.99%** | **65.43 $\pm$ 4.95%** | **65.79 $\pm$ 7.06%** | **53.69 $\pm$ 2.54%** | **81.72 $\pm$ 31.52%** |