# Ablation Study: Few-Shot Support Shot Scaling ($K$)
- **Checkpoint**: `experiments\main_experiment\checkpoints\flame_round_099.pt`
- **Random Seed**: `42`

| $K$ Shots/Class | Total Support Scans | Shenzhen AUC | Shenzhen Acc | Shenzhen Sens | Mont. AUC | Mont. Sens |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3** | 6 | 67.71% | 62.50% | 82.28% | 52.76% | 100.00% |
| **5** | 10 | 69.22% | 63.34% | 79.15% | 53.97% | 98.28% |
| **10** | 20 | 71.35% | 65.26% | 79.75% | 55.86% | 100.00% |
| **15** | 30 | 68.85% | 66.46% | 71.65% | 53.47% | 98.28% |
| **20** | 40 | 67.62% | 65.76% | 67.09% | 53.06% | 98.28% |
| **30** | 60 | 67.25% | 65.45% | 67.97% | 53.02% | 98.28% |