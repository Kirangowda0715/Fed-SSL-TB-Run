# Ablation Study: Projection Head Architecture
- **Checkpoint**: `experiments\main_experiment\checkpoints\flame_round_099.pt`
- **Few-Shot Shots ($K$)**: `20` per class
- **Random Seed**: `42`

| Architecture | Shenzhen AUC | Shenzhen Acc | Shenzhen F1 | Mont. AUC | Mont. Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Direct (None, 192d, 0 params)** | 67.62% | 65.76% | 66.56% | 53.06% | 42.75% |
| **Linear Projection (192 -> 128d)** | 66.73% | 65.76% | 66.14% | 51.53% | 42.03% |
| **Linear Projection (192 -> 192d)** | 67.71% | 65.76% | 66.46% | 52.95% | 42.03% |
| **2-Layer MLP (192 -> GELU -> 64d)** | 67.66% | 65.43% | 65.93% | 53.92% | 42.03% |
| **2-Layer MLP (192 -> GELU -> 128d) [Default]** | 68.06% | 65.92% | 66.77% | 53.53% | 41.30% |
| **2-Layer MLP (192 -> GELU -> 256d)** | 65.81% | 64.79% | 65.94% | 52.35% | 42.75% |