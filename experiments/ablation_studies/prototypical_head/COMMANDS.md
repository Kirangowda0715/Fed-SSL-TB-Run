# Prototypical Head Ablation Study Commands

This document lists all executable terminal commands to run, reproduce, and customize each Prototypical Head ablation study.

The primary runner script is located at: [`scripts/run_proto_ablations.py`](../../../scripts/run_proto_ablations.py).

---

## 🚀 Quick Execution Shortcuts

### 1. Run Everything (All 5 Suites in Sequence)
Runs all 5 ablation studies and updates all metrics, JSON files, and summaries:
```bash
python scripts/run_proto_ablations.py --suite all
```
*Windows Batch Shortcut*:
```cmd
cd experiments\ablation_studies\prototypical_head
run_all_ablations.bat all
```

---

## 🧪 Individual Suite Execution Commands

### Suite 1: Distance Metrics & Temperature Geometries
Tests Cosine ($\tau = 1, 5, 10, 20$), Squared Euclidean, L2 Euclidean, Manhattan ($L_1$), and Hyperspherical Euclidean:
```bash
python scripts/run_proto_ablations.py --suite distance --k 20 --seed 42
```
*Outputs saved to*: [`1_distance_metrics/`](1_distance_metrics/)

---

### Suite 2: Projection Head Architecture
Tests Direct (0 params), Linear Projection (128d, 192d), and 2-Layer MLPs (64d, 128d, 256d):
```bash
python scripts/run_proto_ablations.py --suite architecture --k 20 --seed 42
```
*Outputs saved to*: [`2_architectures/`](2_architectures/)

---

### Suite 3: Prototype Formulation (Mean vs. Median)
Tests classical Arithmetic Mean centroid vs. Coordinate-wise Median prototype:
```bash
python scripts/run_proto_ablations.py --suite aggregation --k 20 --seed 42
```
*Outputs saved to*: [`3_prototype_aggregation/`](3_prototype_aggregation/)

---

### Suite 4: Few-Shot Support Shot Sweep ($K$)
Tests sample scaling across $K \in \{3, 5, 10, 15, 20, 30\}$ shots per class:
```bash
python scripts/run_proto_ablations.py --suite kshot --seed 42
```
*Outputs saved to*: [`4_k_shot_scaling/`](4_k_shot_scaling/)

---

### Suite 5: Multi-Seed Monte Carlo Robustness
Evaluates 10 independent random sampling episodes to generate $\text{Mean} \pm \text{Std}$ statistics:
```bash
python scripts/run_proto_ablations.py --suite seeds --k 20
```
*Outputs saved to*: [`5_seed_robustness/`](5_seed_robustness/)

---

### Suite 6: Prototypical Adaptation Training Epochs Sweep
Evaluates adaptation training progression across $E \in \{0, 10, 25, 50, 75, 100, 150, 200\}$ epochs:
```bash
python scripts/run_proto_ablations.py --suite epochs --k 20 --seed 42
```
*Outputs saved to*: [`6_training_epochs/`](6_training_epochs/)

---

## ⚙️ Custom Execution Flags

| Flag | Default | Description | Example Usage |
| :--- | :--- | :--- | :--- |
| `--suite` | `distance` | Which suite to execute (`distance`, `architecture`, `aggregation`, `kshot`, `seeds`, `epochs`, `all`) | `--suite epochs` |
| `--checkpoint` | `flame_round_099.pt` | Path to the federated encoder checkpoint | `--checkpoint experiments/ablation_30_round/checkpoints/flame_round_029.pt` |
| `--k` | `20` | Few-shot support shots per class ($K$) | `--k 10` |
| `--seed` | `42` | Random seed for support set sampling | `--seed 101` |
| `--config` | `configs/local_stage2.yaml` | Path to dataset and model config YAML | `--config configs/local_stage2.yaml` |
