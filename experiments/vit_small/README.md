# ViT-Small (384-dim) Experiments Directory

This directory stores all artifacts, checkpoints, logs, and evaluation results for the **ViT-Small** (`vit_small_patch16_224`, `embed_dim=384`) backbone.

---

## Directory Structure

```text
experiments/vit_small/
├── checkpoints/         <- Place your ViT-Small Stage 1 checkpoints (*.pt) here
│   └── .gitkeep
├── logs/                <- FL training logs, diagnostic JSONs, and metrics
│   └── .gitkeep
├── results/             <- Prototype evaluation results, tables, ROC curves, JSON reports
│   └── .gitkeep
└── README.md            <- This guide
```

---

## Model Specifications

| Parameter | Value |
|---|---|
| Backbone | `vit_small` (`vit_small_patch16_224` via timm) |
| Feature Dimension | 384 |
| Embedding Dimension | 384 |
| Pooling | Mean patch token pooling |
| Local Config | `configs/vit_small_384.yaml` |
| Kaggle Config | `configs/kaggle_vit_small_384.yaml` |

---

## Running Prototype Evaluation on ViT-Small Checkpoint

Once your checkpoint is placed in `experiments/vit_small/checkpoints/<your_checkpoint>.pt`:

```bash
# Evaluate prototypical head & few-shot metrics on ViT-Small
python run_proto_experiments.py --config configs/vit_small_384.yaml --checkpoint experiments/vit_small/checkpoints/<your_checkpoint>.pt
```

Or run the full 7-suite reproducible ablation protocol:

```bash
python scripts/run_proto_ablations.py --config configs/vit_small_384.yaml --checkpoint experiments/vit_small/checkpoints/<your_checkpoint>.pt --output-dir experiments/vit_small/results
```
