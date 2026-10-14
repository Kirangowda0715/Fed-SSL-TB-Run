# ViT-Small (Round 55) Dual-Dataset Prototypical Head Evaluation Report

- **Backbone**: ViT-Small (`vit_small_patch16_224`, `embed_dim=384`)
- **Stage 1 FL Checkpoint**: `flame_round_055.pt` (Federated MAE, Round 55)

---

## 1. Within-Domain Performance: Shenzhen Test Set (132 Held-Out Images)

Support set sampled from Shenzhen training pool ($K=20$).

| Strategy / Metric | AUC | Accuracy | Balanced Acc | Sensitivity | Specificity | F1-Score |
|---|---|---|---|---|---|---|
| Direct Cosine (\tau=10.0) | 0.6682 | 0.6970 | 0.6967 | 0.7164 | 0.6769 | 0.7059 |
| Centered Cosine (\tau=10.0) | 0.7302 | 0.6970 | 0.6962 | 0.7463 | 0.6462 | 0.7143 |
| Stage 2 Adapted Head | 0.8643 | 0.7879 | 0.7874 | 0.8209 | 0.7538 | 0.7971 |
| Multi-Support Bagging (M=10 bags) | **0.7344** | **0.6818** | **0.6808** | **0.7463** | **0.6154** | **0.7042** |

---

## 2. Cross-Hospital Transfer Performance: Montgomery Test Set (138 Images)

Support set sampled from Shenzhen ($K=20$), tested across domain on Montgomery.

| Strategy / Metric | AUC | Accuracy | Balanced Acc | Sensitivity | Specificity | F1-Score |
|---|---|---|---|---|---|---|
| Direct Cosine (\tau=10.0) | 0.5162 | 0.4275 | 0.5039 | 0.9828 | 0.0250 | 0.5907 |
| Centered Cosine (\tau=10.0) | 0.5234 | 0.4275 | 0.5062 | 1.0000 | 0.0125 | 0.5949 |
| Stage 2 Adapted Head | 0.5282 | 0.4275 | 0.5062 | 1.0000 | 0.0125 | 0.5949 |
| Multi-Support Bagging (M=10 bags) | 0.5413 | 0.4203 | 0.5000 | 1.0000 | 0.0000 | 0.5918 |
