# Ablation 2: Frozen vs Unfrozen Encoder

## Objective
Evaluate whether updating the pretrained encoder during few-shot Stage 2
adaptation affects in-domain and cross-dataset classification performance.

## Protocol
- Initial checkpoint: FLAME Round 99
- Few-shot support: K=20 per class (40 total images)
- Random seed: 42
- Projection dimension: 128
- Adaptation epochs: 100
- Frozen condition: encoder parameters fixed
- Unfrozen condition: encoder and prototype-head parameters trainable

## Results
| Dataset | Metric | Frozen (%) | Unfrozen (%) |
|---|---|---:|---:|
| Shenzhen | AUC | 81.26 | 79.89 |
| Shenzhen | Accuracy | 77.49 | 72.19 |
| Shenzhen | Sensitivity | 78.48 | 76.90 |
| Shenzhen | Specificity | 76.47 | 67.32 |
| Shenzhen | F1-score | 77.99 | 73.75 |
| Shenzhen | Balanced accuracy | 77.48 | 72.11 |
| Montgomery | AUC | 51.29 | 54.61 |
| Montgomery | Accuracy | 47.83 | 52.90 |
| Montgomery | Sensitivity | 60.34 | 51.72 |
| Montgomery | Specificity | 38.75 | 53.75 |
| Montgomery | F1-score | 49.30 | 48.00 |
| Montgomery | Balanced accuracy | 49.55 | 52.74 |

## Interpretation
Freezing the encoder performs better on Shenzhen across all reported metrics.
Unfreezing improves Montgomery AUC, accuracy, specificity, and balanced
accuracy, but reduces sensitivity and F1-score. Both Montgomery AUC values
indicate weak cross-dataset discrimination.

## Limitations
This comparison uses one random seed and one support-set size. The results
do not establish statistical significance or prove that either strategy
generalizes better in general. Further repeated-seed evaluation is needed.
