# FedSSL-TB: IEEE Publication Asset Package

This directory contains high-resolution (300 DPI), publication-grade figures, LaTeX snippets, and empirical data tables formatted specifically for IEEE transactions and conference proceedings (e.g., IEEE Access, IEEE EMBC, IEEE BHI, IEEE ISBI).

---

## 📁 Asset Manifest

| Figure File | IEEE Title / Caption | Key Metrics & Data Ground Truth | LaTeX Label |
| :--- | :--- | :--- | :--- |
| [`figures/fig1_ieee_system_architecture.png`](figures/fig1_ieee_system_architecture.png) <br> [`figures/fig1_ieee_system_architecture.pdf`](figures/fig1_ieee_system_architecture.pdf) | **System Architecture**: Two-stage decentralized framework combining FedProx-MAE and Few-Shot Prototypical Metric Adaptation. | Patch $16\times16$, $75\%$ masking ratio, ViT-Tiny (5.7M params), FedProx $\mu=0.01$, Prototypical Centroids ($K=5, 20$). | `\label{fig:architecture}` |
| [`figures/fig2_federated_loss_convergence.png`](figures/fig2_federated_loss_convergence.png) | **Decentralized Pre-training Convergence**: 100-round MAE reconstruction loss trajectory with inter-client heterogeneity envelope. | Round 1 (0.658) $\to$ Round 100 (0.133), $-79.8\%$ loss drop across 10 Dirichlet non-IID clients ($\alpha=0.5$). | `\label{fig:loss_convergence}` |
| [`figures/fig3_roc_curves_evaluation.png`](figures/fig3_roc_curves_evaluation.png) | **Dual-Cohort Receiver Operating Characteristic (ROC)**: In-domain query (Shenzhen) vs. external cross-site stress test (Montgomery). | Shenzhen: FedSSL R99 ($\mathbf{0.8126}$) vs ImageNet ($0.7676$). Montgomery: FedSSL R29 ($\mathbf{0.6364}$) vs ImageNet ($0.5759$). | `\label{fig:roc_curves}` |
| [`figures/fig4_fedssl_vs_imagenet_benchmark.png`](figures/fig4_fedssl_vs_imagenet_benchmark.png) | **FedSSL-TB vs. ImageNet Pre-trained ViT-Tiny Baseline**: Grouped metric comparison across both clinical cohorts. | Shenzhen: $+4.50\%$ AUC over ImageNet. Montgomery: ImageNet misses $72.4\%$ of TB cases ($27.6\%$ sensitivity); FedSSL preserves high sensitivity. | `\label{fig:benchmark_bars}` |
| [`figures/fig5_few_shot_k_ablation.png`](figures/fig5_few_shot_k_ablation.png) | **Few-Shot Support Size Ablation ($K \in \{5, 10, 15, 20, 30, 50\}$)**: Metric scaling and clinical sensitivity/specificity trade-off. | Optimal clinical balance at $K=20$: $81.26\%$ AUC, $77.49\%$ Acc, $78.48\%$ Sens, $76.47\%$ Spec. | `\label{fig:k_shot_ablation}` |
| [`figures/fig6_clinical_confusion_matrices.png`](figures/fig6_clinical_confusion_matrices.png) | **Clinical Confusion Matrices**: Discrete triage breakdown for Shenzhen ($N=622$) and Montgomery ($N=138$) test cohorts. | Shenzhen: $248$ TP, $234$ TN, $72$ FP, $68$ FN. Montgomery: $35$ TP, $31$ TN, $49$ FP, $23$ FN. | `\label{fig:confusion_matrices}` |

---

## 📝 IEEE LaTeX Figure Snippets

### 1. System Architecture (Double Column Figure)
```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_ieee_system_architecture.pdf}
\caption{\textbf{Overview of the FedSSL-TB Framework.} 
(a) \textit{Stage 1: Decentralized Federated Pre-training.} Ten simulated hospital clients collaboratively pretrain a Vision Transformer (ViT-Tiny) via Masked Autoencoding (MAE) with $75\%$ patch masking on unlabeled radiographs. Local optimization incorporates the FedProx proximal penalty ($\mu=0.01$) to stabilize training under non-IID quantity skews, aggregating model weights on a central server without raw patient data transfer.
(b) \textit{Stage 2: Low-Resource Metric Adaptation.} Edge diagnostic centers leverage the frozen global encoder ($\theta^*$) and evaluate query radiographs using few-shot prototypical nearest-centroid classification ($K=5$ to $20$ reference shots), achieving inference without gradient backpropagation.}
\label{fig:architecture}
\end{figure*}
```

### 2. Federated Convergence (Single Column Figure)
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig2_federated_loss_convergence.png}
\caption{\textbf{Federated Self-Supervised Pre-training Convergence.}
(a) Mean global MAE reconstruction loss (MSE) across 100 communication rounds. Shaded envelope represents client-level variation ($\pm 1\sigma$).
(b) Non-IID local client dataset sizes and final localized reconstruction loss at round 100.}
\label{fig:loss_convergence}
\end{figure}
```

### 3. Dual-Cohort ROC Curves (Double Column Figure)
```latex
\begin{figure*}[t]
\centering
\includegraphics[width=0.95\textwidth]{figures/fig3_roc_curves_evaluation.png}
\caption{\textbf{Receiver Operating Characteristic (ROC) Trajectories.}
(a) In-domain query performance on the Shenzhen Hospital dataset comparing FedSSL-TB ($K=20$, Round 99: $\text{AUC}=0.8126$) against the ImageNet-pretrained baseline ($0.7676$).
(b) Out-of-domain cross-site evaluation on the Montgomery County dataset characterizing institutional domain shift.}
\label{fig:roc_curves}
\end{figure*}
```

### 4. Benchmark Comparison (Double Column Figure)
```latex
\begin{figure*}[t]
\centering
\includegraphics[width=0.98\textwidth]{figures/fig4_fedssl_vs_imagenet_benchmark.png}
\caption{\textbf{Comparative Benchmark: FedSSL-TB vs. ImageNet-Pretrained ViT-Tiny.}
(a) On Shenzhen Hospital, FedSSL-TB trained from scratch outperforms the ImageNet baseline across all primary metrics.
(b) On Montgomery County, ImageNet features exhibit a dangerous screening failure mode, missing $72.4\%$ of active TB cases ($27.59\%$ sensitivity), whereas FedSSL preserves high detection recall.}
\label{fig:benchmark_bars}
\end{figure*}
```

---

## 🔬 Experimental Ground Truth References

All figures are automatically derived from the following audited repository files:
- `experiments/main_experiment/logs/training_log.json` (100-round MAE loss logs)
- `experiments/ablation_studies/few_shot_k/results.json` ($K$-shot ablation evaluations)
- `experiments/ablation_studies/round49_vs_round99/round099_results.json` (Round 99 evaluation)
- `experiments/logs/imagenet_proto_k5.json` (ImageNet $K=5$ evaluation)
- `experiments/results/30_round/logs/stage2_30round_k20/imagenet_proto_k20.json` (ImageNet $K=20$ evaluation)

To re-render all figures at any time:
```bash
python scripts/generate_paper_figures.py
```
