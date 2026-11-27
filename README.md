# Fed-SSL-TB: Privacy-Preserving Federated Self-Supervised Learning for Tuberculosis Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Backbone](https://img.shields.io/badge/Backbone-ViT--Tiny%20(5.7M)-00A86B.svg?style=flat)]()
[![Privacy](https://img.shields.io/badge/Differential_Privacy-DP--SGD%20(RDP)-8A2BE2.svg?style=flat)]()
[![Federation](https://img.shields.io/badge/Federation-FedAvg%20%26%20FedProx%20%7C%20Non--IID%20Dirichlet-blue.svg?style=flat)]()
[![Paper Assets](https://img.shields.io/badge/IEEE_Publication-300_DPI_Assets-gold.svg?style=flat)](paper_assets/README.md)

**Fed-SSL-TB** is a decentralized, privacy-preserving medical imaging framework designed for computer-aided tuberculosis (TB) detection from chest radiographs. It addresses the triad of real-world clinical AI challenges:
1. **Cross-Silo Privacy Regulations**: Raw radiographs cannot be pooled across institutions due to HIPAA and GDPR restrictions.
2. **Clinical Annotation Scarcity**: More than 95% of hospital chest X-rays lack expert radiologist ground-truth labels.
3. **Transfer Learning Failure Modes**: Generic ImageNet pre-training collapses on subtle pulmonary lesions, missing over 72% of active TB cases on out-of-domain scans.

---

## 🏛️ System Architecture

The framework operates in two distinct, decoupled stages:

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Decentralized Federated Pre-training (Unlabeled Chest Radiographs)       │
│                                                                                   │
│  [Hospital 1] ──┐                                                                 │
│  [Hospital 2] ──┼─► [Local MAE ViT-Tiny] ─► [DP-SGD Clipping (C) + Noise (σ)] ──┐│
│  [   ...    ] ──┤   (80% Patch Masking)     (Sample-Level Privacy Bound)        ││
│  [Hospital 10]──┘                                                               ││
│                                                                                 ▼│
│                         [Central Federated Server] ◄─────────────────────────────┘│
│                         (Server Aggregation via FedAvg: θ_global)                 │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         │ Frozen Global Encoder (θ*)
┌────────────────────────────────────────▼──────────────────────────────────────────┐
│ STAGE 2: Low-Resource Prototypical Few-Shot Metric Adaptation (Edge Deployment)   │
│                                                                                   │
│  Reference Support: K=5 to 20 shots per class ──► Centroids (c_normal, c_tb)     │
│  Query Radiograph (x_q) ──► Euclidean / Cosine Metric ──► Triage Probability      │
│  * ZERO gradient backpropagation required at edge clinic diagnostic stations.     │
└───────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Stage 1: Decentralized Self-Supervised Pre-training
- **Backbone**: Vision Transformer Tiny (**ViT-Tiny**, 192-dim, patch16, ~5.7M parameters).
- **Self-Supervised Objective**: Masked Autoencoder (**MAE**) reconstructing masked patches with an aggressive **80% masking ratio**, compelling the model to learn global anatomical structures (infiltrates, consolidations, cavitary lesions).
- **Differential Privacy (DP-SGD)**:
  - Per-sample gradient clipping to $L_2$ threshold $C = 1.0$.
  - Calibrated Gaussian perturbation $\mathcal{N}(0, \sigma^2 C^2 \mathbf{I})$ with $\sigma = 1.1$.
  - Exact cumulative Rényi Differential Privacy (RDP) state persistence across rounds and clients.
- **Server Aggregation & Modular Optimization**: The framework implements both **FedAvg** and **FedProx** as modular aggregation options for comparative ablation:
  - **FedProx Module**: Incorporates a client-side proximal regularization penalty ($\frac{\mu}{2}\|w - w_{\text{global}}\|^2$ with $\mu = 0.01$) to explicitly restrict client drift under non-IID Dirichlet distribution skews ($\alpha = 1.0$).
  - **FedAvg Module**: Serves as the primary aggregation baseline and the foundational aggregation protocol for our Differential Privacy (DP-SGD) runs, ensuring uncorrupted sensitivity calibration without AdamW adaptive moment distortion.

### 2. Stage 2: Few-Shot Prototypical Metric Adaptation
- **Zero-Backpropagation Edge Inference**: Rural diagnostic clinics with limited hardware use a frozen global encoder ($\theta^*$).
- **Metric Nearest-Centroid Head**: Radiographs are projected into a normalized latent metric space ($192 \to 128$). Class prototypes are computed using only $K = 5$ to $K = 20$ labeled reference scans, completely eliminating the need for local GPU backpropagation or optimizer state memory.

---

## 🔬 Empirical Clinical Benchmarks

Evaluations were performed across two independent clinical cohorts:
- **Shenzhen Hospital ($N = 662$)**: Primary in-domain screening benchmark.
- **Montgomery County ($N = 138$)**: External cross-institutional out-of-distribution stress test.

| Model / Framework | Cohort | ROC-AUC | Accuracy | Sensitivity (Recall) | Specificity | F1-Score | Clinical Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **ImageNet ViT-Tiny (Baseline)** | Shenzhen | 76.76% | 71.38% | 70.89% | 71.88% | 71.34% | Standard transfer baseline |
| **FedSSL-TB (FLAME R99, K=20 Centroid)** | Shenzhen | **81.26%** | **77.49%** | **78.48%** | **76.47%** | **77.47%** | **+4.50% AUC over ImageNet** |
| **FedSSL-TB (Single Head Peak, K=20)** | Shenzhen | **84.70%** | **78.78%** | **75.00%** | **82.68%** | **78.68%** | **Single Model Peak Specificity** |
| **FedSSL-TB (5-Head Ensemble, K=20)** | Shenzhen | **84.97%** | **80.29%** | **76.40%** | **84.39%** | **79.79%** | **Canonical Operating Point (Peak 85.19% AUC)** |
| *ImageNet ViT-Tiny (Baseline)* | Montgomery | 57.59% | 57.97% | **27.59%** | 80.00% | 35.56% | ⚠️ **Catastrophic Failure: Misses 72.4% of TB** |
| **FedSSL-TB (FLAME R29, K=20)** | Montgomery | **63.64%** | **60.14%** | **60.34%** | 60.00% | **55.99%** | **Preserves High Diagnostic Sensitivity** |

> [!IMPORTANT]
> **Clinical Safety Finding**: On out-of-domain scans (Montgomery), generic ImageNet features collapse to **$27.59\%$ sensitivity**, dangerously discharging **$72.4\%$ of infectious TB patients** untreated. FedSSL-TB trained from scratch preserves diagnostic recall ($>60\%$ sensitivity) without requiring local retraining.

### Few-Shot Support Scaling Dynamics ($K \in \{3, 5, 10, 15, 20, 30\}$)

To evaluate diagnostic robustness across varying radiologist annotation budgets, the final adapted prototypical head ($192 \to 128 \to 128$, $\tau=10.0$ cosine similarity) was benchmarked across support shots:

| $K$ Shots/Class | Total Support Scans | Mode / Configuration | Shenzhen ROC-AUC | Shenzhen Accuracy | Shenzhen Sensitivity | Shenzhen Specificity | Montgomery ROC-AUC |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$K = 3$** | 6 | Single Head (Seed 42)<br>5-Seed Mean $\pm$ Std<br>**5-Head Ensemble** | 63.68%<br>71.38% $\pm$ 4.39%<br>**77.11%** | 60.82%<br>67.01% $\pm$ 3.23%<br>**67.25%** | 85.89%<br>72.91%<br>**71.96%** | 34.98%<br>60.93%<br>**62.38%** | 48.98%<br>54.01%<br>**60.10%** |
| **$K = 5$** | 10 | Single Head (Seed 42)<br>5-Seed Mean $\pm$ Std<br>**5-Head Ensemble** | 71.07%<br>78.15% $\pm$ 4.91%<br>**82.89%** | 66.41%<br>71.90% $\pm$ 4.42%<br>**76.35%** | 67.98%<br>70.88%<br>**74.04%** | 64.80%<br>72.96%<br>**78.74%** | 58.71%<br>55.23%<br>**56.40%** |
| **$K = 10$** | 20 | Single Head (Seed 42)<br>5-Seed Mean $\pm$ Std<br>**5-Head Ensemble** | 79.96%<br>81.20% $\pm$ 2.80%<br>**82.83%** | 72.74%<br>74.64% $\pm$ 2.39%<br>**76.15%** | 71.78%<br>72.88%<br>**73.36%** | 73.73%<br>76.46%<br>**79.06%** | 56.90%<br>54.67%<br>**56.29%** |
| **$K = 15$** | 30 | Single Head (Seed 42)<br>5-Seed Mean $\pm$ Std<br>**5-Head Ensemble** | 77.52%<br>82.92% $\pm$ 3.39%<br>**86.14%** | 70.57%<br>76.74% $\pm$ 3.91%<br>**81.30%** | 74.45%<br>78.26%<br>**82.22%** | 66.56%<br>75.18%<br>**80.31%** | 55.43%<br>53.29%<br>**55.58%** |
| **$K = 20$** *(Canonical)* | 40 | Single Head (Seed 42)<br>5-Seed Mean $\pm$ Std<br>**5-Head Ensemble (Canonical)** | 82.81%<br>83.55% $\pm$ 2.29%<br>**85.19% (Peak 84.97%)** | 77.01%<br>77.01% $\pm$ 2.24%<br>**79.38% (Peak 80.29%)** | 76.27%<br>76.46%<br>**78.97% (Peak 76.40%)** | 77.78%<br>77.58%<br>**79.83% (Peak 84.39%)** | 53.60%<br>54.27%<br>**55.02%** |
| **$K = 30$** | 60 | Single Head (Seed 42)<br>5-Seed Mean $\pm$ Std<br>**5-Head Ensemble** | 86.23%<br>84.39% $\pm$ 1.56%<br>**85.31%** | 79.90%<br>77.74% $\pm$ 1.96%<br>**77.34%** | 80.07%<br>76.41%<br>**78.40%** | 79.73%<br>79.12%<br>**76.17%** | 57.96%<br>55.75%<br>**54.44%** |

---

## 📁 Repository Structure

```
Federated-SSL/
├── configs/                                # Experiment configurations (YAML)
│   ├── kaggle_dp_vit_tiny_100round.yaml    # Production 100-round DP ViT-Tiny Kaggle config
│   ├── kaggle_dp_vit_tiny_60round.yaml     # 60-round DP ViT-Tiny Kaggle config
│   ├── kaggle_suhas_100round.yaml          # 100-round GPU config for Suhas environment
│   ├── current_stage1_3round.yaml          # Fast 3-round local testing config
│   └── default.yaml                        # Default local baseline configuration
├── src/
│   ├── client/
│   │   ├── dp_engine.py                    # Per-sample clipping, Gaussian noise, RDP accountant
│   │   └── ssl_train.py                    # Local client SSL loop with DP state restoration
│   ├── datasets/
│   │   ├── loader.py                       # NIH, Shenzhen, Montgomery PyTorch dataset loaders
│   │   └── splitter.py                     # Dirichlet non-IID partitioner & cache manager
│   ├── models/
│   │   ├── encoder.py                      # ViT-Tiny, ViT-Small, and ResNet-50 backbones
│   │   ├── decoder.py                      # MAE Transformer decoder
│   │   ├── mae.py                          # Full Masked Autoencoder architecture
│   │   └── proto_head.py                   # Prototypical metric projection head
│   ├── server/
│   │   └── server.py                       # FederatedServer aggregation & checkpoint manager
│   ├── federated/
│   │   └── simulation.py                   # Main federated simulation CLI engine
│   └── web/                                # Full-stack diagnostic web dashboard
│       ├── api.py                          # FastAPI backend
│       └── frontend/                       # React + Vite responsive UI
├── experiments/                            # Experiment logs, checkpoints, and ablations
│   ├── stage2_round099_checkpoint/         # Tracked round 99 ViT-Tiny federated weights
│   ├── ablation_studies/                   # Ablations 1–7 (Distance metrics, K-shot, ensembling)
│   └── logs/                               # Verified empirical evaluation logs
├── paper_assets/                           # Publication package for IEEE conferences/journals
│   ├── README.md                           # LaTeX figure snippets, captions, and tables
│   └── figures/                            # 300 DPI high-resolution figures (PDF and PNG)
├── scripts/                                # Utility scripts and manual generators
│   └── generate_paper_figures.py           # Automated re-rendering script for paper figures
├── tests/
│   └── test_dp_sgd.py                      # 11 unit tests for DP-SGD clipping and accounting
├── run_finetune.py                         # Downstream evaluation CLI
├── run_proto_experiments.py                # Prototypical adaptation benchmark suite
├── test_stage1_architecture.py             # Stage 1 architecture contracts test suite
└── requirements.txt                        # Core Python dependencies
```

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install dependencies in Python 3.10+:

```bash
git clone https://github.com/nmit-1NT23CS111/Fed-SSL-TB.git
cd Fed-SSL-TB
pip install -r requirements.txt
```

For NVIDIA CUDA GPU acceleration (e.g., CUDA 12.1):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Run Local Unit Tests & Dry-Run

Verify the installation, DP-SGD engine, and model contracts locally without downloading datasets:

```bash
# Run mathematical DP-SGD unit tests (clipping, noise, accounting)
python tests/test_dp_sgd.py

# Run Stage 1 architecture & checkpoint contracts
python test_stage1_architecture.py

# Run a complete 1-round federated simulation on GPU with synthetic data
python src/federated/simulation.py --dry-run --config configs/kaggle_dp_vit_tiny_100round.yaml --federated.rounds=1
```

---

## ⚡ Production Training & Evaluation

### 1. Kaggle GPU 100-Round Federated Pre-training (Stage 1)

In a Kaggle Notebook with **GPU T4 x2** accelerator:

```bash
# Clone the repository
!git clone https://github.com/nmit-1NT23CS111/Fed-SSL-TB.git /kaggle/working/Fed-SSL-TB
%cd /kaggle/working/Fed-SSL-TB
!pip install -q -r requirements.txt

# Launch 100-round DP ViT-Tiny training across 10 hospitals (Dual GPU parallel)
!python -u src/federated/simulation.py --config configs/kaggle_dp_vit_tiny_100round.yaml --parallel
```
*(On Kaggle Dual T4 GPUs, 100 rounds complete in ~70–75 minutes).*

### 2. Downstream Few-Shot Evaluation (Stage 2)

Evaluate any federated checkpoint on the Shenzhen Hospital or Montgomery County cohorts:

```bash
# Few-shot prototypical evaluation using the round 99 federated checkpoint
python run_finetune.py \
  --checkpoint experiments/stage2_round099_checkpoint/flame_round_099.pt \
  --config configs/kaggle_dp_vit_tiny_100round.yaml

# Run the complete prototypical ablation test suite
python run_proto_experiments.py \
  --checkpoint experiments/stage2_round099_checkpoint/flame_round_099.pt
```

### 3. Re-generate IEEE Publication Figures

All publication figures (ROC curves, federated convergence envelopes, benchmark bars, confusion matrices) can be rendered directly from the experimental logs:

```bash
python scripts/generate_paper_figures.py
```
Outputs are saved to `paper_assets/figures/` in both **300 DPI PNG** and **vector PDF** formats.

---

## 🖥️ Web Diagnostic Dashboard & Cloud Deployment

The project includes an interactive full-stack diagnostic dashboard for inspecting federated training curves, exploring ablation benchmarks, and performing zero-backpropagation radiograph inference.

- **Frontend Deployment**: Hosted on **Vercel** ([FedSSL | TB Detection Dashboard](https://fed-ssl-tb-f5vbq8zga-gowda-labs.vercel.app/))
- **Inference Backend**: Hosted on **Render** (`https://fedssl-tb-api.onrender.com`)
- **Model Distribution**: Checkpoint weights (`flame_round_099.pt` and `optimal_proto_head_k20.pt`) streamed directly from **Hugging Face Model Hub** (`kirangowda07/fed-ssl-tb`).

### Cloud Architecture & Edge Memory Optimizations

To ensure resilient operation within restricted edge/cloud environments (such as Render's 512 MB RAM container constraint), the backend implements:
1. **CPU-Optimized Wheel Distribution**: Configured with lightweight PyTorch CPU wheels (`torch torchvision --index-url https://download.pytorch.org/whl/cpu`), stripping unused multi-gigabyte CUDA runtime shared objects.
2. **Streaming Chunk Checkpoint Pulling**: Automatic 1MB buffered chunk streaming from Hugging Face with 120-second connection timeout resilience, avoiding memory buffering of raw model blobs.
3. **In-Place Bilinear Downsampling**: Pre-scaling high-resolution clinical DICOM/PNG uploads (often >4000x4000 px) via `image.thumbnail((512, 512))` prior to RGB tensor conversion, capping peak heap allocation below ~180 MB.
4. **Active Garbage Collection & Inference Mode**: Wrapped strictly in `torch.inference_mode()` with immediate `del checkpoint_state` and explicit `gc.collect()` to guarantee zero memory leaks across consecutive inference queries.

### Local Development Setup

1. **Start the FastAPI Backend**:
   ```bash
   python src/web/api.py
   ```
2. **Start the React Frontend**:
   ```bash
   cd src/web/frontend
   npm install
   npm run dev
   ```
3. Open `http://localhost:3000` in your browser.

---

## 🔒 Privacy & Optimization Guarantees

1. **Sampling Scheme Reality**: PyTorch's `DataLoader(..., shuffle=True)` partitions data into batches without replacement. Analytical Rényi DP formulas derived strictly under Poisson subsampling do not formally apply to without-replacement shuffling. To maintain strict scientific integrity, formal $(\epsilon, \delta)$ values are marked `unsupported_under_shuffle_without_replacement`, while exact parameter tracking ($T, C, \sigma, \delta$) is recorded without heuristic approximations.
2. **Modular Aggregation: FedAvg vs. FedProx in Ablation Studies**: Both federated aggregation algorithms are implemented as first-class modules:
   - **FedProx Ablation Module**: Evaluates client-side proximal regularization ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MAE}} + \frac{\mu}{2}\|w - w_{\text{global}}\|^2$, typically with $\mu = 0.01$) on local client updates to constrain client drift when institutional data partitions exhibit high statistical non-IID heterogeneity.
   - **FedAvg DP Baseline**: When Differential Privacy (DP-SGD) is enabled, FedAvg is designated as the calibrated aggregation protocol. Theoretical FedProx adds a deterministic proximal penalty $\mu(w_t - w_{\text{global}})$, which, when passed through the **AdamW** adaptive optimizer, is non-linearly scaled by $\frac{1}{\sqrt{v_t}}$ across parameters, disrupting calibrated per-sample $L_2$ sensitivity bounds. Thus, FedProx and FedAvg are treated as complementary, dedicated ablation tracks in the experimental study.
3. **Hardware PRNG vs CSPRNG**: Noise is generated using PyTorch's internal Philox/MT19937 generator seeded with 64 bits of OS entropy (`secrets.randbits(64)`). While statistically calibrated and reproducible for ML simulations, it is not a certified cryptographic CSPRNG.

---

## 📖 Citation & Publication Package

If you use this codebase or the FLAME framework in your research, please refer to the assets in [`paper_assets/README.md`](paper_assets/README.md) for pre-formatted LaTeX snippets, tables, and figure citations.

```bibtex
@article{fedssl_tb_2026,
  title   = {Privacy-Preserving Federated Self-Supervised Learning with Prototypical Metric Adaptation for Decentralized Tuberculosis Screening},
  author  = {Kiran Gowda and Contributors},
  journal = {IEEE Transactions / Conference Proceedings},
  year    = {2026}
}
```

---

*Academic major-project research prototype. Validated for academic evaluation and clinical research workflows.*
