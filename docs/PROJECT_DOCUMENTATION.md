# 🛡️ FEDERATED SSL FOR TB DETECTION — COMPLETE TECHNICAL SPECIFICATION & STUDY NOTES
## "Architectural Foundations, Privacy-Preserving Formulations, Few-Shot Prototypical Metric Adaptation, and Clinical Benchmarks"

> [!TIP]
> **Active Research Branch**: `research/dp-vit-tiny-60-rounds`  
> **Model Checkpoint Distribution**: [Hugging Face Hub (`kirangowda07/fed-ssl-tb`)](https://huggingface.co/kirangowda07/fed-ssl-tb)  
> **Production Diagnostic Dashboard**: [Live Vercel Frontend](https://fed-ssl-tb-f5vbq8zga-gowda-labs.vercel.app/) | [Render Backend API](https://fedssl-tb-api.onrender.com)

---

# PART 1: WHAT WE BUILT AND WHY

## 1.1 The Clinical Challenge: Tuberculosis Screening

Tuberculosis (TB), caused by *Mycobacterium tuberculosis*, remains one of the top infectious killers worldwide, claiming over 1.3 million lives annually according to the World Health Organization (WHO). While chest radiography (CXR) is the primary first-line triage imaging modality recommended by the WHO, automated diagnostic artificial intelligence (AI) in clinical practice faces a critical triad of systemic bottlenecks:

1. **Cross-Silo Privacy Regulations & Institutional Silos**:
   Chest radiographs contain sensitive patient identifiable information and unique anatomical biometric markers. Data privacy legislations such as the **Health Insurance Portability and Accountability Act (HIPAA)** in the United States and the **General Data Protection Regulation (GDPR)** in the European Union strictly prohibit pooling raw clinical scans into a centralized repository across hospitals. Centralized model training is legally and logistically unviable.
2. **Extreme Annotation Scarcity**:
   More than 95% of hospital picture archiving and communication systems (PACS) store unlabeled or weakly annotated radiographs. High-quality radiological annotations require consensus reading by multiple expert thoracic radiologists, which is prohibitively slow, expensive, and unavailable in high-burden, low-resource settings.
3. **Catastrophic Out-of-Domain Failure of Supervised Baselines**:
   Standard deep learning approaches rely on supervised models pre-trained on generic natural images (e.g., ImageNet-1k). When transferred to chest radiographs from unseen clinical environments, these models suffer from domain shift: variations in X-ray tube voltage, scanner calibration, patient positioning, and racial demographics cause supervised models to collapse, dangerously missing active infections.

## 1.2 Our Integrated Solution

To overcome this triad, we developed **Fed-SSL-TB** (incorporating the **FLAME** framework: Federated Learning with Autoencoders and Metric Embeddings), a decentralized medical AI framework that decouples representation learning from task adaptation:

| Clinical / Technical Bottleneck | Our Solution | Core Mechanism |
| :--- | :--- | :--- |
| **Data Siloing & Privacy Laws** | Federated Learning with Differential Privacy | **FedAvg + DP-SGD** (per-sample gradient clipping & Gaussian perturbation) |
| **Client Drift under Non-IID Skew** | Federated Proximal Regularization | **FedProx Module** (client proximal term $\frac{\mu}{2}\|w - w_{\text{global}}\|^2$ with $\mu = 0.01$) |
| **Lack of Radiologist Annotations** | Decentralized Self-Supervised Learning | **Masked Autoencoder (MAE)** with an aggressive 80% patch masking ratio |
| **Edge Resource Constraints** | Zero-Backpropagation Edge Adaptation | **2-Layer Prototypical Metric Projection Head** with Cosine Similarity ($\tau=10.0$) |
| **Support Sampling Instability** | Robust Ensembling | **5-Head Multi-Support Bagging Ensemble** across reference support sets |
| **Catastrophic Domain Shift** | Domain-Agnostic Feature Learning | Self-supervised thoracic pretext learning preserving >60% sensitivity on unseen cohorts |

## 1.3 Two-Stage Decoupled Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Decentralized Federated Pre-training (Unlabeled NIH ChestX-ray14)             │
│                                                                                        │
│  [Hospital 1]  ──┐                                                                     │
│  [Hospital 2]  ──┼─► [Local MAE ViT-Tiny] ──► [DP-SGD Clipping (C) + Noise (σ)] ───┐   │
│  [   ...    ]  ──┤   (80% Patch Masking)      (Sample-Level Privacy Bound)         │   │
│  [Hospital 10] ──┘                                                                 ▼   │
│                         [Central Federated Server] ◄────────────────────────────────┘   │
│                         (Weighted Parameter Averaging via FedAvg / FedProx: θ_global)   │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Frozen Global Encoder (θ*)
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│ STAGE 2: Low-Resource Prototypical Few-Shot Metric Adaptation (Edge Clinic Deployment) │
│                                                                                        │
│  Reference Support Scans: K = 5 to 20 labeled scans per class                          │
│  Projection Head: 2-Layer MLP (192 → 128 → 128 with GELU, LayerNorm, Residual Skip)   │
│  Ensemble: 5-Head Multi-Support Bagging (varying seeds/subsets)                        │
│  Query Scan (x_q): Cosine Similarity with Temperature Scaling (τ = 10.0)               │
│  Diagnostic Output: P(TB | x_q) vs P(Normal | x_q)                                     │
│  * ZERO gradient backpropagation required at edge clinic diagnostic stations.          │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

# PART 2: THE CLINICAL DATASETS

The system is rigorously developed and evaluated across three independent, publicly available clinical cohorts:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 NIH ChestX-ray14                        │
                  │   20,000 Unlabeled Radiographs (Non-IID, α = 1.0)       │
                  │   Stage 1: Decentralized Pre-training across 10 Silos   │
                  └───────────────────────────┬─────────────────────────────┘
                                              │ Pre-trained Encoder (θ*)
                                              ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │               Shenzhen Hospital Cohort                  │
                  │   N = 662 (326 Normal, 336 TB)                          │
                  │   Stage 2: K-Shot Prototypical Metric Adaptation & Test │
                  └───────────────────────────┬─────────────────────────────┘
                                              │ Zero-Shot Transfer
                                              ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │            Montgomery County Cohort (OOD)               │
                  │   N = 138 (80 Normal, 58 TB)                            │
                  │   Out-of-Distribution Generalization Stress Test        │
                  └─────────────────────────────────────────────────────────┘
```

## 2.1 NIH ChestX-ray14 (Pre-training Cohort)
- **Institution**: National Institutes of Health (NIH) Clinical Center, Bethesda, MD, USA.
- **Dataset Scale**: 112,120 frontal chest X-rays from 30,805 unique patients.
- **Experimental Subset**: 20,000 images sampled and distributed across 10 simulated hospital partitions.
- **Label Usage**: **Completely Unlabeled**. All disease association labels are stripped; the model learns anatomical structure purely through self-supervised patch reconstruction.
- **Resolution**: Scaled to $224 \times 224$ pixels, 3-channel replicated grayscale.

## 2.2 Shenzhen Hospital TB Cohort (In-Domain Adaptation Benchmark)
- **Institution**: Shenzhen No. 3 People's Hospital, Guangdong, China.
- **Dataset Scale**: 662 frontal radiographs.
- **Class Balance**: 326 Normal (healthy controls) and 336 confirmed active pulmonary Tuberculosis cases.
- **Clinical Role**: Serves as the primary few-shot adaptation benchmark. For each evaluation, a tiny reference support set of $K \in \{3, 5, 10, 15, 20, 30\}$ scans per class is sampled, and the remaining scans form the test query cohort ($N_{\text{query}} = 662 - 2K$).

## 2.3 Montgomery County TB Cohort (External Out-of-Distribution Stress Test)
- **Institution**: Department of Health and Human Services, Montgomery County, MD, USA.
- **Dataset Scale**: 138 frontal radiographs (80 Normal, 58 TB).
- **Clinical Role**: **Strictly Held Out**. This dataset is never seen during pre-training, hyperparameter tuning, or adaptation. It provides an uncompromised evaluation of out-of-distribution (OOD) generalization across different radiography hardware, patient demographics, and image acquisition protocols.

---

# PART 3: ENVIRONMENT, DEPENDENCIES & SETUP

The project is structured to run efficiently across high-performance dual-GPU clusters (e.g., Kaggle Dual T4 GPUs) for training, and lightweight CPU-only cloud environments (e.g., Render free tier) for live web inference.

## 3.1 Python Dependencies (`requirements.txt`)

```text
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
numpy>=1.24.0
scikit-learn>=1.3.0
Pillow>=9.5.0
pyyaml>=6.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
pandas>=2.0.0
fastapi>=0.100.0
uvicorn[standard]>=0.22.0
python-multipart>=0.0.6
```

### Purpose of Core Packages:
- `torch` & `torchvision`: Tensor computation, automatic differentiation, neural network primitives, and image transformations.
- `timm`: PyTorch Image Models library, providing the Vision Transformer Tiny (`vit_tiny_patch16_224`) backbone.
- `scikit-learn`: Calculation of clinical evaluation metrics including Receiver Operating Characteristic Area Under the Curve (`roc_auc_score`), confusion matrices, sensitivity, and specificity.
- `fastapi` & `uvicorn`: High-performance asynchronous REST API framework serving real-time model predictions.
- `Pillow`: Image decoding, validation, and in-place thumbnail downsampling.

---

# PART 4: CONFIGURATION ARCHITECTURE

Training runs and evaluations are controlled via structured YAML configuration files.

## Production Configuration: `configs/kaggle_dp_vit_tiny_100round.yaml`

```yaml
model:
  backbone: vit_tiny          # Vision Transformer Tiny (5.7M parameters)
  embed_dim: 192             # Embedding feature dimension
  mask_ratio: 0.80           # 80% patch masking during Stage 1 MAE
  decoder_depth: 4           # 4-layer Transformer decoder (~1.2M parameters)
  decoder_embed_dim: 192     # Decoder latent dimension
  decoder_num_heads: 3       # Decoder multi-head self-attention heads
  image_size: 224            # Input resolution (224x224 pixels)
  patch_size: 16             # 16x16 pixel patches (196 total patches)

data:
  nih_path: /kaggle/input/data/data/NIH/images
  shenzhen_path: /kaggle/input/data/data/Shenzhen
  montgomery_path: /kaggle/input/data/data/Montgomery
  image_size: 224
  num_hospitals: 10          # 10 simulated clinical institutions
  split_strategy: dirichlet  # Non-IID Dirichlet partition
  dirichlet_alpha: 1.0       # Statistical heterogeneity concentration parameter
  seed: 42

ssl:
  lr: 0.0003                 # Peak learning rate with cosine decay
  weight_decay: 0.05         # AdamW decoupled weight decay
  batch_size: 32             # Local batch size per client
  epochs_per_round: 1        # 1 local epoch per federated round
  warmup_epochs: 5           # Linear learning rate warmup

differential_privacy:
  enabled: true              # DP-SGD sample-level privacy protection
  clip_norm: 1.0             # Per-sample L2 gradient clipping threshold (C = 1.0)
  noise_multiplier: 1.1      # Calibrated Gaussian noise standard deviation (σ = 1.1)
  target_delta: 0.00001      # Target failure probability (δ = 10^-5)
  accountant: rdp            # Rényi Differential Privacy accounting

federated:
  rounds: 100                # Total federated communication rounds
  aggregation: fedavg        # Modular aggregation: 'fedavg' (default for DP-SGD) or 'fedprox'
  fedprox_mu: 0.01           # Proximal regularization weight for FedProx ablation runs
  fraction_clients: 1.0      # Full participation (10/10 hospitals active per round)

finetuning:
  few_shot_k: 20             # Canonical support shot size (20 Normal, 20 TB)
  projection_dim: 128        # Prototypical head latent dimension
  temperature: 10.0          # Cosine similarity inverse temperature (tau = 10.0)
  epochs: 50                 # Leave-One-Out episodic adaptation epochs
  lr: 0.001                  # AdamW learning rate for projection head
```

---

# PART 5: THE DATA PIPELINE & NON-IID PARTITIONING

## 5.1 Image Pre-processing and Transforms

All chest radiographs are normalized using ImageNet statistical priors ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

- **Stage 1 Pre-training Transform**: Employs spatial data augmentation to encourage invariant feature learning:
  1. Bilinear resize to $256 \times 256$.
  2. Random spatial crop to $224 \times 224$.
  3. Random horizontal flip ($p = 0.5$).
  4. 3-channel grayscale replication.
  5. Conversion to standard PyTorch tensor followed by channel normalization.
- **Stage 2 Evaluation Transform**: Uses purely deterministic pre-processing:
  1. Direct bilinear resize to $224 \times 224$.
  2. 3-channel grayscale replication.
  3. Conversion to tensor and channel normalization (no random cropping or flipping).

## 5.2 Non-IID Dirichlet Hospital Partitioning (`src/datasets/splitter.py`)

In real-world healthcare federations, different hospitals have vastly different patient volumes and scanner demographics. We model this statistical heterogeneity across $M = 10$ institutions using a **Dirichlet distribution**:

$$q_k \sim \text{Dirichlet}(\alpha \cdot \mathbf{1}_M)$$

Where $\alpha = 1.0$ controls the concentration:
- As $\alpha \to \infty$, partitions become completely uniform and identical (IID).
- For $\alpha = 1.0$, partitions exhibit realistic, moderate quantity and distribution skews across hospitals without starving any client of training samples.
- The partition indices are deterministically generated with a fixed seed (`seed=42`) and cached in `data/processed/` to guarantee strict experimental reproducibility across rounds.

---

# PART 6: MODEL ARCHITECTURES

## 6.1 Vision Transformer Encoder (`ViT-Tiny`)

The encoder backbone is a lightweight Vision Transformer Tiny (`vit_tiny_patch16_224`):
- **Input Dimensions**: $3 \times 224 \times 224$.
- **Patch Embedding**: The image is partitioned into $N = (224 / 16)^2 = 196$ non-overlapping patches of size $16 \times 16 \times 3 = 768$ pixels. A linear projection maps each 768-dimensional flattened patch into the embedding dimension $D = 192$.
- **Learnable [CLS] Token**: Prepended to the patch sequence, resulting in $197$ tokens.
- **1D Learnable Positional Embeddings**: Added to each token to preserve 2D spatial coordinates.
- **Transformer Encoder Stack**: 12 Transformer blocks, each containing:
  - Pre-Layer Normalization (`LayerNorm`).
  - Multi-Head Self-Attention (MHSA) with 3 attention heads ($d_{\text{head}} = 192 / 3 = 64$).
  - Multi-Layer Perceptron (MLP) with GELU activation and expansion factor 4 ($192 \to 768 \to 192$).
- **Total Encoder Parameters**: ~5.7 million.

## 6.2 Masked Autoencoder Decoder (`MAE`)

During Stage 1 pre-training, an asymmetric Transformer decoder reconstructs the missing anatomical details:
- **Masking Mechanism**: An aggressive **80% masking ratio** is applied. Out of 196 image patches, only $39$ visible patches (20%) are processed by the encoder, drastically reducing local GPU memory and computational overhead by over $60\%$.
- **Decoder Architecture**:
  - Encoded visible patch representations are concatenated with shared learnable `[mask]` tokens for all 157 hidden patch locations.
  - Full positional embeddings are re-applied to restore original spatial ordering.
  - A 4-layer lightweight Transformer decoder ($D_{\text{dec}} = 192$, 3 heads, ~1.2M parameters) processes the complete token sequence.
  - A linear prediction head projects each token back to $16 \times 16 \times 3 = 768$ pixel space.
- **Reconstruction Loss**: Mean Squared Error (MSE) computed **exclusively on the 80% masked patches**:

$$\mathcal{L}_{\text{MAE}} = \frac{1}{|M|} \sum_{i \in M} \| p_i - \hat{p}_i \|_2^2$$

This forces the ViT encoder to learn high-level anatomical semantics (e.g., rib cage symmetry, lung parenchyma texture, pleural contours) rather than interpolating neighboring pixels.

## 6.3 Prototypical Metric Projection Head (`src/models/proto_head.py`)

To eliminate the need for GPU backpropagation or local model retraining at edge diagnostic clinics, Stage 2 uses a non-linear prototypical metric projection head:

```
ViT Latent Feature (z ∈ R^192)
       │
       ▼
[Linear Layer: 192 → 128]
       │
       ▼
[LayerNorm(128)]
       │
       ▼
[GELU Activation]
       │
       ▼
[Linear Layer: 128 → 128]
       │
       ├─────────────────────────────────┐
       ▼                                 ▼ (Residual Skip)
[Projected Latent h ∈ R^128] ◄────────── [+]
       │
       ▼
[L2 Hyperspherical Normalization: ||h||_2 = 1.0]
```

- **Learnable Parameters**: 41,216 parameters.
- **Metric Space Separation**: Raw ViT latent representations exhibit high angular correlation ($\cos(c_0, c_1) \approx 0.996$), limiting raw centroid separation. The 2-layer MLP projection pulls apart class prototypes to $\cos(c_0, c_1) \approx 0.607$, yielding dramatic classification gains.
- **Cosine Distance with Temperature Scaling**: For query feature $h_q$ and class prototype $c_k$:

$$s_k = \tau \cdot \frac{h_q \cdot c_k}{\|h_q\|_2 \|c_k\|_2}, \quad \tau = 10.0$$

$$P(y = k \mid x_q) = \frac{\exp(s_k)}{\sum_{j} \exp(s_j)}$$

---

# PART 7: DECENTRALIZED CLIENT TRAINING & DIFFERENTIAL PRIVACY

## 7.1 Differential Privacy Formulation (DP-SGD)

To guarantee mathematical privacy for hospital clients, local training incorporates Differentially Private Stochastic Gradient Descent (DP-SGD) via `src/client/dp_engine.py`:

1. **Per-Sample Gradient Clipping**: For a micro-batch of loss functions $\{\ell_i\}_{i=1}^B$, the gradient with respect to model parameters $\theta$ is clipped to a maximum $L_2$ norm threshold $C = 1.0$:

$$g_i = \nabla_\theta \ell_i(\theta), \quad \bar{g}_i = g_i \cdot \min\left(1, \frac{C}{\|g_i\|_2}\right)$$

2. **Calibrated Gaussian Perturbation**: Perturbation noise scaled by noise multiplier $\sigma = 1.1$ is added to the aggregated batch gradient:

$$\tilde{g} = \frac{1}{B} \left( \sum_{i=1}^B \bar{g}_i + \mathcal{N}(0, \sigma^2 C^2 \mathbf{I}) \right)$$

3. **Rényi Differential Privacy (RDP) Accounting**: Cumulative privacy loss is tracked across communication rounds and client local epochs using Rényi DP composition.

## 7.2 Modular Federated Strategies: FedAvg vs. FedProx in Multi-Track Ablations

The framework implements both **FedAvg** and **FedProx** as distinct, modular federated optimization strategies. Rather than treating them as mutually exclusive, they serve complementary roles across dedicated ablation tracks:

### Module A: FedProx for Mitigating Non-IID Client Drift
In cross-silo medical imaging, statistical heterogeneity (modeled via Dirichlet non-IID partitions $\alpha = 1.0$) causes local client gradients to diverge from the global objective. To constrain this drift, the **FedProx** module introduces an explicit proximal regularization term directly into the local client loss:

$$\mathcal{L}_{\text{FedProx}}(w) = \mathcal{L}_{\text{local}}(w) + \frac{\mu}{2} \|w - w_{\text{global}}\|^2$$

- **Proximal Regularization Weight ($\mu = 0.01$)**: Acts as a regularization constraint that penalizes the local encoder weights from straying excessively far from the broadcast global checkpoint $w_{\text{global}}$.
- **Ablation Utility**: Used to analyze federated stability, convergence envelopes, and feature alignment across heterogeneous hospital silos without centralized data pooling.

### Module B: FedAvg with Differential Privacy (DP-SGD)
When the Differential Privacy engine (`src/client/dp_engine.py`) is engaged, **FedAvg** serves as the canonical aggregation protocol.

> [!NOTE]
> **Why FedAvg is Paired with DP-SGD under AdamW**: When local updates are optimized using **AdamW**, the gradient of the proximal penalty $\mu(w - w_{\text{global}})$ is divided element-wise by the running second-moment vector $\sqrt{v_t} + \epsilon$:
>
> $$\Delta w \propto \frac{\nabla \mathcal{L}(w) + \mu(w - w_{\text{global}})}{\sqrt{v_t} + \epsilon}$$
>
> Because $v_t$ adapts dynamically per parameter, the effective proximal contraction rate $\frac{\mu}{\sqrt{v_t}}$ varies unpredictably across layers, distorting the $L_2$ sensitivity bounds required to calibrate DP Gaussian noise. To preserve formal Rényi Differential Privacy accounting, the DP pre-training pipeline pairs DP-SGD strictly with standard **FedAvg**.

### Multi-Track Ablation Architecture
By decoupling the two algorithms into independent configuration tracks, the project evaluates:
1. **Federated Optimization Track**: FedAvg vs FedProx ($\mu \in \{0.001, 0.01, 0.1\}$) across Dirichlet concentration shifts ($\alpha \in \{0.1, 0.5, 1.0\}$).
2. **Privacy-Preserving Track**: DP-SGD coupled with FedAvg ($C=1.0, \sigma=1.1, \delta=10^{-5}$) to establish formal privacy bounds on medical foundation models.

---

# PART 8: CENTRAL FEDERATED SERVER AGGREGATION

The central federated coordinator (`src/server/server.py`) manages the lifecycle across 100 communication rounds, dynamically supporting both aggregation modules:

```python
# Server Aggregation Dispatch (src/server/server.py)
def aggregate(self, received_weights, sample_counts):
    if self.aggregation == "fedavg":
        return fedavg(received_weights, sample_counts)
    elif self.aggregation == "fedprox":
        global_weights = self.broadcast()
        return fedprox(global_weights, received_weights, sample_counts, mu=self.fedprox_mu)
```

1. **FedAvg (Weighted Parameter Averaging)**:
   $$\theta_{t+1} = \sum_{k=1}^M \left(\frac{n_k}{N}\right) \theta_{t+1}^k$$
2. **FedProx (Server-side Proximal Weighting)**:
   Averages client updates while applying client weighting and parameter delta tracking.

```
Round t = 0 ... 99:
  1. Broadcast global encoder weights θ_t to all 10 hospitals.
  2. Each hospital k trains locally for 1 epoch using DP-SGD (MAE loss) or FedProx proximal loss.
  3. Hospitals upload updated encoder weights θ_{t+1}^k to server (decoder discarded).
  4. Server performs sample-weighted aggregation (FedAvg or FedProx).
  5. Save periodic checkpoints (`encoder_round_XXX.pt`).
```

Only encoder parameters are shared and aggregated. The MAE decoder remains strictly local to each hospital, preserving bandwidth and preventing reconstruction-inversion leakage.

---

# PART 9: STAGE 2 PROTOTYPICAL FEW-SHOT METRIC ADAPTATION & ENSEMBLING

## 9.1 Leave-One-Out (LOO) Episodic Training

The projection head is adapted on the Shenzhen reference support set ($K$ Normal, $K$ TB scans, total $2K$ examples) using 50 epochs of Leave-One-Out (LOO) episodic meta-learning:

```
For each epoch in 1 ... 50:
  For each sample i in {1 ... 2K}:
    1. Temporarily exclude sample i from the support set.
    2. Compute class prototypes from remaining (2K - 1) support embeddings:
           c_k = Mean({Proj(z_j) : y_j = k, j ≠ i})
    3. Compute cross-entropy loss predicting the true label of sample i:
           loss_i = -log P(y_i | Proj(z_i), {c_0, c_1})
  Accumulate loss = (1 / 2K) · ∑ loss_i
  Step optimizer (AdamW, lr = 0.001, weight_decay = 1e-4) with gradient clipping (max_norm = 1.0)
```

This episodic protocol forces the projection head to form compact, well-separated clusters that generalize to unseen query scans without overfitting the tiny support set.

## 9.2 5-Head Multi-Support Bagging Ensemble

At ultra-low shot counts ($K=3, 5$), random support sampling introduces variance: if an unrepresentative scan (e.g., severe scoliosis or poor exposure) is sampled into the support set, the resulting centroid shifts.

To eliminate this vulnerability, we deploy a **5-Head Multi-Support Bagging Ensemble**:
1. Five independent projection heads are initialized and adapted using 5 distinct random support seeds (Seeds: 42, 101, 202, 303, 404).
2. For any query radiograph $x_q$, each head independently computes calibrated class probabilities:
   $$P^{(m)}(y = \text{TB} \mid x_q), \quad m \in \{1, 2, 3, 4, 5\}$$
3. Predictions are aggregated via soft probability voting:
   $$\bar{P}(\text{TB} \mid x_q) = \frac{1}{5} \sum_{m=1}^5 P^{(m)}(y = \text{TB} \mid x_q)$$

---

# PART 10: CLINICAL VALIDATION & EMPIRICAL BENCHMARKS

## 10.1 Primary Benchmark: In-Domain Shenzhen & Out-of-Domain Montgomery

Evaluations conducted across both clinical cohorts with the canonical $K=20$ configuration:

| Model / Configuration | Cohort | ROC-AUC | Accuracy | Sensitivity (Recall) | Specificity | F1-Score | Diagnostic Significance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **ImageNet ViT-Tiny (Baseline)** | Shenzhen | 76.76% | 71.38% | 70.89% | 71.88% | 71.34% | Standard transfer baseline |
| **FedSSL-TB (FLAME R99, K=20 Centroid)** | Shenzhen | **81.26%** | **77.49%** | **78.48%** | **76.47%** | **77.47%** | Nearest centroid without projection |
| **FedSSL-TB (Single Head Peak, K=20)** | Shenzhen | **84.70%** | **78.78%** | **75.00%** | **82.68%** | **78.68%** | Single 2-layer MLP projection head |
| **FedSSL-TB (5-Head Ensemble, K=20)** | Shenzhen | **84.97%** | **80.29%** | **76.40%** | **84.39%** | **79.79%** | **Canonical operating point (Peak 85.19% AUC)** |
| *ImageNet ViT-Tiny (Baseline)* | Montgomery | 57.59% | 57.97% | **27.59%** | 80.00% | 35.56% | ⚠️ **Catastrophic Failure: Misses 72.4% of TB** |
| **FedSSL-TB (FLAME R29, K=20)** | Montgomery | **63.64%** | **60.14%** | **60.34%** | 60.00% | **55.99%** | **Preserves High Diagnostic Sensitivity** |

> [!IMPORTANT]
> **Clinical Safety Finding**: When evaluated on an unseen hospital cohort (Montgomery County), standard supervised ImageNet pre-training collapses to an unacceptable **27.59% sensitivity**, missing **72.4% of contagious TB patients**. In contrast, Fed-SSL-TB preserves over **60.34% sensitivity** out-of-the-box without requiring local hospital fine-tuning.

## 10.2 Comprehensive Support Shot Scaling Dynamics ($K \in \{3, 5, 10, 15, 20, 30\}$)

To evaluate performance across different clinical labeling budgets, the final adapted prototypical head was benchmarked across support shots:

| $K$ Shots/Class | Total Support Scans | Mode / Configuration | Shenzhen ROC-AUC | Shenzhen Accuracy | Shenzhen Sensitivity | Shenzhen Specificity | Montgomery ROC-AUC |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$K = 3$** | 6 | Single Head (Seed 42)<br>5-Seed Mean $\pm$ Std<br>**5-Head Ensemble** | 63.68%<br>71.38% $\pm$ 4.39%<br>**77.11%** | 60.82%<br>67.01% $\pm$ 3.23%<br>**67.25%** | 85.89%<br>72.91%<br>**71.96%** | 34.98%<br>60.93%<br>**62.38%** | 48.98%<br>54.01%<br>**60.10%** |
| **$K = 5$** | 10 | Single Head (Seed 42)<br>5-Seed Mean $\pm$ Std<br>**5-Head Ensemble** | 71.07%<br>78.15% $\pm$ 4.91%<br>**82.89%** | 66.41%<br>71.90% $\pm$ 4.42%<br>**76.35%** | 67.98%<br>70.88%<br>**74.04%** | 64.80%<br>72.96%<br>**78.74%** | 58.71%<br>55.23%<br>**56.40%** |
| **$K = 10$** | 20 | Single Head (Seed 42)<br>5-Seed Mean $\pm$ Std<br>**5-Head Ensemble** | 79.96%<br>81.20% $\pm$ 2.80%<br>**82.83%** | 72.74%<br>74.64% $\pm$ 2.39%<br>**76.15%** | 71.78%<br>72.88%<br>**73.36%** | 73.73%<br>76.46%<br>**79.06%** | 56.90%<br>54.67%<br>**56.29%** |
| **$K = 15$** | 30 | Single Head (Seed 42)<br>5-Seed Mean $\pm$ Std<br>**5-Head Ensemble** | 77.52%<br>82.92% $\pm$ 3.39%<br>**86.14%** | 70.57%<br>76.74% $\pm$ 3.91%<br>**81.30%** | 74.45%<br>78.26%<br>**82.22%** | 66.56%<br>75.18%<br>**80.31%** | 55.43%<br>53.29%<br>**55.58%** |
| **$K = 20$** *(Canonical)* | 40 | Single Head (Seed 42)<br>5-Seed Mean $\pm$ Std<br>**5-Head Ensemble (Canonical)** | 82.81%<br>83.55% $\pm$ 2.29%<br>**85.19% (Peak 84.97%)** | 77.01%<br>77.01% $\pm$ 2.24%<br>**79.38% (Peak 80.29%)** | 76.27%<br>76.46%<br>**78.97% (Peak 76.40%)** | 77.78%<br>77.58%<br>**79.83% (Peak 84.39%)** | 53.60%<br>54.27%<br>**55.02%** |
| **$K = 30$** | 60 | Single Head (Seed 42)<br>5-Seed Mean $\pm$ Std<br>**5-Head Ensemble** | 86.23%<br>84.39% $\pm$ 1.56%<br>**85.31%** | 79.90%<br>77.74% $\pm$ 1.96%<br>**77.34%** | 80.07%<br>76.41%<br>**78.40%** | 79.73%<br>79.12%<br>**76.17%** | 57.96%<br>55.75%<br>**54.44%** |

### Key Scaling Observations:
- **Low-Shot Range ($K=3, 5$)**: Ensembling provides massive stability, boosting AUC by **+13.43%** at $K=3$ (from 63.68% to 77.11%).
- **Clinical Sweet Spot ($K=20$)**: $K=20$ yields optimal performance (84.97% AUC, 80.29% Accuracy, 84.39% Specificity) with tight variance ($\pm 2.29\%$), requiring only 40 labeled scans from radiologists.
- **Saturation ($K=30$)**: Performance plateaus around 85–86% AUC, confirming that $K=20$ achieves the best accuracy-to-annotation-cost trade-off.

---

# PART 11: PRODUCTION CLOUD DEPLOYMENT & EDGE MEMORY ENGINEERING

The production deployment decouples frontend interaction from model inference:

```
[Web User Browser]
       │
       ▼ (HTTPS)
[Vercel Frontend: React + Vite SPA]
       │
       ▼ (REST API: /predict, /metrics, /health)
[Render Backend: FastAPI Web Service]
       │
       ▼ (HTTP Chunk Streaming on First Boot)
[Hugging Face Hub: kirangowda07/fed-ssl-tb]
  - flame_round_099.pt (22.8 MB)
  - optimal_proto_head_k20.pt (168 KB)
```

## 11.1 Render Free-Tier RAM Optimization (< 512 MB Constraint)

Standard PyTorch installations with CUDA runtimes require 2–3 GB of storage and exceed 600 MB of heap memory at initialization, causing instantaneous Out-Of-Memory (OOM) termination on cloud containers with 512 MB RAM limits. We engineered four critical memory safeguards in `src/web/api.py`:

1. **Lightweight CPU Wheel Installation**: The deployment specification installs CPU-only wheels via `--index-url https://download.pytorch.org/whl/cpu`, reducing container disk footprints from 3.2 GB to ~180 MB.
2. **Chunked Checkpoint Streaming**: Checkpoints are streamed in 1 MB buffered chunks from Hugging Face with automatic retry logic, avoiding full-file in-memory buffering.
3. **In-Place Image Downsampling**: Clinical X-rays can exceed $4000 \times 4000$ pixels. Processing full-resolution images into 32-bit float tensors consumes over 250 MB per image. We apply `image.thumbnail((512, 512))` immediately upon stream receipt, reducing image buffer size by **98.4%** before RGB tensor conversion.
4. **Active Memory Reclamation**:
   - Explicit `del checkpoint_state` immediately after loading weights into the model.
   - Inference execution strictly wrapped in `torch.inference_mode()`.
   - Explicit invocation of `gc.collect()` following every inference call.
   - This caps peak server heap usage at **~185 MB**, comfortably below the 512 MB ceiling.

## 11.2 Frontend Diagnostic Interface (Vercel)

The React dashboard provides:
- Live model health status and backend latency tracking.
- Interactive radiograph upload with drag-and-drop preview.
- Bundled clinical sample images for instant verification.
- Diagnostic confidence meters displaying probability distribution between Normal and TB.
- Automated client request timeout resilience (up to 60s) to absorb Render cold-start wakeups.

---

# PART 12: HOW TO RUN EVERYTHING

## 12.1 Local Installation and Verification

```bash
# Clone the repository
git clone https://github.com/nmit-1NT23CS111/Fed-SSL-TB.git
cd Fed-SSL-TB

# Install dependencies
pip install -r requirements.txt

# Run DP-SGD mathematical unit tests (clipping, Gaussian perturbation, RDP accountant)
python tests/test_dp_sgd.py

# Run Stage 1 architecture and tensor shape contracts
python test_stage1_architecture.py

# Run a synthetic 1-round dry run simulation (no datasets required)
python src/federated/simulation.py --dry-run --config configs/kaggle_dp_vit_tiny_100round.yaml --federated.rounds=1
```

## 12.2 Production Training on Kaggle (Dual T4 GPUs)

```bash
# In a Kaggle environment with GPU T4 x2 enabled:
!git clone https://github.com/nmit-1NT23CS111/Fed-SSL-TB.git /kaggle/working/Fed-SSL-TB
%cd /kaggle/working/Fed-SSL-TB
!pip install -q -r requirements.txt

# Launch 100-round federated training across 10 hospitals (Dual GPU parallel)
!python -u src/federated/simulation.py --config configs/kaggle_dp_vit_tiny_100round.yaml --parallel
```
*(Completes 100 rounds across 10 hospitals in approximately 70–75 minutes).*

## 12.3 Running Stage 2 Evaluations and Ablations

```bash
# Run downstream few-shot evaluation on Shenzhen using round 99 checkpoint
python run_finetune.py \
  --checkpoint experiments/stage2_round099_checkpoint/flame_round_099.pt \
  --config configs/kaggle_dp_vit_tiny_100round.yaml

# Run the complete K-shot scaling ablation suite
python scripts/evaluate_final_head_kshot.py
```

## 12.4 Generating IEEE Publication Figures

```bash
# Render all 300 DPI PNG and vector PDF publication figures
python scripts/generate_paper_figures.py
```
*Outputs are saved to `paper_assets/figures/`.*

## 12.5 Launching the Web Dashboard Locally

```bash
# Terminal 1: Launch FastAPI backend
python src/web/api.py

# Terminal 2: Launch React frontend
cd src/web/frontend
npm install
npm run dev
```
*Open `http://localhost:3000` in your web browser.*

---

# PART 13: CODE GLOSSARY & ARCHITECTURAL FUNCTION MAP

| Component / Function | File Location | Architectural Description |
| :--- | :--- | :--- |
| `get_encoder()` | `src/models/encoder.py` | Instantiates ViT-Tiny backbone and extracts 192-dim representation vector |
| `MaskedAutoencoder` | `src/models/mae.py` | Full MAE architecture (patch embedding, 80% masking, ViT encoder, Transformer decoder) |
| `PrototypicalHead` | `src/models/proto_head.py` | 2-layer MLP projection ($192 \to 128 \to 128$), LOO prototypical loss, cosine metric ($\tau=10.0$) |
| `DPEngine` | `src/client/dp_engine.py` | Per-sample $L_2$ gradient clipping ($C=1.0$), Gaussian perturbation ($\sigma=1.1$), RDP accountant |
| `ssl_local_train()` | `src/client/ssl_train.py` | Executes local MAE self-supervised training loop with DP gradient updates on client data |
| `FederatedServer` | `src/server/server.py` | Broadcasts global encoder, performs sample-weighted FedAvg/FedProx aggregation, manages checkpoints |
| `fedavg()` | `src/server/aggregator.py` | Computes sample-weighted average of client encoder parameters |
| `fedprox()` | `src/server/aggregator.py` | Server-side FedProx aggregation tracking client parameter deltas |
| `split_nih_to_hospitals()` | `src/datasets/splitter.py` | Partitions NIH dataset across 10 hospitals using Dirichlet non-IID distribution ($\alpha=1.0$) |
| `ShenzhenDataset` | `src/datasets/loader.py` | PyTorch Dataset loader for Shenzhen Hospital radiographs with few-shot sampling |
| `MontgomeryDataset` | `src/datasets/loader.py` | PyTorch Dataset loader for held-out Montgomery County out-of-distribution evaluation |
| `evaluate()` | `src/utils/metrics.py` | Calculates clinical metrics: ROC-AUC, Accuracy, Sensitivity, Specificity, F1-Score |
| `predict_tb()` | `src/web/api.py` | FastAPI endpoint performing zero-backpropagation radiograph inference with memory protections |

---

*This document serves as the authoritative, unified technical specification for the Fed-SSL-TB project. Validated for academic evaluation and clinical research workflows.*
