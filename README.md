# Federated Self-Supervised Learning (FedSSL) for TB Detection

> [!TIP]
> **[📥 Download Documentation as Word (.docx)](docs/Federated_SSL_Project_Study_Notes.docx)**

> **📹 [Watch the 2-Minute Demo Video Here](#)** *(Replace this with your YouTube/Loom link!)*

A Federated Learning system for detecting Tuberculosis (TB) using self-supervised Masked Autoencoder pre-training and FLAME-inspired adaptation on sparse Shenzhen labels.

---

## 💻 Platform Showcase

*(Once you take screenshots of your UI, save them in the `docs/images/` folder so they appear here!)*

### Real-Time Training Dashboard
Monitor global loss convergence across 5 federated hospitals and track live medical metrics (Sensitivity, Specificity, F1) seamlessly.
![Dashboard UI](docs/images/dashboard.png)

### Live TB Analysis & Inference
Upload unlabelled X-rays for instant diagnostic inference powered by our fine-tuned Prototypical Network, displaying absolute confidence scores.
![TB Analysis UI](docs/images/analysis.png)

### Animated Federated Simulation
Visualize the Privacy-First FedProx protocol in action. Zero raw patient data leaves the hospital boundaries; only mathematically encrypted weights are shared to the central cloud.
![Federated Map UI](docs/images/federated.png)

### System Architecture
A deep dive into the engineering constraints, hardware optimization (RTX 2050), and mathematical frameworks driving the AI.
![Requirements UI](docs/images/requirements.png)

---

## 🌟 Architectural Highlights
- **Federated Learning (FedProx)**: Privacy-preserving training across 5 simulated hospitals. Utilizes FedProx to handle the extreme non-IID (unbalanced) data distributions common in real-world clinics.
- **Vision Transformer (ViT-Tiny)**: Replaced legacy CNNs with a state-of-the-art Transformer backbone. The architecture ensures the federated simulation runs extremely efficiently on consumer-grade GPUs (e.g., RTX 2050 4GB).
- **Masked Autoencoder (MAE)**: Self-supervised learning from unlabeled X-rays. The model learns fundamental human anatomy by reconstructing masked patches of chest scans.
- **Few-Shot TB Detection**: A projection MLP maps encoder features into prototype space; class prototypes are means of exactly `k` Shenzhen Normal and `k` Shenzhen TB support samples.
- **Live Dashboard**: A fully interactive React/FastAPI dashboard to monitor training and perform real-time TB inference.

## 📊 Datasets (Massive Scale)
This project utilizes a subset of three major chest X-ray datasets. Note that due to their large size, the raw images are excluded from Git tracking.

| Dataset | Purpose | Images Used | Split Strategy |
| :--- | :--- | :--- | :--- |
| **NIH ChestX-ray14** | Federated SSL representation learning (unlabeled) | Configured limit | Configured split |
| **Shenzhen TB** | Sparse labeled support and adaptation/query-training data | Dataset-dependent | Deterministic seeded sampling |
| **Montgomery TB** | Held-out cross-hospital final evaluation | Dataset-dependent | Never used during adaptation |

> [!IMPORTANT]
> You must download these datasets manually and place them in the `data/raw/` directory structure as defined in the [Documentation](PROJECT_DOCUMENTATION.md).

## 🚀 Getting Started

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/Kirangowda0715/Federated-SSL
cd Federated-SSL

# Install dependencies
pip install -r requirements.txt

# For GPU support (NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Prepare Data
Ensure your data is placed in `data/raw/` and then run the data splitter to simulate the 5 hospitals:
```bash
python -c "from src.datasets.loader import NIHDataset; from src.datasets.splitter import split_nih_to_hospitals; from src.utils.config import load_config; cfg = load_config(); ds = NIHDataset(cfg.data.nih_path, limit=20000); split_nih_to_hospitals(ds, strategy='non_iid', alpha=2.0)"
```

### 3. Federated Training
To start the simulation:
```bash
python src/federated/simulation.py --config configs/default.yaml
```

On a Kaggle notebook with two visible GPUs, enable per-hospital parallel training:
```bash
!python src/federated/simulation.py --config configs/kaggle.yaml --parallel
```
The simulation assigns at most one concurrent hospital worker to each visible GPU
and prints the device assignment at startup and for each hospital.

The default few-shot setting is 5-shot per class, meaning 5 Normal plus 5 TB
support images. Remaining Shenzhen images are adaptation/query-training data;
Montgomery is reserved for final evaluation. The encoder is trainable by
default and the projection dimension is configured in `finetuning`.

### 4. Start the Live Dashboard
In separate terminal windows, start the backend and frontend:
```bash
# Terminal 2 (Backend)
python src/web/api.py

# Terminal 3 (Frontend)
cd src/web/frontend
npm run dev
```
Navigate to `http://localhost:3000` to interact with the federated metrics.

### Frontend API contract

The frontend reads `VITE_API_URL` (default `http://localhost:8000`) through one
API service. The FastAPI service exposes:

- `GET /metrics` — the persisted round log, including `mean_mae_loss`,
  `mean_proto_loss`, `mean_total_loss`, hospital losses, sample counts, and any
  recorded `eval_metrics`.
- `GET /status` — observable state derived from the log and active config,
  including completed/total rounds, configured hospitals, aggregation strategy,
  and checkpoint availability. An incomplete log is reported as `IDLE`; it is
  not presented as live `RUNNING` without a process signal.
- `GET /metadata` — model, few-shot, and filesystem-derived dataset metadata.
- `POST /predict` — multipart field `file`; returns `prediction`,
  `confidence`, `tb_probability`, and `normal_probability` when inference is
  available.

Shenzhen is the few-shot adaptation/support dataset and Montgomery is the
held-out final evaluation dataset. “5-shot per class” means five Normal plus
five TB support images.

## 📖 Learn More
For a deep dive into the architecture, federated strategies, and physics of the model, see:
👉 **[PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md)**

---
*Developed as a Major Final Year Project.*
