# FedSSL: Federated Self-Supervised Learning for Tuberculosis Detection

FedSSL is a research project exploring federated self-supervised representation learning for chest X-ray images, followed by few-shot adaptation for tuberculosis (TB) classification. It uses NIH ChestX-ray14 images for self-supervised training, Shenzhen data for few-shot support/adaptation, and Montgomery data for held-out cross-dataset evaluation.

> **Research prototype:** This repository is intended for experimentation and academic evaluation, not clinical diagnosis. Federated simulation does not by itself guarantee privacy; encryption and secure aggregation are not implemented here.

## Project highlights

- **Federated self-supervised training:** Simulates training across multiple hospital partitions without pooling the raw training images into one central dataset.
- **Masked Autoencoder (MAE):** Learns image representations by reconstructing masked image patches.
- **FLAME-inspired / prototype-based few-shot adaptation:** Uses Shenzhen support examples to form class prototypes for Normal and TB classification.
- **Non-IID data support:** Hospital partitions can use the configured split strategy and concentration parameter.
- **Evaluation and dashboard:** Includes experiment logs/metrics and a React + FastAPI interface for viewing recorded training metrics and running inference when the required data and model artifacts are available.

## Datasets

Download the datasets separately; raw images are not included in this repository. Configure their paths in the YAML configuration used for your run.

| Dataset | Role |
| --- | --- |
| NIH ChestX-ray14 | Self-supervised representation learning across simulated hospital partitions |
| Shenzhen TB | Few-shot support/adaptation and query evaluation |
| Montgomery TB | Held-out cross-dataset evaluation |

See [Project Documentation](docs/PROJECT_DOCUMENTATION.md) for the expected data layout and project details.

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/Kirangowda0715/Fed-SSL-TB-Run.git
cd Fed-SSL-TB-Run
pip install -r requirements.txt
```

For NVIDIA GPU support, install a PyTorch build compatible with your CUDA driver. For example, the PyTorch CUDA 12.1 wheels can be installed with:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Configure the datasets

Place the downloaded datasets in the locations configured in your YAML file. Review the paths and options in `configs/kaggle.yaml` or `configs/default.yaml` before running any commands.

To prepare NIH hospital partitions using the Kaggle configuration:

```bash
python -c "from src.utils.config import load_config; from src.datasets.loader import NIHDataset; from src.datasets.splitter import split_nih_to_hospitals; cfg = load_config('configs/kaggle.yaml'); ds = NIHDataset(cfg.data.nih_path, limit=cfg.ssl.limit_samples); split_nih_to_hospitals(ds, num_hospitals=cfg.data.num_hospitals, strategy=cfg.data.split_strategy, alpha=cfg.data.split_alpha, save_dir=cfg.data.processed_dir, seed=cfg.finetuning.seed)"
```

### 3. Run federated training

For a standard configured run:

```bash
python src/federated/simulation.py --config configs/default.yaml
```

For a Kaggle environment where two GPUs are visible, the repository supports the parallel option:

```bash
python src/federated/simulation.py --config configs/kaggle.yaml --parallel
```

Check your selected configuration before starting: it determines dataset paths, hospital count, partition strategy, training settings, and few-shot parameters. The default few-shot setting described by the project is 5-shot per class (five Normal and five TB support images); verify the active configuration for the run you are reproducing.

## Run the dashboard

Start the backend in one terminal:

```bash
python src/web/api.py
```

Start the frontend in another terminal:

```bash
cd src/web/frontend
npm install
npm run dev
```

Open `http://localhost:3000` in your browser. The frontend uses `VITE_API_URL`, which defaults to `http://localhost:8000`.

### API overview

- `GET /metrics` — recorded training-round metrics and any evaluation metrics present in the log.
- `GET /status` — status derived from the available log and configuration.
- `GET /metadata` — model, few-shot, and filesystem-derived dataset metadata.
- `POST /predict` — accepts an uploaded image in the multipart `file` field and returns prediction/confidence fields when the required model and data are available.

The dashboard displays recorded information; it should not be interpreted as evidence that a training process is currently running unless a live process is actually active.

## Experiment records

Experiment logs and metric files are organized under `experiments/`. Check the specific experiment folder and its configuration before comparing results. Training loss is not the same as classification accuracy, AUC, sensitivity, or specificity; use the relevant Stage 2 evaluation metrics for classification claims. Do not treat missing evaluation output as a measured result.

Model checkpoint files and datasets may be excluded from Git by `.gitignore`; a clean clone may therefore require you to obtain those artifacts separately.

## Documentation

- [Project documentation](docs/PROJECT_DOCUMENTATION.md)
- [Project study notes (Word)](docs/Federated_SSL_Project_Study_Notes.docx)

---

*Academic major-project research prototype.*
