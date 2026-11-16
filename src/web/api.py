"""FastAPI deployment using the canonical FLAME-inspired few-shot classifier."""

import gc
import io
import json
import os
import sys
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------------
# Auto-download model weights from HuggingFace Hub on first startup
# ---------------------------------------------------------------------------
def _download_models():
    hf_user = os.environ.get("HF_USERNAME", "kirangowda07")
    hf_repo  = os.environ.get("HF_REPO",     "fed-ssl-tb")
    base_url = f"https://huggingface.co/{hf_user}/{hf_repo}/resolve/main"

    _ROOT_HERE = Path(__file__).resolve().parents[2]
    models = {
        _ROOT_HERE / "experiments" / "main_experiment" / "checkpoints" / "flame_round_099.pt":
            f"{base_url}/flame_round_099.pt",
        _ROOT_HERE / "experiments" / "models" / "optimal_proto_head_k20.pt":
            f"{base_url}/optimal_proto_head_k20.pt",
    }
    for local_path, url in models.items():
        if local_path.exists() and local_path.stat().st_size > 1024:
            print(f"[Startup] ✓ {local_path.name} already present", flush=True)
            continue
        print(f"[Startup] Downloading {local_path.name} from HuggingFace ({url})...", flush=True)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as response, open(local_path, "wb") as out_file:
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
            print(f"[Startup] ✓ {local_path.name} downloaded ({local_path.stat().st_size // (1024*1024)} MB)", flush=True)
        except Exception as e:
            print(f"[Startup] WARNING: Failed to download {local_path.name}: {e}", flush=True)

_download_models()
# ---------------------------------------------------------------------------

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.client.local_train import _sample_kshot, finetune_local
from src.datasets.loader import ShenzhenDataset, get_eval_transform
from src.models.encoder import get_encoder
from src.models.proto_head import PrototypicalHead
from src.utils.config import load_config

app = FastAPI(title="FedSSL TB Detection API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
config = load_config(str(_ROOT / "configs" / "default.yaml"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = get_eval_transform(config.data.image_size)
encoder = None
classifier = None
model_error = None


def _rooted_path(configured_path):
    path = Path(configured_path)
    return path if path.is_absolute() else _ROOT / path


def _find_checkpoint():
    search_dirs = [
        _rooted_path(config.logging.checkpoint_dir),
        _ROOT / "experiments" / "stage2_round099_checkpoint",
        _ROOT / "experiments" / "main_experiment" / "checkpoints",
        _ROOT / "experiments" / "results" / "30_round" / "checkpoints",
        _ROOT / "experiments" / "checkpoints" / "main",
        _ROOT / "experiments" / "checkpoints",
    ]
    candidates = []
    for checkpoint_dir in search_dirs:
        if not checkpoint_dir.exists():
            continue
        candidates.extend([checkpoint_dir / "best_flame.pt", checkpoint_dir / "best_encoder.pt"])
        def _get_round(p):
            try:
                return int(p.stem.split("_")[-1])
            except Exception:
                return 0
        candidates.extend(sorted(checkpoint_dir.glob("flame_round_*.pt"), key=_get_round, reverse=True))
        candidates.extend(sorted(checkpoint_dir.glob("encoder_round_*.pt"), key=_get_round, reverse=True))
    return next((path for path in candidates if path.exists() and path.stat().st_size > 1000), None)


def _load_fl_models():
    """Load both encoder and prototypical head with auto-detection from checkpoint."""
    checkpoint = _find_checkpoint()
    checkpoint_state = {}
    if checkpoint is not None:
        try:
            checkpoint_state = torch.load(checkpoint, map_location=device, weights_only=False)
            print(f"[Startup] Found checkpoint: {checkpoint.name} ({checkpoint.stat().st_size // 1024} KB)")
        except Exception as e:
            print(f"[Startup] WARNING: Failed to read {checkpoint}: {e}")

    # Auto-detect backbone and embed_dim from checkpoint config if present
    ckpt_cfg = checkpoint_state.get("config")
    backbone = config.model.backbone
    embed_dim = config.model.embed_dim
    if ckpt_cfg:
        if isinstance(ckpt_cfg, dict):
            backbone = ckpt_cfg.get("backbone", backbone)
            embed_dim = ckpt_cfg.get("embed_dim", embed_dim)
        elif hasattr(ckpt_cfg, "model"):
            backbone = getattr(ckpt_cfg.model, "backbone", backbone)
            embed_dim = getattr(ckpt_cfg.model, "embed_dim", embed_dim)

    # 1. Initialize Encoder
    enc = get_encoder(backbone, embed_dim)
    if "encoder_state_dict" in checkpoint_state:
        enc.load_state_dict(checkpoint_state["encoder_state_dict"])
        print(f"[Startup] FL encoder loaded from {checkpoint.name} ({backbone}, {embed_dim}-dim)")
    else:
        print(f"[Startup] WARNING: No encoder weights in checkpoint; using initialized encoder ({backbone}, {embed_dim}-dim)")
    enc.to(device).eval()

    # 2. Initialize and Load Prototypical Head
    optimal_proto_path = _ROOT / "experiments" / "models" / "optimal_proto_head_k20.pt"
    if optimal_proto_path.exists():
        try:
            proto_ckpt = torch.load(optimal_proto_path, map_location=device, weights_only=False)
            proj_dim = int(proto_ckpt.get("projection_dim", 128))
            proto_head = PrototypicalHead(embed_dim, 2, proj_dim).to(device)
            proto_head.load_state_dict(proto_ckpt["proto_head_state_dict"])
            proto_head.prototypes = proto_ckpt["prototypes"].to(device)
            proto_head._prototypes_computed = True
            proto_head.eval()
            k_shot = proto_ckpt.get("k_shot", 20)
            cos_sim = float(torch.nn.functional.cosine_similarity(proto_head.prototypes[0:1], proto_head.prototypes[1:2]).item())
            print(f"[Startup] Loaded trained optimal prototypical head from {optimal_proto_path.name} (K={k_shot}, Prototype cos_sim={cos_sim:.4f})")
            del checkpoint_state, proto_ckpt
            gc.collect()
            return enc, proto_head
        except Exception as e:
            print(f"[Startup] WARNING: Could not load {optimal_proto_path.name}: {e}")

    # Fallback: Initialize and compute prototypes from support set
    proj_dim = int(getattr(config.finetuning, "projection_dim", 128))
    proto_head = PrototypicalHead(embed_dim, 2, proj_dim).to(device)
    if "proto_head_state_dict" in checkpoint_state:
        proto_head.load_state_dict(checkpoint_state["proto_head_state_dict"], strict=False)
        print(f"[Startup] Restored proto head weights from {checkpoint.name} ({embed_dim} -> {proj_dim})")

    dataset = ShenzhenDataset(str(_rooted_path(config.data.shenzhen_path)), transform=transform, image_size=config.data.image_size)
    if len(dataset) == 0:
        raise RuntimeError("Shenzhen dataset is empty or unavailable")

    k_shot = int(getattr(config.finetuning, "few_shot_k", 20))
    support_indices, _ = _sample_kshot(torch.as_tensor(dataset.get_labels()), k_shot, seed=int(config.finetuning.seed))
    support_loader = DataLoader(torch.utils.data.Subset(dataset, support_indices.tolist()), batch_size=len(support_indices), shuffle=False)
    support_images, support_labels = next(iter(support_loader))

    with torch.no_grad():
        proto_head.compute_prototypes(enc(support_images.to(device)), support_labels.to(device))
    proto_head.eval()

    norms = [round(float(x), 4) for x in proto_head.prototypes.norm(dim=1).tolist()]
    print(f"[Startup] Prototypical head active ({k_shot}-shot support | Prototype norms: {norms})")
    return enc, proto_head


@app.on_event("startup")
async def load_models():
    global encoder, classifier, model_error
    try:
        encoder, classifier = _load_fl_models()
        model_error = None
        print("[Startup] Ready (canonical prototypical few-shot mode)")
    except Exception as error:
        model_error = str(error)
        encoder = None
        classifier = None
        print(f"[Startup] ERROR: {model_error}")


@app.get("/")
def root():
    return {"service": "FedSSL TB Detection API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health_check():
    return {"status": "healthy" if model_error is None else "degraded", "device": str(device), "model": config.model.backbone, "classifier": "prototypical_few_shot", "model_ready": classifier is not None, "model_error": model_error}


@app.get("/metrics")
def get_training_metrics():
    search_paths = [
        _rooted_path(config.logging.log_dir) / "training_log.json",
        _ROOT / "experiments" / "main_experiment" / "logs" / "training_log.json",
        _ROOT / "experiments" / "results" / "30_round" / "logs" / "training_log.json",
        _ROOT / "experiments" / "results" / "30_round" / "logs" / "main_stage1" / "training_log.json",
        _ROOT / "experiments" / "logs" / "main" / "training_log.json",
    ]
    for log_path in search_paths:
        if log_path.exists():
            with log_path.open("r") as handle:
                return json.load(handle)
    return []


@app.get("/status")
def get_training_status():
    metrics = get_training_metrics()
    latest_round = max((entry.get("round", -1) for entry in metrics), default=-1)
    total_rounds = max(int(config.federated.rounds), latest_round + 1)
    checkpoint = _find_checkpoint()
    completed = latest_round + 1
    return {
        "state": "COMPLETED" if completed >= total_rounds else "IDLE",
        "current_round": completed,
        "completed_rounds": completed,
        "total_rounds": total_rounds,
        "hospital_count": int(config.data.num_hospitals),
        "aggregation": config.federated.aggregation,
        "latest_round": metrics[-1] if metrics else None,
        "checkpoint": {"available": bool(checkpoint), "name": checkpoint.name if checkpoint else None},
    }


def _count_images(path):
    if not path.exists():
        return None
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in suffixes)


@app.get("/metadata")
def get_metadata():
    return {
        "model": {
            "backbone": config.model.backbone,
            "embedding_dimension": config.model.embed_dim,
            "projection_dimension": getattr(config.finetuning, "projection_dim", None),
            "classifier": "prototypical_few_shot",
        },
        "training": {
            "hospitals": config.data.num_hospitals,
            "rounds": config.federated.rounds,
            "aggregation": config.federated.aggregation,
            "few_shot_per_class": config.finetuning.few_shot_k,
        },
        "datasets": {
            "NIH": {"purpose": "Federated SSL pretraining", "image_count": _count_images(_rooted_path(config.data.nih_path))},
            "Shenzhen": {"purpose": "Few-shot adaptation", "image_count": _count_images(_rooted_path(config.data.shenzhen_path))},
            "Montgomery": {"purpose": "Held-out final evaluation", "image_count": _count_images(_rooted_path(config.data.montgomery_path))},
        },
    }


@app.get("/evaluation")
def get_evaluation_benchmarks():
    return {
        "peak_condition": {
            "model": "ViT-Tiny (192-dim, patch16, MAE Pre-trained)",
            "checkpoint": "flame_round_099.pt",
            "rounds": 100,
            "support_set": "Shenzhen Hospital (20-shot per class, 40 images total)",
            "adaptation_epochs": 50,
            "learning_rate": 0.001,
            "optimizer": "AdamW",
            "loss_function": "Prototypical Cross-Entropy (Temperature tau=10.0)",
            "ensemble": "5-Head Multi-Support Bagging (Seeds 42, 101, 202, 303, 404)",
        },
        "peak_metrics": {
            "auc": 0.8497,
            "accuracy": 0.8029,
            "specificity": 0.8439,
            "sensitivity": 0.7640,
            "balanced_accuracy": 0.8040,
            "f1": 0.7979,
        },
        "single_head_peak": {
            "auc": 0.8470,
            "accuracy": 0.7878,
            "specificity": 0.8268,
            "sensitivity": 0.7500,
            "balanced_accuracy": 0.7884,
            "f1": 0.7783,
        },
        "canonical_nearest_centroid": {
            "auc": 0.8126,
            "accuracy": 0.7749,
            "specificity": 0.7647,
            "sensitivity": 0.7848,
            "balanced_accuracy": 0.7748,
            "f1": 0.7747,
        },
        "imagenet_baseline": {
            "auc": 0.7676,
            "accuracy": 0.7138,
            "specificity": 0.7188,
            "sensitivity": 0.7089,
            "balanced_accuracy": 0.7138,
            "f1": 0.7134,
        },
        "out_of_distribution_montgomery": {
            "fedssl_r29_auc": 0.6364,
            "fedssl_sensitivity": 0.6034,
            "imagenet_auc": 0.5759,
            "imagenet_sensitivity": 0.2759,
            "clinical_insight": "ImageNet misses 72.4% of TB cases on out-of-domain scans; FedSSL preserves recall (>60%)."
        }
    }


@app.post("/predict")
async def predict_tb(file: UploadFile = File(...)):
    if encoder is None or classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="The uploaded image is empty")
        image = Image.open(io.BytesIO(contents))
        # Prevent OOM on 512MB RAM: downsample large clinical scans (3000x3000+) before conversion
        image.thumbnail((512, 512), Image.Resampling.BILINEAR)
        image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            feat = encoder(tensor)
            _, probabilities = classifier.predict(feat)
        prob_tb = float(probabilities[0, 1])
        prob_normal = float(probabilities[0, 0])
        del contents, image, tensor, feat
        gc.collect()
        return {
            "filename": file.filename,
            "prediction": "TB Positive" if prob_tb >= 0.5 else "Normal",
            "confidence": round(max(prob_tb, prob_normal), 4),
            "tb_probability": round(prob_tb, 4),
            "normal_probability": round(prob_normal, 4),
        }
    except HTTPException:
        raise
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image") from error
    except Exception as error:
        print(f"[Prediction] failed: {error}")
        raise HTTPException(status_code=500, detail="Prediction failed on the server") from error


@app.get("/samples")
def get_sample_images():
    return [
        {"id": "normal_1", "label": "Sample Healthy (Normal)", "type": "normal", "filename": "CHNCXR_0001_0.png"},
        {"id": "normal_2", "label": "Sample Healthy #2", "type": "normal", "filename": "CHNCXR_0002_0.png"},
        {"id": "tb_1", "label": "Sample TB Positive", "type": "tb", "filename": "CHNCXR_0327_1.png"},
        {"id": "tb_2", "label": "Sample TB Positive #2", "type": "tb", "filename": "CHNCXR_0328_1.png"},
    ]


@app.get("/samples/{filename}")
def get_sample_file(filename: str):
    bundled = Path(__file__).resolve().parent / "sample_images" / filename
    if bundled.exists():
        return FileResponse(str(bundled), media_type="image/png")
    search_dir = _ROOT / "data" / "raw" / "Shenzhen" / "images" / "images"
    target = search_dir / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail="Sample not found")
    return FileResponse(str(target), media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
