"""FastAPI deployment using the canonical FLAME-inspired few-shot classifier."""

import io
import json
import sys
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
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
    checkpoint_dir = _rooted_path(config.logging.checkpoint_dir)
    candidates = [checkpoint_dir / "best_flame.pt", checkpoint_dir / "best_encoder.pt"]
    candidates.extend(sorted(checkpoint_dir.glob("flame_round_*.pt"), reverse=True))
    candidates.extend(sorted(checkpoint_dir.glob("encoder_round_*.pt"), reverse=True))
    return next((path for path in candidates if path.exists()), None)


def _load_fl_encoder():
    model = get_encoder(config.model.backbone, config.model.embed_dim)
    checkpoint_dir = _rooted_path(config.logging.checkpoint_dir)
    checkpoint = _find_checkpoint()
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state.get("encoder_state_dict", state))
        print(f"[Startup] FL encoder loaded: {checkpoint.name}")
    else:
        print(f"[Startup] WARNING: no FL checkpoint found in {checkpoint_dir}; using initialized encoder")
    return model.to(device)


@app.on_event("startup")
async def load_models():
    global encoder, classifier, model_error
    try:
        encoder = _load_fl_encoder()
        dataset = ShenzhenDataset(str(_rooted_path(config.data.shenzhen_path)), transform=transform, image_size=config.data.image_size)
        if len(dataset) == 0:
            raise RuntimeError("Shenzhen dataset is empty or unavailable")
        labels = dataset.get_labels()
        required = int(config.finetuning.few_shot_k)
        if min(labels.count(0), labels.count(1)) < required:
            raise RuntimeError(f"Shenzhen dataset needs at least {required} Normal and {required} TB images")
        loader = DataLoader(dataset, batch_size=config.ssl.batch_size, shuffle=False)
        checkpoint = _find_checkpoint()
        checkpoint_state = torch.load(checkpoint, map_location=device, weights_only=False) if checkpoint else {}
        head_state = checkpoint_state.get("proto_head_state_dict")
        if head_state:
            classifier = PrototypicalHead(config.model.embed_dim, 2, int(config.finetuning.projection_dim)).to(device)
            classifier.load_state_dict(head_state, strict=False)
            support_indices, _ = _sample_kshot(torch.as_tensor(dataset.get_labels()), int(config.finetuning.few_shot_k), seed=int(config.finetuning.seed))
            support_loader = DataLoader(torch.utils.data.Subset(dataset, support_indices.tolist()), batch_size=len(support_indices), shuffle=False)
            support_images, support_labels = next(iter(support_loader))
            with torch.no_grad():
                classifier.compute_prototypes(encoder(support_images.to(device)), support_labels.to(device))
            print(f"[Startup] Restored saved prototype head from {checkpoint.name}")
        else:
            classifier, _ = finetune_local(0, encoder, loader, config, device)
        classifier.eval()
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
    log_path = _rooted_path(config.logging.log_dir) / "training_log.json"
    if not log_path.exists():
        return []
    with log_path.open("r") as handle:
        return json.load(handle)


@app.get("/status")
def get_training_status():
    metrics = get_training_metrics()
    latest_round = max((entry.get("round", -1) for entry in metrics), default=-1)
    total_rounds = int(config.federated.rounds)
    checkpoint_dir = _rooted_path(config.logging.checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("*.pt")) if checkpoint_dir.exists() else []
    completed = latest_round + 1
    return {
        "state": "COMPLETED" if completed >= total_rounds else "IDLE",
        "current_round": completed,
        "completed_rounds": completed,
        "total_rounds": total_rounds,
        "hospital_count": int(config.data.num_hospitals),
        "aggregation": config.federated.aggregation,
        "latest_round": metrics[-1] if metrics else None,
        "checkpoint": {"available": bool(checkpoints), "name": checkpoints[-1].name if checkpoints else None},
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


@app.post("/predict")
async def predict_tb(file: UploadFile = File(...)):
    if encoder is None or classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="The uploaded image is empty")
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            _, probabilities = classifier.predict(encoder(tensor))
        prob_tb = float(probabilities[0, 1])
        prob_normal = float(probabilities[0, 0])
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
