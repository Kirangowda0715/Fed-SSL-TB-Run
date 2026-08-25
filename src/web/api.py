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

from src.client.local_train import finetune_local
from src.datasets.loader import ShenzhenDataset, get_eval_transform
from src.models.encoder import get_encoder
from src.utils.config import load_config

app = FastAPI(title="FedSSL TB Detection API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = get_eval_transform(config.data.image_size)
encoder = None
classifier = None


def _load_fl_encoder():
    model = get_encoder(config.model.backbone, config.model.embed_dim)
    checkpoint_dir = Path(config.logging.checkpoint_dir)
    checkpoint = checkpoint_dir / "best_encoder.pt"
    if not checkpoint.exists():
        checkpoints = sorted(checkpoint_dir.glob("encoder_round_*.pt"))
        checkpoint = checkpoints[-1] if checkpoints else checkpoint
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state.get("encoder_state_dict", state))
        print(f"[Startup] FL encoder loaded: {checkpoint.name}")
    else:
        print("[Startup] WARNING: no FL checkpoint found; using initialized encoder")
    return model.to(device)


@app.on_event("startup")
async def load_models():
    global encoder, classifier
    encoder = _load_fl_encoder()
    dataset = ShenzhenDataset(config.data.shenzhen_path, transform=transform, image_size=config.data.image_size)
    loader = DataLoader(dataset, batch_size=config.ssl.batch_size, shuffle=False)
    classifier, _ = finetune_local(0, encoder, loader, config, device)
    classifier.eval()
    print("[Startup] Ready (canonical prototypical few-shot mode)")


@app.get("/health")
def health_check():
    return {"status": "healthy", "device": str(device), "model": config.model.backbone, "classifier": "prototypical_few_shot"}


@app.get("/metrics")
def get_training_metrics():
    log_path = Path(config.logging.log_dir) / "training_log.json"
    if not log_path.exists():
        return []
    with log_path.open("r") as handle:
        return json.load(handle)


@app.post("/predict")
async def predict_tb(file: UploadFile = File(...)):
    if encoder is None or classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
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
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
