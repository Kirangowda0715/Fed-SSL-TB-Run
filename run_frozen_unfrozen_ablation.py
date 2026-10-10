import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.utils.config import load_config
from src.models.encoder import get_encoder
from src.datasets.loader import ShenzhenDataset, MontgomeryDataset, get_eval_transform
from src.client.local_train import finetune_local, evaluate_on_montgomery
from torch.utils.data import DataLoader

BASE = Path("experiments/ablation_studies/frozen_vs_unfrozen")
CHECKPOINT = Path("experiments/main_experiment/checkpoints/flame_round_099.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

config = load_config("configs/local_stage2.yaml")
config.finetuning.few_shot_k = 20
config.finetuning.seed = 42
config.finetuning.projection_dim = 128
config.finetuning.epochs = 100
config.ssl.batch_size = 4

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
if checkpoint.get("round") != 99:
    raise ValueError(f"Expected Round 99 checkpoint; found {checkpoint.get('round')}")

transform = get_eval_transform(int(config.data.image_size))
shenzhen = ShenzhenDataset(config.data.shenzhen_path, transform=transform)
montgomery = MontgomeryDataset(config.data.montgomery_path, transform=transform)
shenzhen_loader = DataLoader(shenzhen, batch_size=4, shuffle=False)

print("Device:", DEVICE)
print("Shenzhen images:", len(shenzhen))
print("Montgomery images:", len(montgomery))
print("Checkpoint round:", checkpoint["round"])

results = {}
support_records = {}

for frozen in (True, False):
    label = "frozen" if frozen else "unfrozen"
    print(f"\n===== Running {label.upper()} encoder =====")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    run_config = copy.deepcopy(config)
    run_config.finetuning.freeze_encoder = frozen

    encoder = get_encoder(run_config).to(DEVICE)
    encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)

    head, shenzhen_metrics = finetune_local(
        0, encoder, shenzhen_loader, run_config, DEVICE
    )

    montgomery_loader = DataLoader(
        montgomery, batch_size=4, shuffle=False
    )
    montgomery_metrics = evaluate_on_montgomery(
        encoder, head, montgomery_loader, DEVICE
    )

    support_records[label] = {
        "support_paths": getattr(head, "support_paths", []),
        "support_indices": getattr(head, "support_indices", [])
    }

    results[label] = {
        "checkpoint_round": 99,
        "few_shot_k_per_class": 20,
        "seed": 42,
        "freeze_encoder": frozen,
        "projection_dim": 128,
        "adaptation_epochs": 100,
        "shenzhen_metrics": shenzhen_metrics,
        "montgomery_metrics": montgomery_metrics
    }

    del encoder, head
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if support_records["frozen"] != support_records["unfrozen"]:
    raise RuntimeError(
        "Support records differ between runs. "
        "Inspect support selection before interpreting this comparison."
    )

(BASE / "results.json").write_text(
    json.dumps(results, indent=2, default=str), encoding="utf-8"
)
(BASE / "support_study_ids.json").write_text(
    json.dumps(support_records, indent=2, default=str), encoding="utf-8"
)

print("\nExperiment completed.")
print("Results:", BASE / "results.json")
print("Support records:", BASE / "support_study_ids.json")
print("\nCompare Shenzhen and Montgomery AUC, accuracy, sensitivity,")
print("specificity, F1, and balanced accuracy.")
