from pathlib import Path
import copy
import csv
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.models.encoder import get_encoder
from src.datasets.loader import (
    ShenzhenDataset, MontgomeryDataset, get_eval_transform
)
from src.client.local_train import finetune_local, evaluate_on_montgomery

BASE = Path("experiments/ablation_studies/few_shot_k")
CHECKPOINT = Path("experiments/main_experiment/checkpoints/flame_round_099.pt")
K20_FILE = Path(
    "experiments/ablation_studies/frozen_vs_unfrozen/results.json"
)

BASE.mkdir(parents=True, exist_ok=True)

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
if not K20_FILE.exists():
    raise FileNotFoundError(f"K=20 baseline not found: {K20_FILE}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = load_config("configs/local_stage2.yaml")

config.finetuning.seed = 42
config.finetuning.freeze_encoder = True
config.finetuning.projection_dim = 128
config.finetuning.epochs = 100
config.ssl.batch_size = 4

checkpoint = torch.load(
    CHECKPOINT, map_location=device, weights_only=False
)
if checkpoint.get("round") != 99:
    raise ValueError(
        f"Expected Round 99 checkpoint; got {checkpoint.get('round')}"
    )

transform = get_eval_transform(int(config.data.image_size))
shenzhen = ShenzhenDataset(
    config.data.shenzhen_path, transform=transform
)
montgomery = MontgomeryDataset(
    config.data.montgomery_path, transform=transform
)

shenzhen_loader = DataLoader(
    shenzhen, batch_size=4, shuffle=False
)
montgomery_loader = DataLoader(
    montgomery, batch_size=4, shuffle=False
)

print("Device:", device)
print("Shenzhen:", len(shenzhen))
print("Montgomery:", len(montgomery))
print("Checkpoint: Round 99")
print("K values to run: 5, 10, 15, 30, 50")

# Load existing frozen K=20 baseline.
with K20_FILE.open(encoding="utf-8") as f:
    previous = json.load(f)

k20 = previous["frozen"]
if (
    k20.get("checkpoint_round") != 99
    or k20.get("few_shot_k_per_class") != 20
    or k20.get("freeze_encoder") is not True
    or k20.get("seed") != 42
):
    raise ValueError(
        "Existing K=20 result does not match the expected protocol."
    )

results = {"k20": k20}

for k in (5, 10, 15, 30, 50):
    print(f"\n===== Running K={k} =====")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    run_config = copy.deepcopy(config)
    run_config.finetuning.few_shot_k = k

    encoder = get_encoder(run_config).to(device)
    encoder.load_state_dict(
        checkpoint["encoder_state_dict"], strict=True
    )

    head, shenzhen_metrics = finetune_local(
        0, encoder, shenzhen_loader, run_config, device
    )

    montgomery_metrics = evaluate_on_montgomery(
        encoder, head, montgomery_loader, device
    )

    results[f"k{k}"] = {
        "checkpoint_round": 99,
        "few_shot_k_per_class": k,
        "support_total": 2 * k,
        "seed": 42,
        "freeze_encoder": True,
        "projection_dim": 128,
        "adaptation_epochs": 100,
        "shenzhen_metrics": shenzhen_metrics,
        "montgomery_metrics": montgomery_metrics,
        "support_paths": getattr(head, "support_paths", []),
        "support_indices": getattr(head, "support_indices", [])
    }

    # Save after every K so completed experiments are not lost.
    (BASE / "results.json").write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8"
    )

    del encoder, head
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Build a compact CSV for reporting.
rows = []
for key in ("k5", "k10", "k15", "k20", "k30", "k50"):
    result = results[key]
    for dataset_name in ("shenzhen", "montgomery"):
        metrics = result[f"{dataset_name}_metrics"]
        rows.append({
            "K_per_class": result.get(
                "few_shot_k_per_class", 20
            ),
            "support_total": result.get(
                "support_total",
                2 * result.get("few_shot_k_per_class", 20)
            ),
            "dataset": dataset_name,
            "AUC": metrics.get("auc"),
            "accuracy": metrics.get("accuracy"),
            "sensitivity": metrics.get("sensitivity"),
            "specificity": metrics.get("specificity"),
            "F1": metrics.get("f1"),
            "balanced_accuracy": metrics.get("balanced_accuracy")
        })

with (BASE / "ablation_summary.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nFew-shot sweep finished.")
print("Detailed results:", BASE / "results.json")
print("Paper summary CSV:", BASE / "ablation_summary.csv")
