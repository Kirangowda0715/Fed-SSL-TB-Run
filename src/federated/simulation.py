"""
src/federated/simulation.py
----------------------------
Main entry point for the Federated SSL simulation.

Usage:
    python src/federated/simulation.py --config configs/default.yaml
    python src/federated/simulation.py --config configs/default.yaml --federated.rounds=30
    python src/federated/simulation.py --config configs/default.yaml --dry-run

Full federated loop:
  For each round:
    1. Server broadcasts global encoder to all hospitals
    2. Each hospital runs ssl_local_train() (sequential or parallel)
    3. Collect encoder weights + sample counts
    4. Server aggregates → updates global model
    5. Save checkpoint
    6. Every 5 rounds: fine-tune on Shenzhen → evaluate on Montgomery
    7. Log per-round summary table
"""

import os
import sys
import json
import time
import copy
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

# ── Make src importable when running as script ─────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.config import load_config
from src.utils.metrics import evaluate, format_metrics
from src.datasets.loader import (
    NIHDataset, ShenzhenDataset, MontgomeryDataset,
    get_base_transform, get_eval_transform,
)
from src.datasets.splitter import split_nih_to_hospitals, load_hospital_indices
from src.models.mae import build_mae
from src.client.ssl_train import ssl_local_train
from src.client.local_train import finetune_local, evaluate_on_montgomery
from src.server.server import FederatedServer


# ─── Dry-Run Synthetic Dataset ────────────────────────────────────────────────

class SyntheticNIHDataset(torch.utils.data.Dataset):
    """Tiny synthetic dataset for smoke-testing without real data."""
    def __init__(self, size=32, image_size=224):
        self.size = size
        self.image_size = image_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = torch.randn(3, self.image_size, self.image_size)
        return img, img.clone()  # two-view tuple


class SyntheticLabeledDataset(torch.utils.data.Dataset):
    def __init__(self, size=20, image_size=224, num_classes=2):
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
        self.labels = [i % num_classes for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = torch.randn(3, self.image_size, self.image_size)
        return img, self.labels[idx]

    def get_labels(self):
        return self.labels


# ─── Logger ───────────────────────────────────────────────────────────────────

class RoundLogger:
    """Tracks per-round metrics and saves logs to disk."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.rounds: List[Dict[str, Any]] = []

    def log(self, round_num: int, data: Dict[str, Any]) -> None:
        entry = {"round": round_num, **data}
        self.rounds.append(entry)
        ssl_loss = data.get("mean_ssl_loss", float("nan"))
        print(f"\n{'-'*70}")
        print(f"  Round {round_num+1:3d} | SSL Loss: {ssl_loss:.4f}", end="")
        if "eval_metrics" in data:
            m = data["eval_metrics"]
            print(f" | {format_metrics(m)}", end="")
        print(f"\n{'-'*70}")

    def save(self, filename: str = "training_log.json") -> str:
        path = self.log_dir / filename
        # Convert numpy arrays for JSON serialization
        def _serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
                return float(obj)
            return str(obj)

        with open(str(path), "w") as f:
            json.dump(self.rounds, f, indent=2, default=_serializable)
        print(f"\n[Logger] Training log saved -> {path}")
        return str(path)

    def load(self, filename: str = "training_log.json") -> bool:
        """Loads existing logs from disk. Returns True if successful."""
        path = self.log_dir / filename
        if path.exists():
            try:
                with open(str(path), "r") as f:
                    self.rounds = json.load(f)
                print(f"[Logger] Restored {len(self.rounds)} rounds from {path}")
                return True
            except Exception as e:
                print(f"[Logger] Failed to load log: {e}")
        return False


# ─── Main Simulation ──────────────────────────────────────────────────────────

def main():
    # ── Parse --dry-run flag separate from config overrides ──────────────
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--dry-run", action="store_true")
    pre_parser.add_argument("--parallel", action="store_true",
                            help="Run hospital training in parallel with threads")
    pre_parser.add_argument("--resume", action="store_true",
                            help="Resume training from latest checkpoint")
    pre_args, remaining = pre_parser.parse_known_args()

    dry_run  = pre_args.dry_run
    parallel = pre_args.parallel
    resume   = pre_args.resume

    # ── Load config (handles --config and dotted overrides) ───────────────
    sys.argv = [sys.argv[0]] + remaining  # pass remaining args to load_config
    config = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"  FedSSL -- Federated Self-Supervised Learning for TB Detection")
    print(f"{'='*70}")
    print(f"  Device        : {device}")
    print(f"  Backbone      : {config.model.backbone}")
    print(f"  Rounds        : {config.federated.rounds}")
    print(f"  Aggregation   : {config.federated.aggregation}")
    print(f"  Split strategy: {config.data.split_strategy}")
    print(f"  Dry run       : {dry_run}")
    print(f"{'='*70}\n")

# --- Build datasets ---
    num_hospitals = config.data.num_hospitals
    image_size    = config.data.image_size
    batch_size    = config.ssl.batch_size

    if dry_run:
        print("[DRY-RUN] Using synthetic data — no real datasets required.\n")
        hospital_loaders = _build_synthetic_hospital_loaders(
            num_hospitals, batch_size, image_size
        )
        shenzhen_loader   = DataLoader(SyntheticLabeledDataset(20, image_size), batch_size=8)
        montgomery_loader = DataLoader(SyntheticLabeledDataset(20, image_size), batch_size=8)
    else:
        hospital_loaders, shenzhen_loader, montgomery_loader = _build_real_loaders(
            config, num_hospitals, batch_size, image_size
        )

# --- Initialize server & global model ---
    server = FederatedServer(config, device=device)
    global_model = server.initialize_global_model()

# --- Resume logic ---
    start_round = 0
    logger = None  # Will be initialized after resume decision
    
    if resume:
        print(f"\n[Resume] Attempting to restore from checkpoint & history...")
        ckpt_dir = Path(config.logging.checkpoint_dir)
        ckpts = list(ckpt_dir.glob("encoder_round_*.pt"))
        
        # Always try to load logger history when resuming (checkpoint may or may not exist)
        logger = RoundLogger(config.logging.log_dir)
        history_loaded = logger.load()
        
        if ckpts:
            # Sort by round number in filename
            ckpts.sort(key=lambda x: int(x.stem.split("_")[-1]))
            latest_ckpt = ckpts[-1]
            latest_round = int(latest_ckpt.stem.split("_")[-1])
            
            print(f"[Resume] Found checkpoint: {latest_ckpt.name}")
            server.load_checkpoint(str(latest_ckpt))
            
            start_round = latest_round + 1
            print(f"[Resume] Checkpoint loaded from round {latest_round}")
            print(f"[Resume] Ready to continue from Round {start_round + 1}")
            if history_loaded:
                print(f"[Resume] Restored {len(logger.rounds)} previous rounds from log\n")
            else:
                print(f"[Resume] WARNING: Could not restore training history\n")
        else:
            print(f"[Resume] No checkpoints found in {ckpt_dir}")
            if history_loaded:
                print(f"[Resume] But restored {len(logger.rounds)} rounds from training log")
                # Infer start_round from loaded history
                if logger.rounds:
                    latest_logged_round = max(h["round"] for h in logger.rounds)
                    start_round = latest_logged_round + 1
                    print(f"[Resume] Continuing from round {start_round}\n")
            else:
                print(f"[Resume] No checkpoint or history found. Starting from scratch.\n")
    
    # Initialize fresh logger if not resuming
    if logger is None:
        logger = RoundLogger(config.logging.log_dir)

    # -- Federated Loop ---------------------------------------------------
    print(f"\n[Simulation] Starting federated training for {config.federated.rounds} rounds...\n")

    for round_num in range(start_round, config.federated.rounds):
        print(f"\n{'='*70}")
        print(f"  ROUND {round_num + 1} / {config.federated.rounds}")
        print(f"{'='*70}")

        # 1. Broadcast global encoder weights
        global_weights = server.broadcast()

        # 2. Local SSL training at each hospital
        if parallel:
            hospital_results = _train_parallel(
                global_model, global_weights, hospital_loaders, config, device
            )
        else:
            hospital_results = _train_sequential(
                global_model, global_weights, hospital_loaders, config, device
            )

        # 3. Collect weights and sample counts
        encoder_weights_list = [r["encoder_weights"] for r in hospital_results]
        sample_counts        = [r["num_samples"] for r in hospital_results]
        epoch_losses         = [r["epoch_losses"][-1] for r in hospital_results]
        mean_ssl_loss        = float(np.mean(epoch_losses))

        print(f"\n  [Round {round_num+1}] Mean SSL Loss: {mean_ssl_loss:.4f}")

        # 4. Aggregate
        aggregated_weights = server.aggregate(encoder_weights_list, sample_counts)

        # 5. Update global model
        server.update_global_model(aggregated_weights)

        # 6. Every 5 rounds — fine-tune + evaluate
        eval_metrics = None
        if (round_num + 1) % 5 == 0 or (round_num + 1) == config.federated.rounds:
            print(f"\n  [Round {round_num+1}] Running few-shot fine-tuning + evaluation...")
            encoder_copy = copy.deepcopy(server.get_encoder()).to(device)

            try:
                proto_head, finetune_metrics = finetune_local(
                    hospital_id=0,
                    encoder=encoder_copy,
                    shenzhen_loader=shenzhen_loader,
                    config=config,
                    device=device,
                )

                if dry_run:
                    # Use synthetic evaluation metrics for dry-run
                    y_true  = np.array([0, 1, 0, 1, 0, 1, 0, 1])
                    y_pred  = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
                    eval_metrics = evaluate(y_true, y_pred)
                else:
                    eval_metrics = evaluate_on_montgomery(
                        encoder=encoder_copy,
                        proto_head=proto_head,
                        montgomery_loader=montgomery_loader,
                        support_loader=shenzhen_loader,
                        config=config,
                        device=device,
                    )

                print(f"  [Round {round_num+1}] Montgomery: {format_metrics(eval_metrics)}")

            except Exception as e:
                print(f"  [WARNING] Evaluation failed: {e}")
                traceback.print_exc()

        # 7. Save checkpoint
        ckpt_path = server.save_checkpoint(round_num, metrics=eval_metrics)
        print(f"  [Round {round_num+1}] Checkpoint saved -> {ckpt_path}")

        # 8. Log round
        log_entry = {
            "mean_ssl_loss": mean_ssl_loss,
            "hospital_losses": epoch_losses,
            "sample_counts": sample_counts,
        }
        if eval_metrics:
            log_entry["eval_metrics"] = eval_metrics

        logger.log(round_num, log_entry)

    # ── Final Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"  {server.summary()}")
    print(f"{'='*70}\n")

    logger.save()
    print("[Simulation] Done.")


# ─── Hospital Training Helpers ────────────────────────────────────────────────

def _train_sequential(
    global_model,
    global_weights,
    hospital_loaders,
    config,
    device,
) -> List[Dict[str, Any]]:
    """Train hospitals one-by-one (default mode)."""
    results = []
    for hospital_id, loader in enumerate(hospital_loaders, start=1):
        # Give each hospital a fresh copy of the global model
        hospital_model = copy.deepcopy(global_model)
        result = ssl_local_train(
            hospital_id=hospital_id,
            model=hospital_model,
            dataloader=loader,
            config=config,
            global_weights=global_weights,
            device=device,
        )
        results.append(result)
    return results


def _train_parallel(
    global_model,
    global_weights,
    hospital_loaders,
    config,
    device,
) -> List[Dict[str, Any]]:
    """Train hospitals in parallel using ThreadPoolExecutor."""
    results = [None] * len(hospital_loaders)

    def _train_one(args):
        hospital_id, loader = args
        hospital_model = copy.deepcopy(global_model)
        return hospital_id, ssl_local_train(
            hospital_id=hospital_id,
            model=hospital_model,
            dataloader=loader,
            config=config,
            global_weights=global_weights,
            device=device,
        )

    with ThreadPoolExecutor(max_workers=min(len(hospital_loaders), 4)) as pool:
        futures = {
            pool.submit(_train_one, (hid, loader)): hid
            for hid, loader in enumerate(hospital_loaders, start=1)
        }
        for future in as_completed(futures):
            hospital_id, result = future.result()
            results[hospital_id - 1] = result

    return results


# ─── Loader Builders ─────────────────────────────────────────────────────────

def _build_synthetic_hospital_loaders(num_hospitals, batch_size, image_size) -> List[DataLoader]:
    return [
        DataLoader(
            SyntheticNIHDataset(size=64, image_size=image_size),
            batch_size=batch_size,
            shuffle=True,
        )
        for _ in range(num_hospitals)
    ]


def _build_real_loaders(config, num_hospitals, batch_size, image_size):
    """Build real dataset loaders for NIH (split), Shenzhen, and Montgomery."""
    # Read limit from config, fallback to 5000 if not set
    limit_val = getattr(config.ssl, "limit_samples", 5000)

    nih_dataset = NIHDataset(
        root_dir=config.data.nih_path,
        image_size=image_size,
        limit=limit_val,
    )

    if len(nih_dataset) == 0:
        print(f"\n[ERROR] NIH dataset at {config.data.nih_path} is empty.")
        print("Please ensure images are present. If you want to test the full pipeline without real data,")
        print("run the mock data generator:  python src/utils/generate_mock_data.py")
        sys.exit(1)
    
    print(f"Loaded NIH dataset with {len(nih_dataset)} images.")

    # Split NIH → hospitals (loads from disk if already computed)
    processed_dir = "data/processed"
    hospital_indices_list = []
    hospital_1_index_path = Path(processed_dir) / "hospital_1" / "indices.npy"

    # Robustness: Check if pre-computed indices are valid for current dataset size
    should_recompute = not hospital_1_index_path.exists()
    if not should_recompute:
        # Load all indices into a list to check total coverage
        all_loaded_indices = []
        for i in range(1, num_hospitals + 1):
            all_loaded_indices.extend(load_hospital_indices(i, save_dir=processed_dir))
        
        # Recompute if existing total indices don't match current dataset size
        if len(all_loaded_indices) != len(nih_dataset):
            print(f"[Splitter] Pre-computed indices count ({len(all_loaded_indices)}) differs from current "
                  f"dataset size ({len(nih_dataset)}). Re-computing...")
            should_recompute = True

    if not should_recompute:
        print("[Splitter] Loading pre-computed hospital splits from disk...")
        for i in range(1, num_hospitals + 1):
            indices = load_hospital_indices(i, save_dir=processed_dir)
            hospital_indices_list.append(indices)
    else:
        print("[Splitter] Computing hospital splits...")
        alpha_val = getattr(config.data, "split_alpha", 0.5)
        hospital_indices_list = split_nih_to_hospitals(
            dataset=nih_dataset,
            num_hospitals=num_hospitals,
            strategy=config.data.split_strategy,
            alpha=alpha_val,
            save_dir=processed_dir,
        )

    hospital_loaders = [
        DataLoader(
            Subset(nih_dataset, indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2 if os.name != "nt" else 0, # num_workers > 0 can be unstable on Windows in some envs
            pin_memory=True,
        )
        for indices in hospital_indices_list
    ]

    shenzhen_dataset = ShenzhenDataset(
        root_dir=config.data.shenzhen_path,
        image_size=image_size,
    )
    if len(shenzhen_dataset) == 0:
        print(f"\n[ERROR] Shenzhen dataset at {config.data.shenzhen_path} is empty.")
        sys.exit(1)

    shenzhen_loader = DataLoader(
        shenzhen_dataset, batch_size=batch_size,
        shuffle=True, num_workers=2 if os.name != "nt" else 0,
    )

    montgomery_dataset = MontgomeryDataset(
        root_dir=config.data.montgomery_path,
        image_size=image_size,
    )
    if len(montgomery_dataset) == 0:
        print(f"\n[ERROR] Montgomery dataset at {config.data.montgomery_path} is empty.")
        sys.exit(1)

    montgomery_loader = DataLoader(
        montgomery_dataset, batch_size=batch_size,
        shuffle=False, num_workers=2 if os.name != "nt" else 0,
    )

    return hospital_loaders, shenzhen_loader, montgomery_loader


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
