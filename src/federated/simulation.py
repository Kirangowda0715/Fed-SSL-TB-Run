"""
src/federated/simulation.py
----------------------------
Main entry point for the Federated SSL simulation.

Usage:
    python src/federated/simulation.py --config configs/default.yaml
    python src/federated/simulation.py --config configs/default.yaml --federated.rounds=30
    python src/federated/simulation.py --config configs/default.yaml --dry-run

Full federated loop (Stage 1 Pure SSL):
  For each round:
    1. Server broadcasts global encoder to all hospitals
    2. Each hospital runs ssl_local_train() (sequential or parallel) on local NIH
    3. Collect encoder weights + sample counts
    4. Server aggregates (FedAvg/FedProx) → updates global model
    5. Save checkpoint
    6. Log per-round SSL loss summary table
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
from src.utils.reproducibility import seed_everything
from src.utils.metrics import evaluate, format_metrics
from src.datasets.loader import (
    NIHDataset,
    get_base_transform, get_eval_transform,
)
from src.datasets.splitter import split_nih_to_hospitals, load_hospital_indices, split_cache_matches
from src.models.mae import build_mae
from src.client.ssl_train import ssl_local_train
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
        mean_ssl_loss = data.get("mean_ssl_loss", data.get("mean_total_loss", float("nan")))
        print(f"\n{'-'*70}")
        print(f"  Round {round_num+1:3d} | SSL Loss: {mean_ssl_loss:.4f}", end="")
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
    seed_everything(int(config.finetuning.seed))

    training_devices = _get_training_devices()
    device = training_devices[0]
    print(f"\n{'='*70}")
    print(f"  FedSSL -- Federated Self-Supervised Learning for TB Detection")
    print(f"{'='*70}")
    print(f"  Server device : {device}")
    print(f"  Training GPUs : {', '.join(str(item) for item in training_devices)}")
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
    else:
        hospital_loaders = _build_real_loaders(
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
        if not ckpt_dir.exists():
            rel_ckpt_dir = _ROOT / "experiments" / "checkpoints" / ckpt_dir.name
            if rel_ckpt_dir.exists():
                ckpt_dir = rel_ckpt_dir

        log_dir = Path(config.logging.log_dir)
        if not log_dir.exists():
            rel_log_dir = _ROOT / "experiments" / "logs" / log_dir.name
            if rel_log_dir.exists():
                log_dir = rel_log_dir

        # Prioritize flame_round_*.pt naming convention
        ckpts = list(ckpt_dir.glob("flame_round_*.pt"))
        if not ckpts:
            ckpts = list(ckpt_dir.glob("encoder_round_*.pt"))
        
        # Always try to load logger history when resuming
        logger = RoundLogger(log_dir)
        history_loaded = logger.load()
        
        if ckpts:
            # Sort numerically by round number in filename
            ckpts.sort(key=lambda x: int(x.stem.split("_")[-1]))
            latest_ckpt = ckpts[-1]
            latest_round = int(latest_ckpt.stem.split("_")[-1])
            
            print(f"[Resume] Found checkpoint: {latest_ckpt.name} in {ckpt_dir}")
            loaded_round = server.load_checkpoint(str(latest_ckpt))
            
            effective_round = loaded_round if loaded_round is not None else latest_round
            start_round = effective_round + 1
            print(f"[Resume] Successfully restored model weights from completed Round {effective_round}.")
            print(f"[Resume] Resuming execution at Round {start_round + 1} of {config.federated.rounds} (round_num={start_round}).")
            if history_loaded:
                print(f"[Resume] Restored {len(logger.rounds)} previous rounds from log\n")
            else:
                print(f"[Resume] Notice: Training history log not found in {log_dir}, new rounds will be appended.\n")

            privacy_cfg = getattr(config, "privacy", None)
            if bool(getattr(privacy_cfg, "enabled", False)):
                latest_ckpt_data = torch.load(str(latest_ckpt), map_location="cpu", weights_only=False)
                saved_metrics = latest_ckpt_data.get("metrics", {})
                saved_privacy = saved_metrics.get("privacy", {})
                if saved_privacy and "client_states" in saved_privacy:
                    client_privacy_states = {
                        int(k): v for k, v in saved_privacy["client_states"].items()
                    }
                    print(f"[Resume] Restored cumulative privacy states for {len(client_privacy_states)} clients.")
                else:
                    raise ValueError(
                        f"Cannot resume DP-enabled training from checkpoint {latest_ckpt.name}: "
                        f"checkpoint does not contain valid cumulative privacy metadata. "
                        f"Resuming without privacy expenditure history would falsely under-report cumulative budget."
                    )
        else:
            print(f"[Resume] No checkpoints found in {ckpt_dir}")
            if history_loaded:
                print(f"[Resume] Restored {len(logger.rounds)} rounds from training log")
                if logger.rounds:
                    latest_logged_round = max(h["round"] for h in logger.rounds)
                    start_round = latest_logged_round + 1
                    print(f"[Resume] Continuing from round {start_round}\n")
            else:
                print(f"[Resume] No checkpoint or history found. Starting from scratch.\n")
    
    # Initialize fresh logger if not resuming
    if logger is None:
        logger = RoundLogger(config.logging.log_dir)

    # Initialize per-client cumulative privacy states if not already restored
    privacy_cfg = getattr(config, "privacy", None)
    use_dp = bool(getattr(privacy_cfg, "enabled", False))
    if "client_privacy_states" not in locals() or client_privacy_states is None:
        client_privacy_states = {
            hid: {
                "hospital_id": hid,
                "steps": 0,
                "accountant_state": None,
            }
            for hid in range(1, len(hospital_loaders) + 1)
        }

    # -- Federated Loop ---------------------------------------------------
    print(f"\n[Simulation] Starting federated training for {config.federated.rounds} rounds...\n")

    for round_num in range(start_round, config.federated.rounds):
        print(f"\n{'='*70}")
        print(f"  ROUND {round_num + 1} / {config.federated.rounds}")
        print(f"{'='*70}")

        # 1. Broadcast global encoder + decoder weights to hospitals.
        global_weights = server.broadcast()

        # 2. Local SSL training at each hospital with cumulative privacy state
        if parallel:
            hospital_results = _train_parallel(
                global_model, global_weights, hospital_loaders, config, training_devices, client_privacy_states
            )
        else:
            hospital_results = _train_sequential(
                global_model, global_weights, hospital_loaders, config, device, client_privacy_states
            )

        # 3. Collect weights and sample counts
        federated_weights_list = [r["federated_weights"] for r in hospital_results]
        sample_counts        = [r["num_samples"] for r in hospital_results]
        ssl_losses           = [r["epoch_losses"][-1] for r in hospital_results]
        mean_ssl_loss        = float(np.mean(ssl_losses))

        print(f"\n  [Round {round_num+1}] Mean SSL Loss: {mean_ssl_loss:.4f}")

        # 4. Aggregate federated encoder + decoder weights
        aggregated_weights = server.aggregate(federated_weights_list, sample_counts)

        # 5. Update global encoder + decoder
        server.update_global_model(aggregated_weights)

        # 6. Build federation-wide per-client privacy summary
        privacy_summary = None
        if use_dp:
            privacy_summary = {
                "privacy_unit": "per_image_local_partition",
                "guarantee_scope": (
                    "Each hospital trains on a disjoint local dataset partition. "
                    "By parallel composition across disjoint client datasets, an individual image's "
                    "privacy budget is bounded strictly by its respective hospital client's local execution."
                ),
                "client_states": {
                    hid: {
                        "steps": r["privacy"]["steps"],
                        "max_grad_norm": r["privacy"]["max_grad_norm"],
                        "noise_multiplier": r["privacy"]["noise_multiplier"],
                        "delta": r["privacy"]["delta"],
                        "formal_epsilon_status": r["privacy"]["formal_epsilon_status"],
                        "epsilon": r["privacy"]["epsilon"],
                        "accountant_state": r["privacy"]["accountant_state"],
                    }
                    for hid, r in enumerate(hospital_results, start=1)
                    if "privacy" in r
                },
                "max_client_steps": max(
                    (r["privacy"]["steps"] for r in hospital_results if "privacy" in r),
                    default=0,
                ),
            }

        # Save checkpoint
        ckpt_metrics = {"loss": mean_ssl_loss}
        if privacy_summary:
            ckpt_metrics["privacy"] = privacy_summary

        ckpt_path = server.save_checkpoint(round_num, metrics=ckpt_metrics)
        print(f"  [Round {round_num+1}] Checkpoint saved -> {ckpt_path}")

        # 7. Log round
        log_entry = {
            "mean_ssl_loss": mean_ssl_loss,
            "hospital_losses": [{"ssl": l} for l in ssl_losses],
            "sample_counts": sample_counts,
        }
        if privacy_summary:
            log_entry["privacy"] = privacy_summary
            print(f"  [Round {round_num+1}] DP Tracking: max_client_steps = {privacy_summary['max_client_steps']} | unit = {privacy_summary['privacy_unit']}")
            for hid, c_data in privacy_summary["client_states"].items():
                print(f"    - Hospital {hid}: steps={c_data['steps']}, status={c_data['formal_epsilon_status']}, eps={c_data['epsilon']}")
        logger.log(round_num, log_entry)

    # ── Final Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  STAGE 1 PRETRAINING COMPLETE")
    print(f"  {server.summary()}")
    print(f"{'='*70}\n")

    logger.save()
    print("[Simulation] Done.")




def _get_training_devices() -> List[torch.device]:
    """Return one device per visible GPU, or CPU when CUDA is unavailable."""
    if not torch.cuda.is_available():
        return [torch.device("cpu")]
    return [torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count())]


# ─── Hospital Training Helpers ────────────────────────────────────────────────

def _train_sequential(
    global_model,
    global_weights,
    hospital_loaders,
    config,
    device,
    client_privacy_states=None,
) -> List[Dict[str, Any]]:
    """Train hospitals one-by-one (default mode)."""
    results = []
    for hospital_id, loader in enumerate(hospital_loaders, start=1):
        hospital_model = copy.deepcopy(global_model)
        p_state = client_privacy_states.get(hospital_id) if client_privacy_states else None
        result = ssl_local_train(
            hospital_id=hospital_id,
            model=hospital_model,
            dataloader=loader,
            config=config,
            global_weights=global_weights,
            device=device,
            privacy_state=p_state,
        )
        results.append(result)
        if client_privacy_states is not None and "privacy" in result:
            client_privacy_states[hospital_id] = result["privacy"]
    return results


def _train_parallel(
    global_model,
    global_weights,
    hospital_loaders,
    config,
    training_devices,
    client_privacy_states=None,
) -> List[Dict[str, Any]]:
    """Train hospitals in parallel, with no more than one worker per device."""
    results = [None] * len(hospital_loaders)

    def _train_one(args):
        hospital_id, loader, hospital_device = args
        hospital_model = copy.deepcopy(global_model)
        print(f"[Hospital {hospital_id}] Training on {hospital_device}")
        p_state = client_privacy_states.get(hospital_id) if client_privacy_states else None
        res = ssl_local_train(
            hospital_id=hospital_id,
            model=hospital_model,
            dataloader=loader,
            config=config,
            global_weights=global_weights,
            device=hospital_device,
            privacy_state=p_state,
        )
        return hospital_id, res

    worker_count = min(len(hospital_loaders), len(training_devices))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(
                _train_one,
                (hid, loader, training_devices[(hid - 1) % len(training_devices)]),
            ): hid
            for hid, loader in enumerate(hospital_loaders, start=1)
        }
        for future in as_completed(futures):
            hospital_id, result = future.result()
            results[hospital_id - 1] = result
            if client_privacy_states is not None and "privacy" in result:
                client_privacy_states[hospital_id] = result["privacy"]

    return results

    worker_count = min(len(hospital_loaders), len(training_devices))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(
                _train_one,
                (hid, loader, training_devices[(hid - 1) % len(training_devices)]),
            ): hid
            for hid, loader in enumerate(hospital_loaders, start=1)
        }
        for future in as_completed(futures):
            hospital_id, result = future.result()
            results[hospital_id - 1] = result

    return results


# ─── Loader Builders ─────────────────────────────────────────────────────────

def _build_synthetic_hospital_loaders(num_hospitals, batch_size, image_size):
    return [
        DataLoader(SyntheticNIHDataset(size=64, image_size=image_size), batch_size=batch_size, shuffle=True)
        for _ in range(num_hospitals)
    ]


def _build_real_loaders(config, num_hospitals, batch_size, image_size):
    """Build real dataset loaders for NIH (split across hospitals)."""
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

    # Split NIH → hospitals. Reuse a cache only when its exact configuration
    # matches dataset size, hospital count, strategy, alpha and seed.
    processed_dir = getattr(config.data, "processed_dir", "data/processed")
    hospital_indices_list = []
    alpha_val = float(getattr(config.data, "split_alpha", 0.5))
    split_seed = int(config.finetuning.seed)

    cache_valid = split_cache_matches(
        save_dir=processed_dir,
        n=len(nih_dataset),
        num_hospitals=num_hospitals,
        strategy=config.data.split_strategy,
        alpha=alpha_val,
        seed=split_seed,
    )

    if cache_valid:
        print("[Splitter] Loading configuration-matched hospital splits from disk...")
        try:
            for i in range(1, num_hospitals + 1):
                hospital_indices_list.append(load_hospital_indices(i, save_dir=processed_dir))
        except FileNotFoundError:
            cache_valid = False
            hospital_indices_list = []

    if not cache_valid:
        print("[Splitter] Cached split missing or configuration-mismatched; recomputing...")
        hospital_indices_list = split_nih_to_hospitals(
            dataset=nih_dataset,
            num_hospitals=num_hospitals,
            strategy=config.data.split_strategy,
            alpha=alpha_val,
            save_dir=processed_dir,
            seed=split_seed,
        )

    pin_memory = torch.cuda.is_available()
    hospital_loaders = [
        DataLoader(
            Subset(nih_dataset, indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2 if os.name != "nt" else 0, # num_workers > 0 can be unstable on Windows in some envs
            pin_memory=pin_memory,
        )
        for indices in hospital_indices_list
    ]

    return hospital_loaders


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
