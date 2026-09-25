"""
src/server/server.py
--------------------
Central Federated Learning server orchestration.

Responsibilities:
  - Maintain global MAE model
  - Broadcast global encoder weights to hospitals
  - Aggregate received encoder weights using FedAvg / FedProx
  - Update and checkpoint the global model
  - Track best model by Montgomery AUC
"""

import os
import copy
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn

from src.models.mae import build_mae, MaskedAutoencoder
from src.models.proto_head import PrototypicalHead
from src.server.aggregator import fedavg, fedprox


class FederatedServer:
    """
    Central server for federated SSL training.

    Args:
        config : Loaded config SimpleNamespace
        device : torch.device for the server-side global model
    """

    def __init__(self, config, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Global model (will be initialized in initialize_global_model)
        self.global_model: Optional[MaskedAutoencoder] = None

        # Checkpoint bookkeeping
        self.checkpoint_dir = Path(config.logging.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Best model tracking (by Montgomery AUC)
        self.best_auc: float = 0.0
        self.best_round: int = -1
        self.best_encoder_weights: Optional[Dict[str, Any]] = None

        # Aggregation strategy
        self.aggregation = config.federated.aggregation.lower()
        self.fedprox_mu = getattr(config.federated, "fedprox_mu", 0.01)

    # ─── Model Initialization ────────────────────────────────────────────────

    def initialize_global_model(self) -> MaskedAutoencoder:
        """
        Build the global MAE with random initialization.

        Returns:
            global_model : MaskedAutoencoder
        """
        self.global_model = build_mae(
            backbone=self.config.model.backbone,
            embed_dim=self.config.model.embed_dim,
            mask_ratio=self.config.model.mask_ratio,
            decoder_depth=self.config.model.decoder_depth,
            image_size=self.config.data.image_size,
            projection_dim=getattr(self.config.model, "projection_dim", 128),
        ).to(self.device)

        print(
            f"[Server] Global model initialized | "
            f"Backbone: {self.config.model.backbone} | "
            f"Embed dim: {self.config.model.embed_dim} | "
            f"Device: {self.device}"
        )
        return self.global_model

    # ─── Broadcast ────────────────────────────────────────────────────────────

    def broadcast(self) -> Dict[str, Any]:
        """
        Return a copy of the current global encoder weights to broadcast to hospitals.

        Returns:
            encoder_weights : Deep-copied encoder state_dict
        """
        assert self.global_model is not None, (
            "Global model not initialized. Call initialize_global_model() first."
        )
        return self.global_model.get_encoder_weights()

    def get_global_weights(self) -> Dict[str, Any]:
        """Alias for broadcast — returns global encoder state_dict."""
        return self.broadcast()

    # ─── Aggregation ─────────────────────────────────────────────────────────

    def aggregate(
        self,
        received_weights: List[Dict[str, Any]],
        sample_counts: List[int],
    ) -> Dict[str, Any]:
        """
        Aggregate encoder weights from hospitals using configured strategy.

        Args:
            received_weights : List of encoder state_dicts from hospitals
            sample_counts    : Number of training samples per hospital

        Returns:
            Aggregated encoder state_dict
        """
        if self.aggregation == "fedavg":
            aggregated = fedavg(received_weights, sample_counts)

        elif self.aggregation == "fedprox":
            global_weights = self.broadcast()
            aggregated = fedprox(
                global_weights,
                received_weights,
                sample_counts,
                mu=self.fedprox_mu,
            )
        else:
            raise ValueError(
                f"Unknown aggregation strategy '{self.aggregation}'. "
                f"Use 'fedavg' or 'fedprox'."
            )

        return aggregated

    # ─── Update Global Model ─────────────────────────────────────────────────

    def update_global_model(self, aggregated_weights: Dict[str, Any]) -> None:
        """
        Load aggregated weights into the global encoder.

        Args:
            aggregated_weights : Aggregated encoder state_dict from aggregate()
        """
        assert self.global_model is not None
        if "encoder" in aggregated_weights:
            self.global_model.load_federated_weights(aggregated_weights)
        else:
            self.global_model.load_encoder_weights(aggregated_weights)

    # ─── Checkpointing ───────────────────────────────────────────────────────

    def save_checkpoint(
        self,
        round_num: int,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save global encoder checkpoint after each federated round.

        Args:
            round_num : Current round number (0-indexed)
            metrics   : Optional evaluation metrics dict

        Returns:
            Path to saved checkpoint file
        """
        assert self.global_model is not None

        checkpoint = {
            "round": round_num,
            "encoder_state_dict": self.global_model.encoder.state_dict(),
            "decoder_state_dict": self.global_model.decoder.state_dict(),
            "proto_head_state_dict": self.global_model.proto_head.state_dict(),
            "config": {
                "backbone": self.config.model.backbone,
                "embed_dim": self.config.model.embed_dim,
                "mask_ratio": self.config.model.mask_ratio,
            },
        }
        if metrics:
            checkpoint["metrics"] = metrics

        ckpt_path = self.checkpoint_dir / f"flame_round_{round_num:03d}.pt"
        torch.save(checkpoint, str(ckpt_path))

        # Track best model by AUC
        if metrics and "auc" in metrics:
            current_auc = metrics["auc"]
            if current_auc > self.best_auc:
                self.best_auc = current_auc
                self.best_round = round_num
                self.best_encoder_weights = copy.deepcopy(
                    self.global_model.get_encoder_weights()
                )
                # Save best model separately
                best_path = self.checkpoint_dir / "best_flame.pt"
                torch.save(
                    {**checkpoint, "best_auc": self.best_auc},
                    str(best_path),
                )
                print(
                    f"  [Server] [BEST MODEL] saved | "
                    f"Round {round_num} | AUC={self.best_auc:.4f}"
                )

        return str(ckpt_path)

    def load_checkpoint(self, path: str) -> int:
        """
        Load a checkpoint and restore global encoder weights.

        Args:
            path : Path to .pt checkpoint file

        Returns:
            round_num : The round number stored in the checkpoint
        """
        assert self.global_model is not None
        ckpt = torch.load(path, map_location=self.device)
        if "decoder_state_dict" in ckpt:
            self.global_model.load_federated_weights({
                "encoder": ckpt["encoder_state_dict"],
                "decoder": ckpt["decoder_state_dict"],
                "proto_head": ckpt["proto_head_state_dict"],
            })
        else:
            self.global_model.load_encoder_weights(ckpt["encoder_state_dict"])
        round_num = ckpt.get("round", 0)
        print(f"[Server] Loaded checkpoint from round {round_num}: {path}")
        return round_num

    # ─── Utility ─────────────────────────────────────────────────────────────

    def get_encoder(self) -> nn.Module:
        """Return the global encoder module (for fine-tuning)."""
        assert self.global_model is not None
        return self.global_model.encoder

    def get_global_model(self) -> MaskedAutoencoder:
        """Return the full global MAE model."""
        assert self.global_model is not None
        return self.global_model

    def summary(self) -> str:
        """Return a summary string of server state."""
        if self.best_round >= 0:
            return (
                f"FederatedServer | "
                f"Aggregation: {self.aggregation} | "
                f"Best AUC: {self.best_auc:.4f} @ Round {self.best_round}"
            )
        return f"FederatedServer | Aggregation: {self.aggregation}"
