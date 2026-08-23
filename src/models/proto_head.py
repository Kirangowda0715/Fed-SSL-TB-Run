"""
src/models/proto_head.py
------------------------
Prototypical Network classification head for few-shot TB detection.

Prototypical Networks (Snell et al., NeurIPS 2017):
  - Compute a prototype (mean embedding) per class from the support set
  - Classify query samples by nearest prototype using Euclidean distance
  - The pretrained encoder IS the feature extractor — no extra projection needed

Key design decision: We do NOT add a trainable projection layer here.
The MAE encoder (trained via FedProx for 14 rounds) already produces
meaningful embeddings. Adding a randomly-initialized projection on top
of it destroys this learned structure and causes random predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PrototypicalHead(nn.Module):
    """
    Prototypical Network classification head for binary TB/Normal classification.

    Inference:
        1. Call compute_prototypes(support_embeddings, support_labels)
        2. Call predict(query_embedding) to get class + probability

    Args:
        embed_dim   : Encoder output dimensionality (must match encoder)
        num_classes : Number of classes (default 2: Normal=0, TB=1)
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_classes: int = 2,
        use_linear: bool = True,   # kept for backward-compat, not used in proto path
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_linear = use_linear

        # Fallback linear head (not used in prototypical path)
        if use_linear:
            self.linear_head = nn.Linear(embed_dim, num_classes)

        # Stored prototypes (set during few-shot episode)
        self.register_buffer("prototypes", torch.zeros(num_classes, embed_dim))
        self._prototypes_computed = False

    # ─── Prototype Computation ───────────────────────────────────────────────

    def compute_prototypes(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute class prototypes as mean of raw encoder embeddings per class.

        Args:
            support_embeddings : (N_support, embed_dim) — raw encoder outputs
            support_labels     : (N_support,) — integer class labels {0, 1}

        Returns:
            prototypes : (num_classes, embed_dim)
        """
        device = support_embeddings.device
        support_labels = support_labels.to(device)
        prototypes = torch.zeros(
            self.num_classes, self.embed_dim,
            device=device,
            dtype=support_embeddings.dtype,
        )
        for c in range(self.num_classes):
            mask = (support_labels == c)
            if mask.sum() == 0:
                continue
            prototypes[c] = support_embeddings[mask].mean(dim=0)

        self.prototypes = prototypes.detach()
        self._prototypes_computed = True
        return prototypes

    # ─── Prototypical Forward ────────────────────────────────────────────────

    def forward(
        self,
        query_embeddings: torch.Tensor,
        prototypes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Classify query samples by nearest prototype (Euclidean distance).

        Args:
            query_embeddings : (B, embed_dim) raw encoder outputs
            prototypes       : (num_classes, embed_dim) — if None, use stored

        Returns:
            probs : (B, num_classes) softmax probabilities
        """
        if prototypes is None:
            if not self._prototypes_computed:
                raise RuntimeError(
                    "Prototypes not computed. Call compute_prototypes() first."
                )
            prototypes = self.prototypes

        # Ensure prototypes are on the same device as queries
        prototypes = prototypes.to(query_embeddings.device)

        return self._prototypical_logits(query_embeddings, prototypes)

    def _prototypical_logits(
        self,
        queries: torch.Tensor,
        protos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Negative squared Euclidean distance → softmax probabilities.
        This is the exact formulation from Snell et al. (2017).

        Args:
            queries : (B, D)
            protos  : (C, D)
        Returns:
            probs : (B, C)
        """
        # Ensure both tensors are on the same device (derive from queries)
        protos = protos.to(queries.device)
        # (B, 1, D) - (1, C, D) → (B, C, D)
        diffs = queries.unsqueeze(1) - protos.unsqueeze(0)
        # Squared Euclidean distance: (B, C)
        sq_dists = (diffs ** 2).sum(dim=-1)
        # Negative distance as logit (closer = higher score)
        logits = -sq_dists
        probs = F.softmax(logits, dim=-1)
        return probs

    # ─── Linear CE Forward (fallback) ────────────────────────────────────────

    def linear_forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if not self.use_linear:
            raise RuntimeError("Linear head not instantiated. Set use_linear=True.")
        return self.linear_head(embeddings)

    # ─── Loss Helpers ─────────────────────────────────────────────────────────

    def prototypical_loss(
        self,
        query_embeddings: torch.Tensor,
        query_labels: torch.Tensor,
        prototypes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CE loss over prototypical probabilities."""
        probs = self.forward(query_embeddings, prototypes)
        log_probs = torch.log(probs + 1e-8)
        # Ensure labels are on same device as log_probs
        query_labels = query_labels.to(log_probs.device)
        loss = F.nll_loss(log_probs, query_labels)
        return loss, probs

    def linear_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.linear_forward(embeddings)
        loss = F.cross_entropy(logits, labels)
        return loss, logits

    # ─── Utility ──────────────────────────────────────────────────────────────

    def predict(
        self,
        query_embeddings: torch.Tensor,
        prototypes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            predicted_labels : (B,) int tensor
            probs            : (B, num_classes) probability tensor
        """
        probs = self.forward(query_embeddings, prototypes)
        labels = probs.argmax(dim=-1)
        return labels, probs

    # ─── For backward compat with local_train.py ─────────────────────────────

    def get_learnable_prototypes(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Same as compute_prototypes but retains grad for training."""
        device = support_embeddings.device
        support_labels = support_labels.to(device)
        prototypes = torch.zeros(
            self.num_classes, self.embed_dim,
            device=device,
            dtype=support_embeddings.dtype,
        )
        for c in range(self.num_classes):
            mask = (support_labels == c)
            if mask.sum() > 0:
                prototypes[c] = support_embeddings[mask].mean(dim=0)
        return prototypes
