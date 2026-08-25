"""FLAME-inspired prototypical classifier with a trainable projection space."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalHead(nn.Module):
    """Project encoder features, average support features, and classify by distance."""

    def __init__(
        self,
        embed_dim: int = 192,
        num_classes: int = 2,
        projection_dim: Optional[int] = None,
        use_linear: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.projection_dim = projection_dim or embed_dim
        self.use_linear = use_linear
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, self.projection_dim),
            nn.GELU(),
            nn.Linear(self.projection_dim, self.projection_dim),
        )
        if use_linear:
            self.linear_head = nn.Linear(self.projection_dim, num_classes)
        self.register_buffer(
            "prototypes", torch.zeros(num_classes, self.projection_dim)
        )
        self._prototypes_computed = False

    def _project(self, embeddings: torch.Tensor, already_projected: bool) -> torch.Tensor:
        if already_projected:
            if embeddings.shape[-1] != self.projection_dim:
                raise ValueError("Projected embeddings have the wrong dimensionality.")
            return embeddings
        if embeddings.shape[-1] != self.embed_dim:
            raise ValueError("Encoder embeddings have the wrong dimensionality.")
        return self.projection(embeddings)

    def compute_prototypes(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
        already_projected: bool = False,
    ) -> torch.Tensor:
        """Compute class means in projected embedding space."""
        embeddings = self._project(support_embeddings, already_projected)
        labels = support_labels.to(embeddings.device).long()
        prototypes = []
        for class_id in range(self.num_classes):
            class_embeddings = embeddings[labels == class_id]
            if class_embeddings.numel() == 0:
                prototypes.append(torch.zeros(self.projection_dim, device=embeddings.device, dtype=embeddings.dtype))
                continue
            prototypes.append(class_embeddings.mean(dim=0))
        result = torch.stack(prototypes)
        self.prototypes = result.detach()
        self._prototypes_computed = True
        return result

    def get_learnable_prototypes(
        self, support_embeddings: torch.Tensor, support_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute projected prototypes without detaching the support graph."""
        return self.compute_prototypes(support_embeddings, support_labels)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        prototypes: Optional[torch.Tensor] = None,
        already_projected: bool = False,
    ) -> torch.Tensor:
        queries = self._project(query_embeddings, already_projected)
        if prototypes is None:
            if not self._prototypes_computed:
                raise RuntimeError("Prototypes not computed. Call compute_prototypes() first.")
            prototypes = self.prototypes
        return self._prototypical_logits(queries, prototypes.to(queries.device))

    @staticmethod
    def _prototypical_logits(
        queries: torch.Tensor, prototypes: torch.Tensor
    ) -> torch.Tensor:
        distances = (queries.unsqueeze(1) - prototypes.unsqueeze(0)).pow(2).sum(dim=-1)
        return F.softmax(-distances, dim=-1)

    def prototypical_loss(
        self,
        query_embeddings: torch.Tensor,
        query_labels: torch.Tensor,
        prototypes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = self.forward(query_embeddings, prototypes)
        labels = query_labels.to(probs.device).long()
        return F.nll_loss(torch.log(probs.clamp_min(1e-8)), labels), probs

    def predict(
        self,
        query_embeddings: torch.Tensor,
        prototypes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = self.forward(query_embeddings, prototypes)
        return probs.argmax(dim=-1), probs

    def linear_forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if not self.use_linear:
            raise RuntimeError("Linear head is disabled for the few-shot classifier.")
        return self.linear_head(self.projection(embeddings))

    def linear_loss(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.linear_forward(embeddings)
        return F.cross_entropy(logits, labels), logits
