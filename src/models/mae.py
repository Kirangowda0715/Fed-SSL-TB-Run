"""
src/models/mae.py
-----------------
Full Masked Autoencoder (MAE) model combining encoder + decoder.

Design:
  - patchify()        : Split 224×224 image into 14×14 grid of 16×16 patches
  - random_masking()  : Mask 75% of patches, return visible + restore indices
  - forward()         : Encode visible → decode all → MSE loss on masked patches
  - get_encoder_weights() : Return ONLY encoder state_dict (for federated sharing)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any

from src.models.encoder import get_encoder, ResNet50Encoder, ViTSmallEncoder
from src.models.decoder import MAEDecoder
from src.models.proto_head import PrototypicalHead


class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder for SSL pretraining on chest X-rays.

    Only the encoder weights are shared across hospitals in federated learning.
    The decoder remains local to each hospital.

    Args:
        encoder       : Encoder nn.Module (ResNet50 or ViT-Small)
        decoder       : MAEDecoder nn.Module
        mask_ratio    : Fraction of patches to mask (default 0.75)
        image_size    : Input image size (default 224)
        patch_size    : Patch side length in pixels (default 16)
        in_channels   : Number of image channels (default 3)
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        mask_ratio: float = 0.75,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        projection_dim: int = 128,
        proto_head: nn.Module = None,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"

        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.proto_head = proto_head or PrototypicalHead(
            embed_dim=encoder.embed_dim, projection_dim=projection_dim
        )
        self.num_patches = (image_size // patch_size) ** 2   # 196 for 224/16

        # For ResNet50: we need a patch embedding layer to get per-patch tokens
        # For ViT: the backbone already tokenizes patches
        self._is_resnet = isinstance(encoder, ResNet50Encoder)

        if self._is_resnet:
            # Patch embedding: image → (B, num_patches, embed_dim) tokens
            # We use a small patch embedding dim then project to embed_dim
            _patch_dim = patch_size * patch_size * in_channels  # raw pixel dim per patch
            self.patch_embed = nn.Linear(_patch_dim, encoder.embed_dim)
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, encoder.embed_dim),
                requires_grad=False,
            )
            self._init_pos_embed()
            self.pre_norm = nn.LayerNorm(encoder.embed_dim)
            # Lightweight per-patch encoder: 2-layer MLP
            self.patch_encoder = nn.Sequential(
                nn.Linear(encoder.embed_dim, encoder.embed_dim),
                nn.GELU(),
                nn.Linear(encoder.embed_dim, encoder.embed_dim),
                nn.LayerNorm(encoder.embed_dim),
            )

    def _init_pos_embed(self) -> None:
        """Initialize sinusoidal positional embeddings."""
        from src.models.decoder import _get_sinusoidal_pos_embed
        emb = _get_sinusoidal_pos_embed(self.encoder.embed_dim, self.num_patches)
        self.pos_embed.data.copy_(emb.unsqueeze(0))

    # ─── Patchify & Unpatchify ───────────────────────────────────────────────

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Divide image into non-overlapping patches.

        Args:
            x : (B, C, H, W) image tensor
        Returns:
            patches : (B, num_patches, patch_size²×C)
        """
        B, C, H, W = x.shape
        p = self.patch_size
        assert H == W == self.image_size, f"Expected {self.image_size}×{self.image_size}, got {H}×{W}"
        h = w = H // p
        # (B, C, h, p, w, p) → (B, h, w, p, p, C) → (B, h*w, p*p*C)
        x = x.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, h * w, p * p * C)
        return x

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct image from patch predictions.

        Args:
            patches : (B, num_patches, patch_size²×C)
        Returns:
            img : (B, C, H, W)
        """
        B, N, _ = patches.shape
        p = self.patch_size
        C = self.in_channels
        h = w = int(N ** 0.5)
        patches = patches.reshape(B, h, w, p, p, C)
        patches = patches.permute(0, 5, 1, 3, 2, 4)
        img = patches.reshape(B, C, h * p, w * p)
        return img

    # ─── Random Masking ──────────────────────────────────────────────────────

    def random_masking(
        self,
        tokens: torch.Tensor,
        mask_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly mask mask_ratio fraction of patch tokens.

        Args:
            tokens     : (B, num_patches, D) patch tokens
            mask_ratio : Fraction to mask (e.g., 0.75)

        Returns:
            visible_tokens : (B, num_visible, D) — unmasked tokens
            mask           : (B, num_patches) — 1=masked, 0=visible
            ids_restore    : (B, num_patches) — indices to restore original order
        """
        B, N, D = tokens.shape
        num_keep = int(N * (1 - mask_ratio))

        # Random noise for shuffling
        noise = torch.rand(B, N, device=tokens.device)

        # Sort by noise: ids_shuffle[i][j] = index of j-th patch in sorted order
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # inverse permutation

        # Keep the first num_keep patches (lowest noise values)
        ids_keep = ids_shuffle[:, :num_keep]
        visible_tokens = torch.gather(
            tokens, dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        # Binary mask: 1 = masked, 0 = visible
        mask = torch.ones(B, N, device=tokens.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return visible_tokens, mask, ids_restore

    # ─── Encode ──────────────────────────────────────────────────────────────

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to per-patch tokens using ViT patch embedding."""
        tokens = self.encoder.vit.patch_embed(x)              # (B, N, D_vit)
        tokens = tokens + self.encoder.vit.pos_embed[:, 1:, :]  # 196 patch tokens (skip CLS)
        tokens = self.encoder.vit.pos_drop(tokens)
        return tokens

    def _encode_visible(self, visible_tokens: torch.Tensor) -> torch.Tensor:
        """Run ViT transformer blocks on visible patch tokens -> project to embed_dim."""
        x = visible_tokens
        for block in self.encoder.vit.blocks:
            x = block(x)
        x = self.encoder.vit.norm(x)               # (B, num_visible, D_vit)
        z = self.encoder.projection(x)             # (B, num_visible, embed_dim)
        return z

    # ─── Forward (MAE Loss) ──────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full MAE forward pass.

        Args:
            x : (B, C, H, W) or Tuple[(B,C,H,W), (B,C,H,W)] from two-view loader

        Returns:
            loss   : Scalar MAE reconstruction loss (MSE on masked patches only)
            pred   : (B, num_patches, patch_size²×C) reconstructed patches
            mask   : (B, num_patches) binary mask
        """
        # Handle two-view tuple from NIHDataset
        if isinstance(x, (list, tuple)):
            x = x[0]

        # Ground-truth patches
        target = self.patchify(x)                              # (B, N, P²C)

        # Tokenize → mask → encode → decode → loss
        tokens = self._tokenize(x)                             # (B, N, D)
        visible_tokens, mask, ids_restore = self.random_masking(tokens, self.mask_ratio)
        encoded = self._encode_visible(visible_tokens)         # (B, num_visible, embed_dim)
        pred = self.decoder(encoded, ids_restore)              # (B, N, P²C)

        # MSE loss on masked patches only
        loss = self._mae_loss(pred, target, mask)
        return loss, pred, mask

    def _mae_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss computed only on masked patches."""
        loss = F.mse_loss(pred, target, reduction="none")  # (B, N, P²C)
        loss = loss.mean(dim=-1)                            # (B, N)
        # Apply mask: only compute loss where mask==1
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss

    # ─── Federated Utility ───────────────────────────────────────────────────

    def get_encoder_weights(self) -> Dict[str, Any]:
        """
        Return only the encoder state_dict for federated server sharing.
        """
        return {k: v.clone() for k, v in self.encoder.state_dict().items()}

    def get_federated_weights(self) -> Dict[str, Dict[str, Any]]:
        """Return every trainable component shared by the FLAME server."""
        return {
            "encoder": {k: v.detach().cpu().clone() for k, v in self.encoder.state_dict().items()},
            "decoder": {k: v.detach().cpu().clone() for k, v in self.decoder.state_dict().items()},
            "proto_head": {k: v.detach().cpu().clone() for k, v in self.proto_head.state_dict().items()},
        }

    def load_federated_weights(self, state: Dict[str, Dict[str, Any]]) -> None:
        """Load a complete FLAME state, while accepting old encoder-only states."""
        if "encoder" not in state:
            self.load_encoder_weights(state)
            return
        self.encoder.load_state_dict(state["encoder"])
        self.decoder.load_state_dict(state["decoder"])
        self.proto_head.load_state_dict(state["proto_head"])

    def load_encoder_weights(self, state_dict: Dict[str, Any]) -> None:
        """Load encoder weights from a state_dict (from federated server)."""
        self.encoder.load_state_dict(state_dict)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get a global image embedding (for fine-tuning / evaluation).
        Uses full image with mean-patch token pooling.

        Args:
            x : (B, C, H, W)
        Returns:
            z : (B, embed_dim)
        """
        if isinstance(x, (list, tuple)):
            x = x[0]
        return self.encoder(x)


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_mae(
    backbone: str = "vit_tiny",
    embed_dim: int = 192,
    mask_ratio: float = 0.75,
    decoder_depth: int = 4,
    image_size: int = 224,
    patch_size: int = 16,
    in_channels: int = 3,
    projection_dim: int = 128,
) -> MaskedAutoencoder:
    """
    Build a full MaskedAutoencoder model with ViT-Tiny backbone.

    Args:
        backbone      : 'vit_tiny'
        embed_dim     : Embedding dimensionality (192 for vit_tiny)
        mask_ratio    : Fraction of patches to mask (default 0.75)
        decoder_depth : Number of Transformer decoder blocks (default 4)
        image_size    : Input image resolution (default 224)
        patch_size    : Patch size in pixels (default 16)
        in_channels   : Number of image channels (default 3)

    Returns:
        MaskedAutoencoder model
    """
    num_patches = (image_size // patch_size) ** 2

    encoder = get_encoder(backbone=backbone, embed_dim=embed_dim)
    decoder = MAEDecoder(
        embed_dim=embed_dim,
        num_patches=num_patches,
        patch_size=patch_size,
        in_channels=in_channels,
        decoder_embed_dim=256,
        decoder_depth=decoder_depth,
        decoder_num_heads=8,
    )
    return MaskedAutoencoder(
        encoder=encoder,
        decoder=decoder,
        mask_ratio=mask_ratio,
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        projection_dim=projection_dim,
    )

