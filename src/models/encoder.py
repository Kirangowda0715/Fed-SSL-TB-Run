"""
src/models/encoder.py
---------------------
Encoder backbone factory: ResNet50 or ViT-Small.

Only the encoder is shared across hospitals in the federated setting.
The decoder stays local (privacy + efficiency).
"""

import torch
import torch.nn as nn
from typing import Tuple


# ─── ResNet50 Encoder ─────────────────────────────────────────────────────────

class ResNet50Encoder(nn.Module):
    """
    ResNet50 backbone with the final FC layer replaced by a
    linear projection head: Linear(2048 → embed_dim).
    """

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        import torchvision.models as models

        backbone = models.resnet50(weights=None)  # pretrained=False equivalent

        # Remove the original avgpool + fc; we keep everything up to layer4
        self.feature_extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.avgpool = backbone.avgpool           # Global average pool → (B, 2048, 1, 1)
        self.projection = nn.Linear(2048, embed_dim)

        self.embed_dim = embed_dim
        self.feature_dim = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, H, W) image tensor
        Returns:
            z : (B, embed_dim) embedding
        """
        feat = self.feature_extractor(x)          # (B, 2048, 7, 7)
        feat = self.avgpool(feat)                  # (B, 2048, 1, 1)
        feat = feat.flatten(1)                     # (B, 2048)
        z = self.projection(feat)                  # (B, embed_dim)
        return z


# ─── ViT-Small Encoder ────────────────────────────────────────────────────────

class ViTSmallEncoder(nn.Module):
    """
    ViT-Small (patch16, 224) from timm.
    Head replaced with Linear(384 → embed_dim).
    """

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required for ViT-Small encoder. pip install timm")

        self.vit = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=0,  # Remove classification head → returns CLS token (384-dim)
        )
        vit_dim = self.vit.embed_dim   # 384 for vit_small
        self.projection = nn.Linear(vit_dim, embed_dim)

        self.embed_dim = embed_dim
        self.feature_dim = vit_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, H, W) image tensor (must be 224×224 for vit_small_patch16_224)
        Returns:
            z : (B, embed_dim) embedding
        """
        feat = self.vit(x)             # (B, 384) — CLS token representation
        z = self.projection(feat)      # (B, embed_dim)
        return z


# ─── ViT-Tiny Encoder (Primary Lightweight Backbone) ─────────────────────────

class ViTTinyEncoder(nn.Module):
    """
    ViT-Tiny (patch16, 224) from timm.
    Primary lightweight backbone for privacy-preserving FedSSL TB detection.
    
    Dimensions:
        - Input  : (B, 3, 224, 224)
        - Latent : (B, 192) embedding
    """

    def __init__(self, embed_dim: int = 192):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required for ViT-Tiny encoder. Run: pip install timm")

        self.vit = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            num_classes=0,  # Returns sequence or pooled features
        )
        vit_dim = self.vit.embed_dim   # 192 for vit_tiny
        self.projection = nn.Linear(vit_dim, embed_dim)

        self.embed_dim = embed_dim
        self.feature_dim = vit_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract global feature embedding using mean pooling across patch tokens.
        This matches the representation learned during MAE pre-training.

        Args:
            x : (B, C, H, W) image tensor
        Returns:
            z : (B, embed_dim) embedding
        """
        # Patch embed + positional embedding over 196 patches
        tokens = self.vit.patch_embed(x)
        tokens = tokens + self.vit.pos_embed[:, 1:, :]  # 196 patch tokens
        tokens = self.vit.pos_drop(tokens)

        # Pass patch tokens through Transformer blocks
        for block in self.vit.blocks:
            tokens = block(tokens)
        tokens = self.vit.norm(tokens)

        # Mean patch token pooling -> projection
        pooled = tokens.mean(dim=1)  # (B, 192)
        z = self.projection(pooled)   # (B, embed_dim)
        return z


# ─── Factory ─────────────────────────────────────────────────────────────────

def get_encoder(backbone: str = "vit_tiny", embed_dim: int = 192) -> nn.Module:
    """
    Create and return the ViT-Tiny encoder backbone.

    Args:
        backbone  : 'vit_tiny' (recommended lightweight backbone)
        embed_dim : Output embedding dimensionality (default 192)
    """
    return ViTTinyEncoder(embed_dim=embed_dim)

