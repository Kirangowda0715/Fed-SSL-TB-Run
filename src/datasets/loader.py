"""
src/datasets/loader.py
----------------------
PyTorch Dataset classes for NIH ChestX-ray14, Shenzhen TB, and Montgomery TB.

NIH     → unlabeled (SSL pretraining) with two-view augmentation
Shenzhen → binary TB/Normal labels (few-shot fine-tuning)
Montgomery → binary TB/Normal labels (held-out test set)
"""

import os
import glob
from pathlib import Path
from typing import Optional, Callable, Tuple, List

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# ─── Standard transforms ─────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_base_transform(image_size: int = 224) -> T.Compose:
    """Standard chest X-ray augmentation for supervised/fine-tuning."""
    return T.Compose([
        T.Resize((image_size + 32, image_size + 32)),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(),
        T.Grayscale(num_output_channels=3),   # X-rays are grayscale → 3-ch
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform(image_size: int = 224) -> T.Compose:
    """Deterministic transform for evaluation (no random ops)."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class TwoViewTransform:
    """
    Wraps any transform and applies it twice to the same image,
    returning (view1, view2) for SSL contrastive / MAE pre-training.
    """
    def __init__(self, base_transform: Callable):
        self.transform = base_transform

    def __call__(self, img) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.transform(img), self.transform(img)


# ─── NIH ChestX-ray14 Dataset ────────────────────────────────────────────────

class NIHDataset(Dataset):
    """
    Loads images from the NIH Chest X-ray dataset for Self-Supervised Learning.
    Supports recursive scanning across images_001, images_002, etc.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        image_size: int = 224,
        limit: Optional[int] = 5000,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform or get_base_transform(image_size)
        self.image_paths: List[Path] = []

        print(f"Scanning NIH dataset in {self.root_dir}...")
        
        # Search recursively for all images
        all_found = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            all_found.extend(list(self.root_dir.rglob(ext)))
        
        all_found.sort()
        if limit and len(all_found) > limit:
            step = len(all_found) // limit
            self.image_paths = all_found[::step][:limit]
            print(f"Limited NIH to {len(self.image_paths)} images for performance.")
        else:
            self.image_paths = all_found
            print(f"Found {len(self.image_paths)} NIH images.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        try:
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {e}")
            return torch.zeros(3, 224, 224)


# ─── Shenzhen TB Dataset ─────────────────────────────────────────────────────

class ShenzhenDataset(Dataset):
    """
    Smart loader for the Shenzhen TB dataset.
    Supports either folder-based labels or filename suffix labels.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        image_size: int = 224,
        split: str = "all",
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform or get_base_transform(image_size)
        self.image_paths: List[Path] = []
        self.labels: List[int] = []

        def _collect_images(src_dir: Path, label: int) -> None:
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for p in sorted(src_dir.rglob(ext)):
                    self.image_paths.append(p)
                    self.labels.append(label)

        tb_dir = self.root_dir / "TB"
        normal_dir = self.root_dir / "Normal"
        pos_dir = self.root_dir / "Positive"
        neg_dir = self.root_dir / "Negative"

        if tb_dir.exists() and normal_dir.exists():
            _collect_images(tb_dir, 1)
            _collect_images(normal_dir, 0)
        elif pos_dir.exists() and neg_dir.exists():
            _collect_images(pos_dir, 1)
            _collect_images(neg_dir, 0)
        else:
            all_imgs = []
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                all_imgs.extend(list(self.root_dir.rglob(ext)))
            
            for p in sorted(all_imgs):
                name = p.stem
                if name.endswith("_0"):
                    self.labels.append(0)
                    self.image_paths.append(p)
                elif name.endswith("_1"):
                    self.labels.append(1)
                    self.image_paths.append(p)

        print(f"Shenzhen Dataset: Found {len(self.image_paths)} images.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

    def get_labels(self) -> List[int]:
        return self.labels


# ─── Montgomery TB Dataset ────────────────────────────────────────────────────

class MontgomeryDataset(Dataset):
    """
    Smart loader for the Montgomery TB dataset.
    Supports either folder-based labels or filename suffix labels.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        image_size: int = 224,
        split: str = "all",
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform or get_base_transform(image_size)
        self.image_paths: List[Path] = []
        self.labels: List[int] = []

        def _collect_images(src_dir: Path, label: int) -> None:
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for p in sorted(src_dir.rglob(ext)):
                    self.image_paths.append(p)
                    self.labels.append(label)

        tb_dir = self.root_dir / "TB"
        normal_dir = self.root_dir / "Normal"
        pos_dir = self.root_dir / "Positive"
        neg_dir = self.root_dir / "Negative"

        if tb_dir.exists() and normal_dir.exists():
            _collect_images(tb_dir, 1)
            _collect_images(normal_dir, 0)
        elif pos_dir.exists() and neg_dir.exists():
            _collect_images(pos_dir, 1)
            _collect_images(neg_dir, 0)
        else:
            all_imgs = []
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                all_imgs.extend(list(self.root_dir.rglob(ext)))
            
            for p in sorted(all_imgs):
                name = p.stem
                if name.endswith("_0"):
                    self.labels.append(0)
                    self.image_paths.append(p)
                elif name.endswith("_1"):
                    self.labels.append(1)
                    self.image_paths.append(p)

        print(f"Montgomery Dataset: Found {len(self.image_paths)} images.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

    def get_labels(self) -> List[int]:
        return self.labels
