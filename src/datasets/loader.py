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
import csv
from pathlib import Path
from typing import Optional, Callable, Tuple, List

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# ─── Standard transforms ─────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _metadata_labels(root_dir: Path, image_paths: List[Path]):
    """Resolve optional metadata labels by study ID or filename."""
    candidates = list(root_dir.glob("*_metadata.csv")) + list(root_dir.glob("*metadata*.csv"))
    if not candidates:
        return None, [p.stem for p in image_paths]
    metadata_path = candidates[0]
    with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not rows[0]:
        raise ValueError(f"Metadata file is empty: {metadata_path}")
    fields = {field.lower(): field for field in rows[0]}
    id_field = next((fields[name] for name in ("study_id", "image_id", "filename", "file_name") if name in fields), None)
    label_field = next((fields[name] for name in ("label", "class", "finding", "findings", "diagnosis") if name in fields), None)
    if not id_field or not label_field:
        raise ValueError(f"Metadata must contain an image ID and label column: {metadata_path}")
    mapping = {}
    for row in rows:
        key = Path(row[id_field]).stem
        value = str(row[label_field]).strip().lower()
        if value in {"0", "normal", "negative", "no_tb", "no tuberculosis"}:
            label = 0
        elif value in {"1", "tb", "tuberculosis", "positive", "yes_tb"}:
            label = 1
        else:
            raise ValueError(f"Ambiguous label {row[label_field]!r} in {metadata_path}")
        if key in mapping and mapping[key] != label:
            raise ValueError(f"Conflicting metadata labels for study ID {key}")
        mapping[key] = label
    labels = []
    study_ids = []
    for path in image_paths:
        key = path.stem
        if key not in mapping:
            raise ValueError(f"No metadata label for image {path.name}")
        labels.append(mapping[key])
        study_ids.append(key)
    return labels, study_ids


def _validate_image_ids(image_paths: List[Path], study_ids: List[str]) -> None:
    if len(set(str(path.resolve()) for path in image_paths)) != len(image_paths):
        raise ValueError("Duplicate image paths detected.")
    if len(set(study_ids)) != len(study_ids):
        raise ValueError("Duplicate study/image IDs detected.")


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
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
            all_found.extend(list(self.root_dir.rglob(ext)))
        
        unique = {str(p.resolve()): p for p in all_found}
        all_found = list(unique.values())
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

# ─── Shenzhen TB Dataset ─────────────────────────────────────────────────────

class ShenzhenDataset(Dataset):
    """
    Smart loader for the Shenzhen TB dataset.

    Supported structures:

        TB_Chest_Radiography_Database/
        ├── Normal/
        └── Tuberculosis/

    or:

        root/
        ├── TB/
        └── Normal/

    or:

        root/
        ├── Positive/
        └── Negative/

    Labels:
        Normal        -> 0
        Tuberculosis  -> 1
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
        self.study_ids: List[str] = []

        def _collect_images(src_dir: Path, label: int) -> None:
            """Collect image files recursively and assign a label."""
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
                for p in sorted(src_dir.rglob(ext)):
                    self.image_paths.append(p)
                    self.labels.append(label)

        # ---------------------------------------------------------------
        # Check that root exists
        # ---------------------------------------------------------------
        if not self.root_dir.exists():
            print(f"[ERROR] Shenzhen root does not exist: {self.root_dir}")
            return

        print(f"Scanning Shenzhen dataset in {self.root_dir}...")

        # ---------------------------------------------------------------
        # 1. Actual Shenzhen TB Chest Radiography Database structure
        #
        #    Normal/
        #    Tuberculosis/
        # ---------------------------------------------------------------
        tuberculosis_dir = self.root_dir / "Tuberculosis"
        normal_dir = self.root_dir / "Normal"

        if tuberculosis_dir.exists() and normal_dir.exists():

            print("[Shenzhen] Detected:")
            print(f"  Normal directory       : {normal_dir}")
            print(f"  Tuberculosis directory: {tuberculosis_dir}")

            _collect_images(normal_dir, 0)
            _collect_images(tuberculosis_dir, 1)

        # ---------------------------------------------------------------
        # 2. Alternative TB / Normal structure
        # ---------------------------------------------------------------
        elif (self.root_dir / "TB").exists() and normal_dir.exists():

            print("[Shenzhen] Detected TB/Normal directory structure.")

            _collect_images(self.root_dir / "Normal", 0)
            _collect_images(self.root_dir / "TB", 1)

        # ---------------------------------------------------------------
        # 3. Alternative Positive / Negative structure
        # ---------------------------------------------------------------
        elif (
            (self.root_dir / "Positive").exists()
            and (self.root_dir / "Negative").exists()
        ):

            print("[Shenzhen] Detected Positive/Negative directory structure.")

            _collect_images(self.root_dir / "Negative", 0)
            _collect_images(self.root_dir / "Positive", 1)

        # ---------------------------------------------------------------
        # 4. Filename-based fallback
        # ---------------------------------------------------------------
        else:

            print(
                "[Shenzhen] No recognized class directories found. "
                "Trying filename-based labels..."
            )

            all_imgs = []

            for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
                all_imgs.extend(self.root_dir.rglob(ext))

            for p in sorted(all_imgs):

                name = p.stem

                if name.endswith("_0"):
                    self.image_paths.append(p)
                    self.labels.append(0)

                elif name.endswith("_1"):
                    self.image_paths.append(p)
                    self.labels.append(1)

        # ---------------------------------------------------------------
        # Print final statistics
        # ---------------------------------------------------------------
        unique = {}
        for path, label in zip(self.image_paths, self.labels):
            unique[str(path.resolve())] = (path, label)
        self.image_paths = [item[0] for item in unique.values()]
        self.labels = [item[1] for item in unique.values()]
        metadata_labels, metadata_ids = _metadata_labels(self.root_dir, self.image_paths)
        if metadata_labels is not None:
            if self.labels and metadata_labels != self.labels:
                raise ValueError("Metadata labels disagree with Shenzhen folder labels.")
            self.labels = metadata_labels
        self.study_ids = metadata_ids
        _validate_image_ids(self.image_paths, self.study_ids)
        normal_count = self.labels.count(0)
        tb_count = self.labels.count(1)

        print(f"Shenzhen Dataset: Found {len(self.image_paths)} images.")
        print(f"  Normal       : {normal_count}")
        print(f"  Tuberculosis : {tb_count}")

        if len(self.image_paths) == 0:
            print(
                f"[WARNING] No Shenzhen images found in: "
                f"{self.root_dir}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:

        image_path = self.image_paths[idx]

        try:
            image = Image.open(image_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image, self.labels[idx]

        except Exception as e:
            print(f"[ERROR] Failed to load image {image_path}: {e}")

            # Return a valid tensor in case of a corrupted image
            return torch.zeros(3, 224, 224), self.labels[idx]

    def get_labels(self) -> List[int]:
        return self.labels

    def get_study_id(self, idx: int) -> str:
        return self.study_ids[idx]

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
        self.study_ids: List[str] = []

        def _collect_images(src_dir: Path, label: int) -> None:
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
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
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
                all_imgs.extend(list(self.root_dir.rglob(ext)))
            
            for p in sorted(all_imgs):
                name = p.stem
                if name.endswith("_0"):
                    self.labels.append(0)
                    self.image_paths.append(p)
                elif name.endswith("_1"):
                    self.labels.append(1)
                    self.image_paths.append(p)

        unique = {}
        for path, label in zip(self.image_paths, self.labels):
            unique[str(path.resolve())] = (path, label)
        self.image_paths = [item[0] for item in unique.values()]
        self.labels = [item[1] for item in unique.values()]
        metadata_labels, metadata_ids = _metadata_labels(self.root_dir, self.image_paths)
        if metadata_labels is not None:
            if self.labels and metadata_labels != self.labels:
                raise ValueError("Metadata labels disagree with Montgomery folder labels.")
            self.labels = metadata_labels
        self.study_ids = metadata_ids
        _validate_image_ids(self.image_paths, self.study_ids)
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

    def get_study_id(self, idx: int) -> str:
        return self.study_ids[idx]
