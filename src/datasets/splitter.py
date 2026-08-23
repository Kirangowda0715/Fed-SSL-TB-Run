"""
src/datasets/splitter.py
-------------------------
Split the NIH dataset into 5 hospital shards using IID or Non-IID strategy.

Non-IID: Dirichlet(alpha=0.5) distribution → simulates real-world heterogeneity.
IID    : Random equal partition.

Saves per-hospital index lists to data/processed/hospital_{i}/indices.npy
"""

import os
import numpy as np
from pathlib import Path
from typing import List
from torch.utils.data import Dataset


def split_nih_to_hospitals(
    dataset: Dataset,
    num_hospitals: int = 5,
    strategy: str = "non_iid",
    alpha: float = 0.5,
    save_dir: str = "data/processed",
    seed: int = 42,
) -> List[List[int]]:
    """
    Split NIH dataset indices across `num_hospitals` hospitals.

    Args:
        dataset       : NIHDataset (or any Dataset); only __len__ is used.
        num_hospitals : Number of hospital partitions (default 5).
        strategy      : 'iid' or 'non_iid'.
        alpha         : Dirichlet concentration parameter (non-IID only).
                        Lower alpha → more skewed distributions.
        save_dir      : Root directory to save hospital index files.
        seed          : Random seed for reproducibility.

    Returns:
        List of index lists, one per hospital.
        Also saves each list to data/processed/hospital_{i}/indices.npy
    """
    np.random.seed(seed)
    n = len(dataset)
    if num_hospitals < 1:
        raise ValueError("num_hospitals must be at least 1.")
    if n < num_hospitals:
        raise ValueError(
            f"Cannot create {num_hospitals} non-empty hospital splits from {n} samples."
        )
    all_indices = np.arange(n)

    if strategy == "iid":
        hospital_indices = _split_iid(all_indices, num_hospitals)
    elif strategy == "non_iid":
        hospital_indices = _split_non_iid(all_indices, num_hospitals, alpha)
    else:
        raise ValueError(f"Unknown split strategy '{strategy}'. Use 'iid' or 'non_iid'.")

    # Save to disk
    _save_indices(hospital_indices, save_dir)

    # Print distribution summary
    print(f"\n[Splitter] Strategy: {strategy} | Hospitals: {num_hospitals} | Total samples: {n}")
    for i, idx in enumerate(hospital_indices):
        print(f"  Hospital {i+1}: {len(idx):5d} samples  ({100*len(idx)/n:.1f}%)")

    return [idx.tolist() for idx in hospital_indices]


def _split_iid(indices: np.ndarray, num_hospitals: int) -> List[np.ndarray]:
    """Randomly shuffle and divide equally across hospitals."""
    np.random.shuffle(indices)
    return np.array_split(indices, num_hospitals)


def _split_non_iid(
    indices: np.ndarray,
    num_hospitals: int,
    alpha: float,
) -> List[np.ndarray]:
    """
    Dirichlet-based non-IID split.

    Since NIH images are unlabeled for SSL, we partition the raw index space
    using Dirichlet proportions (simulating varying data volumes per hospital,
    which is the primary form of heterogeneity in this unlabeled setting).

    For labeled datasets, you'd also distribute disease categories unevenly.
    """
    n = len(indices)
    np.random.shuffle(indices)

    # Sample proportions from Dirichlet distribution
    proportions = np.random.dirichlet(alpha=np.ones(num_hospitals) * alpha)

    # Reserve one sample per hospital, then distribute the remainder.
    counts = np.ones(num_hospitals, dtype=int)
    counts += np.random.multinomial(n - num_hospitals, proportions)

    # Slice indices
    hospital_indices = []
    start = 0
    for count in counts:
        hospital_indices.append(indices[start : start + count])
        start += count

    return hospital_indices


def _save_indices(hospital_indices: List[np.ndarray], save_dir: str) -> None:
    """Save each hospital's index array to disk as .npy file."""
    save_root = Path(save_dir)
    for i, idx in enumerate(hospital_indices):
        hospital_dir = save_root / f"hospital_{i+1}"
        hospital_dir.mkdir(parents=True, exist_ok=True)
        out_path = hospital_dir / "indices.npy"
        np.save(str(out_path), idx)
        print(f"  Saved: {out_path}")


def load_hospital_indices(hospital_id: int, save_dir: str = "data/processed") -> List[int]:
    """
    Load pre-saved hospital indices from disk.

    Args:
        hospital_id : 1-indexed hospital ID (1..5)
        save_dir    : Root processed data directory

    Returns:
        List of dataset indices for this hospital.
    """
    path = Path(save_dir) / f"hospital_{hospital_id}" / "indices.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved indices found at {path}. "
            f"Run split_nih_to_hospitals() first."
        )
    return np.load(str(path)).tolist()
