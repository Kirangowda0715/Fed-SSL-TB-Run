"""
src/datasets/splitter.py
-------------------------
Reproducible IID / non-IID partitioning of the unlabeled NIH SSL dataset.

Non-IID uses a Dirichlet distribution over hospital sample counts. Because
NIH labels are not used in Stage 1, this represents quantity/statistical
heterogeneity rather than confirmed pathology-prevalence heterogeneity.
"""

import json
import numpy as np
from pathlib import Path
from typing import List
from torch.utils.data import Dataset


_METADATA_NAME = "split_metadata.json"


def _metadata_path(save_dir: str) -> Path:
    return Path(save_dir) / _METADATA_NAME


def _expected_metadata(
    n: int,
    num_hospitals: int,
    strategy: str,
    alpha: float,
    seed: int,
) -> dict:
    return {
        "dataset_size": int(n),
        "num_hospitals": int(num_hospitals),
        "strategy": str(strategy),
        "alpha": float(alpha),
        "seed": int(seed),
    }


def split_cache_matches(
    save_dir: str,
    n: int,
    num_hospitals: int,
    strategy: str,
    alpha: float,
    seed: int,
) -> bool:
    """Return True only when cached indices were generated for this exact split."""
    path = _metadata_path(save_dir)
    if not path.exists():
        return False
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata == _expected_metadata(
        n, num_hospitals, strategy, alpha, seed
    )


def split_nih_to_hospitals(
    dataset: Dataset,
    num_hospitals: int = 5,
    strategy: str = "non_iid",
    alpha: float = 0.5,
    save_dir: str = "data/processed",
    seed: int = 42,
) -> List[List[int]]:
    """
    Split NIH dataset indices across hospitals and persist configuration metadata.
    """
    strategy = str(strategy).lower()
    if strategy not in {"iid", "non_iid"}:
        raise ValueError(f"Unknown split strategy '{strategy}'. Use 'iid' or 'non_iid'.")
    if alpha <= 0:
        raise ValueError("alpha must be > 0.")
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
    else:
        hospital_indices = _split_non_iid(all_indices, num_hospitals, alpha)

    _save_indices(
        hospital_indices,
        save_dir,
        metadata=_expected_metadata(n, num_hospitals, strategy, alpha, seed),
    )

    print(
        f"\n[Splitter] Strategy: {strategy} | Hospitals: {num_hospitals} | "
        f"Alpha: {alpha} | Seed: {seed} | Total samples: {n}"
    )
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
    """Dirichlet quantity-skew split for unlabeled NIH images."""
    n = len(indices)
    np.random.shuffle(indices)
    proportions = np.random.dirichlet(alpha=np.ones(num_hospitals) * alpha)

    counts = np.ones(num_hospitals, dtype=int)
    counts += np.random.multinomial(n - num_hospitals, proportions)

    hospital_indices = []
    start = 0
    for count in counts:
        hospital_indices.append(indices[start : start + count])
        start += count
    return hospital_indices


def _save_indices(
    hospital_indices: List[np.ndarray],
    save_dir: str,
    metadata: dict,
) -> None:
    """Save hospital indices and the exact split configuration."""
    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    for i, idx in enumerate(hospital_indices):
        hospital_dir = save_root / f"hospital_{i+1}"
        hospital_dir.mkdir(parents=True, exist_ok=True)
        out_path = hospital_dir / "indices.npy"
        np.save(str(out_path), idx)
        print(f"  Saved: {out_path}")

    _metadata_path(save_dir).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"  Saved: {_metadata_path(save_dir)}")


def load_hospital_indices(hospital_id: int, save_dir: str = "data/processed") -> List[int]:
    """Load a saved 1-indexed hospital partition."""
    path = Path(save_dir) / f"hospital_{hospital_id}" / "indices.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved indices found at {path}. Run split_nih_to_hospitals() first."
        )
    return np.load(str(path)).tolist()
