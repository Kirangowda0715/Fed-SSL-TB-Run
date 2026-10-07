"""Tests for configuration-aware NIH hospital split caching."""

import tempfile
import unittest

from torch.utils.data import TensorDataset
import torch

from src.datasets.splitter import split_nih_to_hospitals, split_cache_matches


class SplitCacheTests(unittest.TestCase):
    def test_cache_requires_exact_configuration(self):
        dataset = TensorDataset(torch.randn(100, 3, 2, 2))
        with tempfile.TemporaryDirectory() as tmp:
            split_nih_to_hospitals(
                dataset,
                num_hospitals=10,
                strategy="non_iid",
                alpha=1.0,
                save_dir=tmp,
                seed=42,
            )

            self.assertTrue(
                split_cache_matches(tmp, 100, 10, "non_iid", 1.0, 42)
            )
            self.assertFalse(
                split_cache_matches(tmp, 100, 10, "non_iid", 0.5, 42)
            )
            self.assertFalse(
                split_cache_matches(tmp, 100, 10, "non_iid", 1.0, 7)
            )
            self.assertFalse(
                split_cache_matches(tmp, 100, 5, "non_iid", 1.0, 42)
            )
            self.assertFalse(
                split_cache_matches(tmp, 101, 10, "non_iid", 1.0, 42)
            )


if __name__ == "__main__":
    unittest.main()
