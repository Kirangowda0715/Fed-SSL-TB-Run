"""Tests for the Stage 2 few-shot data-isolation protocol."""

import types
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

import src.client.local_train as local_train


class TrackingShenzhenDataset(Dataset):
    def __init__(self):
        # 4 samples per class: 2 support + 2 evaluation-query samples.
        self.samples = [
            (0, 0.0), (0, 1.0), (0, 2.0), (0, 3.0),
            (1, 4.0), (1, 5.0), (1, 6.0), (1, 7.0),
        ]
        self.accessed = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        self.accessed.append(index)
        label, value = self.samples[index]
        image = torch.full((3, 2, 2), float(value))
        return image, label

    def get_labels(self):
        return torch.tensor([label for label, _ in self.samples], dtype=torch.long)


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(12, 4)

    def forward(self, x):
        return self.proj(x.flatten(1))


class Stage2LeakageTests(unittest.TestCase):
    def test_query_samples_are_not_accessed_before_adaptation(self):
        """Evaluation-query samples must not be read before the adaptation step."""
        dataset = TrackingShenzhenDataset()
        loader = DataLoader(dataset, batch_size=8, shuffle=False)

        config = types.SimpleNamespace(
            model=types.SimpleNamespace(embed_dim=4),
            finetuning=types.SimpleNamespace(
                seed=42,
                few_shot_k=2,
                projection_dim=4,
                freeze_encoder=True,
                lr=1e-3,
                epochs=1,
            ),
            ssl=types.SimpleNamespace(batch_size=4),
        )

        # With seed=42, the exact support indices are determined by _sample_kshot.
        support_idx, query_idx = local_train._sample_kshot(
            dataset.get_labels(), k=2, seed=42
        )
        query_set = set(query_idx.tolist())
        step_checked = {"value": False}

        class TrackingOptimizer:
            def __init__(self, params, lr, weight_decay):
                self.params = list(params)

            def zero_grad(self):
                pass

            def step(self):
                # This is the critical assertion: no evaluation-query image
                # may have been accessed before adaptation completes.
                leaked = query_set.intersection(dataset.accessed)
                if leaked:
                    raise AssertionError(
                        f"Query samples were accessed during adaptation: {sorted(leaked)}"
                    )
                step_checked["value"] = True

        original_optimizer = local_train.AdamW
        local_train.AdamW = TrackingOptimizer
        try:
            local_train.finetune_local(
                hospital_id=1,
                encoder=TinyEncoder(),
                shenzhen_loader=loader,
                config=config,
                device=torch.device("cpu"),
            )
        finally:
            local_train.AdamW = original_optimizer

        self.assertTrue(step_checked["value"])
        self.assertTrue(
            query_set.intersection(dataset.accessed),
            "Query samples should be accessed only during final evaluation.",
        )
        self.assertEqual(
            set(support_idx.tolist()) | query_set,
            set(range(len(dataset))),
        )


if __name__ == "__main__":
    unittest.main()
