import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.reproducibility import seed_everything
from src.datasets.loader import ShenzhenDataset, MontgomeryDataset, get_eval_transform
from src.models.encoder import get_encoder
from src.client.local_train import finetune_local, evaluate_on_montgomery


def main():
    config = load_config()
    config.finetuning.few_shot_k = 20
    config.finetuning.seed = 42
    config.finetuning.freeze_encoder = True
    config.finetuning.projection_dim = 128
    config.finetuning.epochs = 100
    config.ssl.batch_size = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path("experiments/main_experiment/checkpoints")
    results_dir = Path("experiments/main_experiment/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)

    transform = get_eval_transform(config.data.image_size)
    shenzhen = ShenzhenDataset(
        config.data.shenzhen_path, transform=transform,
        image_size=config.data.image_size
    )
    montgomery = MontgomeryDataset(
        config.data.montgomery_path, transform=transform,
        image_size=config.data.image_size
    )

    shenzhen_loader = DataLoader(
        shenzhen, batch_size=4, shuffle=False, num_workers=0
    )
    montgomery_loader = DataLoader(
        montgomery, batch_size=4, shuffle=False, num_workers=0
    )

    summary = {}

    for round_num in (49, 99):
        seed_everything(42)
        checkpoint_path = checkpoint_dir / f"flame_round_{round_num:03d}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if checkpoint.get("round") != round_num:
            raise ValueError(f"Checkpoint round mismatch: {checkpoint_path}")

        encoder = get_encoder(config.model.backbone, config.model.embed_dim)
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        encoder.to(device)

        print(f"\n{'=' * 55}")
        print(f"Evaluating Stage 1 checkpoint: round {round_num}")
        print(f"Device: {device}")
        print(f"{'=' * 55}")

        head, shenzhen_metrics = finetune_local(
            0, encoder, shenzhen_loader, config, device
        )
        montgomery_metrics = evaluate_on_montgomery(
            encoder, head, montgomery_loader, config=config, device=device
        )

        def clean_metrics(metrics):
            return {
                key: value
                for key, value in metrics.items()
                if key not in ("support_paths", "support_study_ids")
            }

        result = {
            "checkpoint_round": round_num,
            "checkpoint": str(checkpoint_path),
            "few_shot_k_per_class": 20,
            "seed": 42,
            "freeze_encoder": True,
            "projection_dim": 128,
            "adaptation_epochs": 100,
            "shenzhen_image_count": len(shenzhen),
            "shenzhen_query_count": len(shenzhen) - 40,
            "montgomery_image_count": len(montgomery),
            "shenzhen_query_metrics": clean_metrics(shenzhen_metrics),
            "montgomery_test_metrics": clean_metrics(montgomery_metrics),
            "support_study_ids": shenzhen_metrics.get("support_study_ids", []),
        }

        output_path = results_dir / f"stage2_k20_round{round_num:03d}_controlled.json"
        output_path.write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        summary[str(round_num)] = result
        print("Shenzhen query metrics:", result["shenzhen_query_metrics"])
        print("Montgomery metrics:", result["montgomery_test_metrics"])
        print("Saved:", output_path)

        del encoder, head, checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ids49 = summary["49"]["support_study_ids"]
    ids99 = summary["99"]["support_study_ids"]
    if ids49 != ids99:
        raise RuntimeError("Support sample mismatch between checkpoints.")

    comparison = {}
    for dataset_key in ("shenzhen_query_metrics", "montgomery_test_metrics"):
        comparison[dataset_key] = {}
        metric_keys = (
            "auc", "accuracy", "sensitivity", "specificity",
            "f1", "balanced_accuracy"
        )
        for metric in metric_keys:
            comparison[dataset_key][metric] = {
                "round_49": summary["49"][dataset_key].get(metric),
                "round_99": summary["99"][dataset_key].get(metric),
            }

    comparison_path = results_dir / "round49_vs_round99_k20_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    print("\n=== ROUND 49 VS ROUND 99 COMPARISON ===")
    print(json.dumps(comparison, indent=2))
    print("\nSaved comparison:", comparison_path)


if __name__ == "__main__":
    main()
