"""
scripts/manual_builder/part11_debugging.py
------------------------------------------
Part XI: Engineering Troubleshooting Runbook — 14 Real Failure Scenarios,
Diagnostic Commands, Root Cause Analysis, and Regression Testing.
"""

from scripts.manual_builder.styles import (
    add_h1, add_h2, add_h3, add_h4, add_body, add_bullet,
    add_code_block, add_callout, add_table, add_horizontal_rule
)


def build_part11(doc):
    add_h1(doc, "Part XI: Engineering Troubleshooting & Diagnostic Runbook")

    add_h2(doc, "11.1 The Senior Engineer's Debugging Protocol")
    add_body(doc, "When deep learning systems fail, junior developers make random edits until errors change. Senior engineers follow a systematic 5-step diagnostic loop:")
    add_bullet(doc, "Inspect the stack trace to determine the exact file, line number, and tensor shape.", bold_prefix="1. Localize Failure: ")
    add_bullet(doc, "Formulate a concrete, falsifiable hypothesis regarding data types, shapes, or state.", bold_prefix="2. Formulate Hypothesis: ")
    add_bullet(doc, "Write a 5-line minimal script reproducing the issue without running the full 100-round loop.", bold_prefix="3. Minimal Reproduction: ")
    add_bullet(doc, "Apply the minimal structural correction.", bold_prefix="4. Targeted Correction: ")
    add_bullet(doc, "Execute automated unit tests (test_smoke.py, research_audit.py) to prevent regressions.", bold_prefix="5. Regression Verification: ")

    add_h2(doc, "11.2 Fourteen Real Production Failure Scenarios & Solutions")

    scenarios = [
        (
            "Scenario 1: Tensor Shape Mismatch in Patchify",
            "RuntimeError: shape '[16, 3, 14, 16, 14, 16]' is invalid for input of size 2408448",
            "An image with non-square or non-224 dimensions (e.g., 256x256 or 512x512) was fed into patchify() without passing through get_base_transform().",
            "Ensure all dataloaders pass raw images through T.Resize((image_size, image_size)) before forward passes.",
            "python -c \"import torch; from src.models.mae import build_mae; m=build_mae(); m.patchify(torch.randn(2,3,224,224))\""
        ),
        (
            "Scenario 2: NaN Loss During Early MAE Pretraining",
            "Loss outputs 'nan' after 15 batches; all model weights become NaN.",
            "Learning rate too high (lr > 5e-3) without gradient norm clipping, or division by zero in masked patch loss normalization.",
            "Verify nn.utils.clip_grad_norm_(train_params, max_norm=1.0) is active; add eps=1e-8 to mask denominator: loss / (mask.sum() + 1e-8).",
            "Run test_smoke.py with artificial large input tensors."
        ),
        (
            "Scenario 3: CUDA Out of Memory (OOM) on Local Workstation",
            "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB",
            "Batch size too large (e.g. batch_size=64) or accumulation of client models in memory without garbage collection.",
            "Reduce ssl.batch_size to 16 in YAML; ensure client models in simulation.py call del hospital_model and torch.cuda.empty_cache().",
            "Monitor VRAM via 'nvidia-smi -l 1' during local execution."
        ),
        (
            "Scenario 4: State Dict Key Mismatch on Checkpoint Load",
            "RuntimeError: Error(s) in loading state_dict for MaskedAutoencoder: Missing key(s)... Unexpected key(s)...",
            "Attempting to load a checkpoint that saved full MAE weights into an isolated ViTTinyEncoder, or prefix mismatch ('encoder.' vs '').",
            "Inspect checkpoint keys: ckpt = torch.load(path); if 'encoder_state_dict' in ckpt: model.load_state_dict(ckpt['encoder_state_dict']).",
            "python -c \"import torch; c=torch.load('experiments/main_experiment/checkpoints/flame_round_099.pt', map_location='cpu'); print(c.keys())\""
        ),
        (
            "Scenario 5: Resume Round Offset Bug",
            "Training restarts from Round 1 even though checkpoint from Round 49 was loaded.",
            "The loop variable in simulation.py starts from 0 rather than start_round = latest_round + 1.",
            "Set start_round = int(latest_ckpt.stem.split('_')[-1]) + 1 and loop range(start_round, config.federated.rounds).",
            "Execute simulation.py --resume and verify log prints 'Continuing from Round 50'."
        ),
        (
            "Scenario 6: K < 2 in Leave-One-Out Adaptation",
            "ValueError: Few-shot adaptation with a trainable prototypical head requires K >= 2 per class.",
            "User configured few_shot_k: 1 in YAML. Leave-one-out requires leaving one sample out while retaining >= 1 prototype sample.",
            "Set few_shot_k >= 2 (recommended K=5 or K=20). For pure K=1, use non-parametric centroid lookup without adaptation.",
            "python -m unittest test_few_shot.py"
        ),
        (
            "Scenario 7: Data Leakage Between Support and Test Cohorts",
            "Fatal ValueError: Shenzhen support and Montgomery test images overlap.",
            "Accidental misconfiguration pointing support and test loaders to the same image directory.",
            "Keep separate dataset paths; verify evaluate_on_montgomery() path set intersection check passes.",
            "python -m unittest test_stage2_leakage.py"
        ),
        (
            "Scenario 8: Split Cache Staleness Across Different Hyperparameters",
            "Simulation uses old 5-hospital split indices after user changed num_hospitals: 10 in YAML.",
            "The splitter loaded indices.npy from disk without verifying if the partition metadata matched runtime settings.",
            "src/datasets/splitter.py implements split_cache_matches() which compares split_metadata.json against current YAML.",
            "python -m unittest test_split_cache.py"
        ),
        (
            "Scenario 9: Integer Buffer Corruption in FedAvg",
            "TypeError: Cannot multiply sequence by non-float of type 'float' or buffer precision loss.",
            "Aggregator multiplied non-floating buffers (e.g., int64 step counters) by float sample weights.",
            "In aggregator.py, check if parameter.dtype.is_floating_point; if False, clone from client 0 without averaging.",
            "python -m unittest test_stage1_architecture.py"
        ),
        (
            "Scenario 10: Single-Class Batch in Evaluation Metric",
            "ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.",
            "Query or test set batch contained only Normal (0) or only TB (1) images.",
            "Use stratified sampling in _sample_kshot() and ensure DataLoader has shuffle=False on evaluation sets.",
            "research_audit.py dataset_audit()"
        ),
        (
            "Scenario 11: Missing or Empty Dataset Directory",
            "SystemExit: NIH dataset at data/raw/NIH is empty.",
            "Images not downloaded or incorrect relative path provided in YAML.",
            "Update data.nih_path in configs/default.yaml to absolute or verified relative path; test via len(NIHDataset(...)) > 0.",
            "python -c \"from src.datasets.loader import NIHDataset; print(len(NIHDataset('data/raw/NIH', limit=10)))\""
        ),
        (
            "Scenario 12: Zero Gradient Flow in FedProx Penalty",
            "AssertionError: Encoder parameters received zero gradients during FedProx loss backprop!",
            "Proximal term was computed on detached parameter copies (.detach() or .data) instead of live parameter tensors.",
            "Compute proximal penalty directly on live module parameters: penalty = penalty + torch.sum((param - ref) ** 2).",
            "python test_smoke.py"
        ),
        (
            "Scenario 13: FastApi Startup Failure Due to Missing Shenzhen Data",
            "RuntimeError: Shenzhen dataset is empty or unavailable (HTTP 503 on /predict).",
            "FastAPI backend started without local access to raw Shenzhen support images needed to compute prototypes.",
            "Provide pre-computed prototype state dict in checkpoint ('proto_head_state_dict') to enable zero-data inference.",
            "curl http://localhost:8000/health"
        ),
        (
            "Scenario 14: Git Merge Conflict in YAML Configuration",
            "error: Your local changes to 'configs/default.yaml' would be overwritten by merge.",
            "Local edits conflicted with remote commits on second/dev.",
            "Stash local edits via 'git stash', pull remote updates via 'git pull second dev', then re-apply via 'git stash pop'.",
            "git status"
        )
    ]

    for title, symptom, cause, fix, diag in scenarios:
        add_h3(doc, title)
        add_body(doc, symptom, bold_prefix="Symptom: ", italic=True)
        add_body(doc, cause, bold_prefix="Root Cause: ")
        add_body(doc, fix, bold_prefix="Engineering Fix: ")
        add_body(doc, diag, bold_prefix="Diagnostic Command: ")

    add_horizontal_rule(doc)
