"""
scripts/manual_builder/part4_trace.py
-------------------------------------
Part IV: Terminal-to-Result Execution Trace: Step-by-Step Runtime Lifecycle,
Single-Batch Worked Tensor Transformations, and Federated Round Trace.
"""

from scripts.manual_builder.styles import (
    add_h1, add_h2, add_h3, add_h4, add_body, add_bullet,
    add_code_block, add_callout, add_table, add_horizontal_rule
)


def build_part4(doc):
    add_h1(doc, "Part IV: Follow an Execution from Terminal to Result")

    add_h2(doc, "4.1 The 21-Step End-to-End Execution Pipeline")
    add_body(doc, "When an engineer launches federated pretraining from the terminal with the standard command:")

    add_code_block(doc, "python src/federated/simulation.py --config configs/default.yaml --federated.rounds=30")

    add_body(doc, "The system executes the following 21 sequential operations across client and server modules:")

    steps_headers = ["Step", "Operational Stage", "Executing File & Function", "Key Inputs / Outputs & State Mutation"]
    steps_rows = [
        ["1", "CLI Argument Parsing", "simulation.py: pre_parser", "Extracts flags (--dry-run, --parallel, --resume)."],
        ["2", "Config Loading & Overrides", "src/utils/config.py: load_config()", "Loads default.yaml; applies --federated.rounds=30 override into SimpleNamespace."],
        ["3", "Configuration Validation", "src/utils/config.py: _validate()", "Asserts presence of required sections: data, model, ssl, federated, finetuning, etc."],
        ["4", "Deterministic Seeding", "src/utils/reproducibility.py: seed_everything()", "Sets seeds for torch, torch.cuda, np.random, and random.seed(42)."],
        ["5", "Device Enumeration", "simulation.py: _get_training_devices()", "Detects CUDA availability; allocates server device and per-worker GPU devices."],
        ["6", "Dataset Discovery", "src/datasets/loader.py: NIHDataset.__init__()", "Scans data/raw/NIH recursively for images; limits to limit_samples (500-20000)."],
        ["7", "Transform Construction", "src/datasets/loader.py: get_base_transform()", "Builds torchvision Compose: Resize(256), RandomCrop(224), Flip, Grayscale(3), Normalize."],
        ["8", "Hospital Shard Splitting", "src/datasets/splitter.py: split_nih_to_hospitals()", "Checks cache; computes Dirichlet quantity-skew indices; saves to data/processed/."],
        ["9", "DataLoader Instantiation", "simulation.py: DataLoader(Subset(...))", "Creates 10 hospital DataLoaders with batch_size=16, pin_memory=True, shuffle=True."],
        ["10", "Server & Model Init", "src/server/server.py: initialize_global_model()", "Instantiates central MaskedAutoencoder with ViT-Tiny backbone and MAEDecoder."],
        ["11", "Resume Checkpoint Check", "simulation.py: server.load_checkpoint()", "If --resume flag set, scans for flame_round_*.pt; sets start_round = latest + 1."],
        ["12", "Round Broadcast", "src/server/server.py: broadcast()", "Extracts global encoder and decoder state_dicts; returns nested dictionary."],
        ["13", "Client Model Clone", "simulation.py: copy.deepcopy(global_model)", "Spawns isolated client model copies in memory; ensures strict client independence."],
        ["14", "Local Training Epochs", "src/client/ssl_train.py: ssl_local_train()", "Loops over local NIH batches; executes MAE forward pass to get reconstruction loss."],
        ["15", "FedProx Penalty Addition", "src/client/ssl_train.py: _fedprox_penalty()", "Computes (mu/2) * ||w_local - w_global||^2 on live parameters; adds to total loss."],
        ["16", "Backprop & Gradient Clip", "ssl_train.py: loss.backward()", "Calculates autograd gradients; clips grad norm to 1.0; executes optimizer.step()."],
        ["17", "Weight Collection", "simulation.py: federated_weights_list", "Gathers trained local encoder + decoder state_dicts and sample counts from 10 nodes."],
        ["18", "Server Aggregation", "src/server/aggregator.py: fedavg() / fedprox()", "Computes sample-weighted average: w_global = sum(n_k / N * w_k) for all parameters."],
        ["19", "Global Model Update", "src/server/server.py: update_global_model()", "Loads aggregated weights into server global model; preserves non-floating buffers."],
        ["20", "Checkpoint Persistence", "src/server/server.py: save_checkpoint()", "Saves experiments/checkpoints/flame_round_XXX.pt containing full state_dicts and metadata."],
        ["21", "Telemetry Logging", "simulation.py: RoundLogger.log() & save()", "Appends round statistics; writes experiments/logs/training_log.json for dashboard."]
    ]
    add_table(doc, steps_headers, steps_rows, [0.6, 1.8, 2.0, 2.1])

    add_h2(doc, "4.2 Worked Execution Trace for One Representative Batch")
    add_body(doc, "To understand exactly what PyTorch computes at the tensor level, let us trace one single batch of 16 chest X-ray images through ssl_local_train():")

    batch_trace = (
        "1. Input Batch Tensor [imgs]:\n"
        "   Shape: (16, 3, 224, 224) | Dtype: torch.float32 | Value Range: Normalized ~[-2.1, +2.6]\n"
        "\n"
        "2. Ground Truth Patch Extraction [patchify(imgs)]:\n"
        "   Formula: h = w = 224 // 16 = 14 patches per axis -> N = 14 * 14 = 196 total patches\n"
        "   Patch Pixel Dim: P * P * C = 16 * 16 * 3 = 768 pixels per patch\n"
        "   Target Tensor Shape: (16, 196, 768)\n"
        "\n"
        "3. Tokenization & Positional Embedding [_tokenize(imgs)]:\n"
        "   Linear Patch Projection: (16, 3, 224, 224) -> (16, 196, 192)\n"
        "   Add Positional Embeddings: tokens = tokens + pos_embed[:, 1:, :] -> (16, 196, 192)\n"
        "\n"
        "4. Random Masking [random_masking(tokens, mask_ratio=0.80)]:\n"
        "   Total Patches N = 196\n"
        "   Keep Ratio = 1.0 - 0.80 = 0.20 -> num_keep = int(196 * 0.20) = 39 patches\n"
        "   Masked Patches = 196 - 39 = 157 patches (80% hidden)\n"
        "   Output Tensors:\n"
        "     - visible_tokens: (16, 39, 192)  <-- Only these enter the ViT encoder!\n"
        "     - mask:           (16, 196)       <-- Binary: 1 = masked, 0 = visible\n"
        "     - ids_restore:    (16, 196)       <-- Indices to reconstruct original 14x14 order\n"
        "\n"
        "5. Encoder Forward Pass [_encode_visible(visible_tokens)]:\n"
        "   12 Transformer Blocks execute Multi-Head Self-Attention on 39 tokens only:\n"
        "   encoded Shape: (16, 39, 192)  <-- Enormous 5x compute savings over processing 196 tokens!\n"
        "\n"
        "6. Decoder Reconstruction [decoder(encoded, ids_restore)]:\n"
        "   Projection to Decoder Dim: Linear(192 -> 256) -> (16, 39, 256)\n"
        "   Insert Learnable Mask Tokens: Concat 157 mask tokens -> (16, 196, 256)\n"
        "   Unshuffle to Original Order: torch.gather(..., dim=1, index=ids_restore) -> (16, 196, 256)\n"
        "   Add Decoder Positional Embeddings: (16, 196, 256)\n"
        "   4 Transformer Decoder Blocks: (16, 196, 256)\n"
        "   Linear Head: Linear(256 -> 768) -> pred Shape: (16, 196, 768)\n"
        "\n"
        "7. Loss Calculation [_mae_loss(pred, target, mask)]:\n"
        "   Element-wise Squared Error: (pred - target)^2 -> (16, 196, 768)\n"
        "   Mean over pixels per patch: (16, 196)\n"
        "   Masked Patch Filter: (diff * mask).sum() / (mask.sum() + 1e-8)\n"
        "   Scalar MAE Loss: e.g., 1.3150\n"
        "\n"
        "8. FedProx Proximal Regularization [_fedprox_penalty]:\n"
        "   Sum over all encoder + decoder params: sum((param - global_ref)^2)\n"
        "   Total Batch Loss: loss = 1.3150 + (0.01 / 2.0) * proximal_term = 1.3185\n"
        "\n"
        "9. Backpropagation & Optimizer Step:\n"
        "   loss.backward() -> Computes gradients for all encoder and decoder parameters\n"
        "   clip_grad_norm_(params, max_norm=1.0) -> Prevents exploding gradients in ViT\n"
        "   optimizer.step() -> AdamW updates parameter tensors with weight_decay=0.05"
    )
    add_code_block(doc, batch_trace)

    add_horizontal_rule(doc)
