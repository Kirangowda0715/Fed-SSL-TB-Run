"""
scripts/manual_builder/part2_repo_map.py
----------------------------------------
Part II: Complete Engineering Repository Map, Module Specifications,
Call Graphs, and Near-Line-by-Line Walkthroughs.
"""

from scripts.manual_builder.styles import (
    add_h1, add_h2, add_h3, add_h4, add_body, add_bullet,
    add_code_block, add_callout, add_table, add_horizontal_rule
)


def build_part2(doc):
    add_h1(doc, "Part II: Engineering Repository Map & Module Walkthroughs")

    add_h2(doc, "2.1 Repository Directory Topology")
    add_body(doc, "The repository follows a clean, modular deep-learning architecture separating configuration, dataset ingestion, model architectures, federated client/server logic, and experimental tracking.")

    tree_str = (
        "Federated-SSL/\n"
        "├── configs/                         # YAML hyperparameter configurations\n"
        "│   ├── default.yaml                 # 10-hospital non-IID default configuration (alpha=1.0, 30 rounds)\n"
        "│   ├── kaggle_suhas_100round.yaml   # 100-round GPU production configuration for Kaggle\n"
        "│   └── stage2_30round_k20.yaml      # Few-shot Stage 2 adaptation configuration\n"
        "├── data/                            # Raw images and generated partitions\n"
        "│   ├── raw/                         # Source datasets (NIH, Shenzhen, Montgomery)\n"
        "│   └── processed/                   # Cached non-IID hospital split indices (hospital_1 .. hospital_10)\n"
        "├── experiments/                     # Experimental artifacts, checkpoints, and telemetry\n"
        "│   ├── main_experiment/             # 100-round run checkpoints (flame_round_000.pt .. 099.pt)\n"
        "│   ├── ablation_studies/            # Controlled studies (few_shot_k, frozen_vs_unfrozen, round49_vs_99)\n"
        "│   └── ablation_30_round/           # 30-round baseline artifacts and calibrated telemetry\n"
        "├── src/                             # Core Python source code\n"
        "│   ├── client/                      # Local hospital training logic\n"
        "│   │   ├── ssl_train.py             # Stage 1 MAE local training with FedProx penalty\n"
        "│   │   ├── local_train.py           # Stage 2 few-shot adaptation & Stage 3 Montgomery evaluation\n"
        "│   │   └── flame_local_train.py     # Joint MAE + Prototypical training loop\n"
        "│   ├── datasets/                    # Medical dataset loading and partitioning\n"
        "│   │   ├── loader.py                # PyTorch Dataset classes (NIH, Shenzhen, Montgomery)\n"
        "│   │   └── splitter.py              # Dirichlet quantity-skew partitioning & cache validation\n"
        "│   ├── federated/                   # Orchestration and simulation harness\n"
        "│   │   └── simulation.py            # Master simulation loop across rounds and clients\n"
        "│   ├── models/                      # Deep neural network architectures\n"
        "│   │   ├── encoder.py               # Vision Transformer backbone factory (ViT-Tiny, ViT-Small, ResNet50)\n"
        "│   │   ├── decoder.py               # Lightweight MAE Transformer decoder\n"
        "│   │   ├── mae.py                   # Masked Autoencoder orchestrator (patchify, masking, loss)\n"
        "│   │   └── proto_head.py            # Prototypical metric projection head & cosine classifier\n"
        "│   ├── server/                      # Federated aggregation and server state\n"
        "│   │   ├── aggregator.py            # FedAvg and FedProx mathematical aggregation\n"
        "│   │   └── server.py                # Central coordinator (broadcast, aggregate, checkpoint)\n"
        "│   ├── utils/                       # Configuration, metrics, and reproducibility\n"
        "│   │   ├── config.py                # Dot-access SimpleNamespace YAML parser with CLI overrides\n"
        "│   │   ├── metrics.py               # ROC-AUC, Accuracy, Sensitivity, Specificity, F1 calculations\n"
        "│   │   └── reproducibility.py       # Seed synchronization for PyTorch, NumPy, and Python random\n"
        "│   └── web/                         # Real-time telemetry dashboard & inference service\n"
        "│       ├── api.py                   # FastAPI REST API (endpoints: /predict, /status, /metrics)\n"
        "│       └── frontend/                # React / Vite web console\n"
        "├── research_audit.py                # Automated contract and data leakage diagnostic harness\n"
        "├── test_smoke.py                    # End-to-end smoke test validating forward/backward autograd\n"
        "├── test_stage1_architecture.py      # Unit tests for Stage 1 pure SSL federated contracts\n"
        "└── test_stage2_leakage.py           # Unit tests confirming zero overlap between support and test sets"
    )
    add_code_block(doc, tree_str)

    add_h2(doc, "2.2 Module Specification Table")
    add_body(doc, "Every core module has a strictly bounded responsibility in the runtime pipeline:")

    mod_headers = ["Module Path", "Primary Classes / Functions", "Direct Caller", "Outputs & Side Effects"]
    mod_rows = [
        ["src/models/encoder.py", "ViTTinyEncoder, get_encoder()", "src/models/mae.py", "Produces 192-dim global embeddings via mean patch token pooling."],
        ["src/models/decoder.py", "MAEDecoder, _get_sinusoidal_pos_embed()", "src/models/mae.py", "Reconstructs 768-dim pixel vectors for all 196 image patches."],
        ["src/models/mae.py", "MaskedAutoencoder, build_mae()", "src/server/server.py", "Computes MSE loss on masked patches (80% ratio); returns encoder weights."],
        ["src/models/proto_head.py", "PrototypicalHead, compute_prototypes()", "src/client/local_train.py", "Computes cosine logits scaled by tau=10.0 against class centroids."],
        ["src/client/ssl_train.py", "ssl_local_train(), _fedprox_penalty()", "src/federated/simulation.py", "Performs AdamW optimization with proximal penalty; returns state_dicts."],
        ["src/client/local_train.py", "finetune_local(), evaluate_on_montgomery()", "Standalone / evaluation scripts", "Runs leave-one-out support adaptation; evaluates Montgomery metrics."],
        ["src/server/aggregator.py", "fedavg(), fedprox()", "src/server/server.py", "Computes sample-weighted average of state_dicts; preserves non-float buffers."],
        ["src/server/server.py", "FederatedServer, broadcast(), save_checkpoint()", "src/federated/simulation.py", "Maintains global model state; saves flame_round_XXX.pt checkpoints."],
        ["src/datasets/loader.py", "NIHDataset, ShenzhenDataset, MontgomeryDataset", "simulation.py / local_train.py", "Loads images, applies transforms, extracts labels/study IDs, prevents leakage."],
        ["src/datasets/splitter.py", "split_nih_to_hospitals(), split_cache_matches()", "src/federated/simulation.py", "Generates and validates Dirichlet quantity-skew partitions across hospitals."],
        ["src/utils/config.py", "load_config(), _apply_override()", "All CLI entry points", "Parses YAML into nested SimpleNamespace; handles --federated.rounds=30."],
        ["src/utils/metrics.py", "evaluate(), format_metrics()", "Evaluation modules & API", "Calculates ROC-AUC, accuracy, sensitivity, specificity, and F1-score."],
        ["src/web/api.py", "FastAPI app, /predict, /status, /metrics", "Uvicorn ASGI runner", "Serves real-time inference and telemetry to the React dashboard."]
    ]
    add_table(doc, mod_headers, mod_rows, [1.5, 1.8, 1.6, 1.6])

    add_h2(doc, "2.3 Module Dependency & Runtime Call Graphs")
    add_body(doc, "Understanding how components connect at runtime is critical to tracing bugs and explaining system behavior in technical interviews.")

    dep_graph = (
        "                    [simulation.py (main)]                                       \n"
        "                       |       |       |                                        \n"
        "         +-------------+       |       +---------------+                        \n"
        "         v                     v                       v                        \n"
        "  [load_config()]    [FederatedServer]        [split_nih_to_hospitals()]        \n"
        "         |                     |                       |                        \n"
        "         |                     v                       v                        \n"
        "         |              [build_mae()]             [NIHDataset]                  \n"
        "         |                 /       \\                   |                        \n"
        "         |       [ViTTinyEncoder] [MAEDecoder]   [hospital_loaders]             \n"
        "         |                     |                       |                        \n"
        "         +---------------------+-----------------------+                        \n"
        "                               |                                                \n"
        "                               v (per round, per hospital)                      \n"
        "                      [ssl_local_train()]                                       \n"
        "                               |                                                \n"
        "                        +------+------+                                         \n"
        "                        v             v                                         \n"
        "                 [model(imgs)]  [_fedprox_penalty()]                            \n"
        "                        |             |                                         \n"
        "                        +------+------+                                         \n"
        "                               |                                                \n"
        "                               v                                                \n"
        "                      [server.aggregate()]                                      \n"
        "                               |                                                \n"
        "                               v                                                \n"
        "                      [server.save_checkpoint()]                                \n"
        "                               |                                                \n"
        "                               v                                                \n"
        "                    [RoundLogger.save()] -> training_log.json"
    )
    add_code_block(doc, dep_graph)

    add_h2(doc, "2.4 Near-Line-by-Line Code Walkthrough of Core Modules")
    add_body(doc, "Let us inspect the exact Python statements in the primary operational files.")

    add_h3(doc, "Walkthrough A: src/models/encoder.py — ViTTinyEncoder")
    add_body(doc, "ViTTinyEncoder encapsulates the ViT-Tiny vision transformer backbone from timm and defines the global feature representation:", bold_prefix="Implementation Detail: ")

    code_enc = (
        "class ViTTinyEncoder(nn.Module):\n"
        "    def __init__(self, embed_dim: int = 192):\n"
        "        super().__init__()\n"
        "        import timm\n"
        "        self.vit = timm.create_model(\n"
        "            'vit_tiny_patch16_224',\n"
        "            pretrained=False,\n"
        "            num_classes=0,  # Strips final classification head\n"
        "        )\n"
        "        vit_dim = self.vit.embed_dim   # 192 for vit_tiny\n"
        "        self.projection = nn.Linear(vit_dim, embed_dim)\n"
        "        self.embed_dim = embed_dim\n"
        "        self.feature_dim = vit_dim\n"
        "\n"
        "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
        "        # 1. Patchify and add positional embeddings\n"
        "        tokens = self.vit.patch_embed(x)                       # (B, 196, 192)\n"
        "        tokens = tokens + self.vit.pos_embed[:, 1:, :]         # 196 spatial patch tokens\n"
        "        tokens = self.vit.pos_drop(tokens)\n"
        "\n"
        "        # 2. Pass patch tokens through 12 Transformer blocks\n"
        "        for block in self.vit.blocks:\n"
        "            tokens = block(tokens)                             # (B, 196, 192)\n"
        "        tokens = self.vit.norm(tokens)\n"
        "\n"
        "        # 3. Global feature representation via Mean Patch Pooling\n"
        "        pooled = tokens.mean(dim=1)                            # (B, 192)\n"
        "        z = self.projection(pooled)                            # (B, embed_dim)\n"
        "        return z"
    )
    add_code_block(doc, code_enc)

    add_body(doc, "Line-by-Line Engineering Analysis:", bold_prefix="Deep Dive: ")
    add_bullet(doc, "Notice that num_classes=0 in timm removes the linear classifier, allowing access to the raw token representations.", bold_prefix="num_classes=0: ")
    add_bullet(doc, "In self.vit.pos_embed[:, 1:, :], index 0 is skipped because standard ViT models place the learnable [CLS] token at index 0. Here, MAE operates purely over the 196 spatial patches.", bold_prefix="pos_embed[:, 1:, :]: ")
    add_bullet(doc, "Instead of using the [CLS] token for global image representation, the forward method computes tokens.mean(dim=1). This global average pooling over all 196 patch embeddings forces the encoder to distribute diagnostic information uniformly across all anatomical regions of the lung, perfectly matching the pretext training objective of the MAE.", bold_prefix="Mean Patch Pooling: ")

    add_h3(doc, "Walkthrough B: src/models/proto_head.py — PrototypicalHead")
    add_body(doc, "The prototypical head projects embeddings into a metric space and classifies queries via temperature-scaled cosine similarity:", bold_prefix="Implementation Detail: ")

    code_proto = (
        "class PrototypicalHead(nn.Module):\n"
        "    def __init__(self, embed_dim: int = 192, num_classes: int = 2, projection_dim: int = 128):\n"
        "        super().__init__()\n"
        "        self.embed_dim = embed_dim\n"
        "        self.num_classes = num_classes\n"
        "        self.projection_dim = projection_dim or embed_dim\n"
        "        # Trainable 2-layer projection MLP\n"
        "        self.projection = nn.Sequential(\n"
        "            nn.Linear(embed_dim, self.projection_dim),\n"
        "            nn.GELU(),\n"
        "            nn.Linear(self.projection_dim, self.projection_dim),\n"
        "        )\n"
        "        self.register_buffer('prototypes', torch.zeros(num_classes, self.projection_dim))\n"
        "        self._prototypes_computed = False\n"
        "\n"
        "    @staticmethod\n"
        "    def _prototypical_logits(queries: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:\n"
        "        # L2 Normalize queries and prototypes\n"
        "        queries_norm = F.normalize(queries, p=2, dim=-1)\n"
        "        prototypes_norm = F.normalize(prototypes, p=2, dim=-1)\n"
        "        # Cosine similarity scaled by inverse temperature tau = 10.0\n"
        "        return torch.mm(queries_norm, prototypes_norm.t()) * 10.0"
    )
    add_code_block(doc, code_proto)

    add_body(doc, "Line-by-Line Engineering Analysis:", bold_prefix="Deep Dive: ")
    add_bullet(doc, "register_buffer('prototypes', ...) ensures class centroids are tracked in the state_dict and moved between CPU and GPU automatically with model.to(device), but are NOT registered as trainable autograd parameters.", bold_prefix="register_buffer: ")
    add_bullet(doc, "torch.mm(queries_norm, prototypes_norm.t()) * 10.0 computes the cosine similarity matrix. Scaling by 10.0 sharpens the probability distribution when fed into cross_entropy, preventing gradient stagnation during leave-one-out adaptation.", bold_prefix="Temperature Scaling: ")

    add_horizontal_rule(doc)
