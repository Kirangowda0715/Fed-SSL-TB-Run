"""
scripts/manual_builder/part1_architecture.py
--------------------------------------------
Part I: Complete Mental Model, Clinical Problem, and System Architecture.
"""

from scripts.manual_builder.styles import (
    add_h1, add_h2, add_h3, add_h4, add_body, add_bullet,
    add_code_block, add_callout, add_table, add_horizontal_rule
)


def build_part1(doc):
    add_h1(doc, "Part I: Mental Model, Clinical Problem & System Architecture")

    add_h2(doc, "1.1 The Clinical & Regulatory Problem Statement")
    add_body(doc, "Tuberculosis (TB), caused by Mycobacterium tuberculosis, remains one of the world's deadliest infectious diseases, claiming over 1.3 million lives annually according to the World Health Organization (WHO). Chest radiography (CXR) is the frontline diagnostic triage tool recommended by the WHO for mass community screening. However, reading CXRs requires skilled thoracic radiologists who are critically scarce in high-burden regions (e.g., sub-Saharan Africa, South Asia, rural healthcare clinics).")

    add_body(doc, "Deploying conventional supervised Deep Learning (DL) models to automate TB detection is blocked by three severe structural bottlenecks in healthcare AI:", bold_prefix="The Triple Bottleneck: ")

    add_bullet(doc, "Radiological annotation requires certified specialists, costing upwards of $20-$50 per scan. Consequently, while hospitals store tens of thousands of raw chest X-rays, over 95% of them are completely unannotated and unusable by standard supervised methods.", bold_prefix="1. Massive Annotation Deficit: ")
    add_bullet(doc, "Healthcare data is strictly siloed under data privacy statutes such as HIPAA (United States), GDPR (European Union), and the Digital Personal Data Protection Act (DPDPA, India). Hospitals are legally prohibited from pooling patient imaging into a centralized cloud repository to train large models.", bold_prefix="2. Data Privacy & Legal Silos: ")
    add_bullet(doc, "Individual medical centers, especially rural clinics or newly opened screening facilities, often have access to only a handful of confirmed, gold-standard microbiological TB cases (e.g., 5 to 10 confirmed sputum-culture positive patients). Standard deep convolutional networks or vision transformers overfit catastrophically when trained on such tiny cohorts.", bold_prefix="3. Extreme Few-Shot Scarcity: ")

    add_h2(doc, "1.2 The Triad Solution: SSL + FL + Few-Shot Learning")
    add_body(doc, "FedSSL-TB introduces a unified three-pillar architectural triad designed specifically to break the triple bottleneck:")

    triad_headers = ["Pillar", "Addressed Bottleneck", "Core Technology", "Project Implementation Role"]
    triad_rows = [
        [
            "Pillar 1: Self-Supervised Learning",
            "Annotation Deficit (95% Unlabeled Images)",
            "Masked Autoencoder (MAE) with ViT-Tiny",
            "Learns high-level anatomical representations by reconstructing 80% masked chest X-ray patches from NIH ChestX-ray14."
        ],
        [
            "Pillar 2: Federated Learning",
            "Privacy Laws & Legal Data Silos",
            "FedProx Optimization (mu=0.01) across 10 Clients",
            "Coordinates decentralized training across 10 simulated hospital nodes. Raw images never leave hospital boundaries; only weights travel."
        ],
        [
            "Pillar 3: Metric Few-Shot Learning",
            "Extreme Clinical Scarcity (K=5 to 20 samples)",
            "Prototypical Networks (Cosine Metric, tau=10.0)",
            "Constructs diagnostic centroids from 5-20 labeled Shenzhen Hospital images, classifying query X-rays via nearest cosine distance."
        ]
    ]
    add_table(doc, triad_headers, triad_rows, [1.5, 1.5, 1.7, 1.8])

    add_h2(doc, "1.3 Complete System Architecture Schematic")
    add_body(doc, "The diagram below illustrates the exact structural topology of the verified repository, detailing how modules in src/ interact across hospital client silos and the central coordinator.")

    arch_diagram = (
        "+---------------------------------------------------------------------------------------------------------+\n"
        "|                                     FEDSSL-TB SYSTEM TOPOLOGY                                          |\n"
        "+---------------------------------------------------------------------------------------------------------+\n"
        "                                                                                                           \n"
        "     [CENTRAL FEDERATED SERVER: src/server/server.py]                                                     \n"
        "     - Global ViT-Tiny MAE (Encoder: 192-dim, Decoder: 256-dim)                                           \n"
        "     - Broadcasts weights w_global -> Clients                                                             \n"
        "     - Aggregates updates via FedProx (src/server/aggregator.py)                                          \n"
        "     - Checkpoint Manager: experiments/checkpoints/flame_round_XXX.pt                                     \n"
        "            |                                             ^                                               \n"
        "            | Broadcast Weights                           | Upload Local Weights                          \n"
        "            v                                             | (Encoder + Decoder)                           \n"
        "+-------------------------------------------------------------------------------------------------------+ \n"
        "| SIMULATED HOSPITAL CLIENTS (10 Nodes) : src/client/ssl_train.py                                       | \n"
        "|                                                                                                       | \n"
        "|  +------------------------+   +------------------------+        +------------------------+            | \n"
        "|  | Hospital 1 (Node 1)    |   | Hospital 2 (Node 2)    |  ...   | Hospital 10 (Node 10)  |            | \n"
        "|  | Data: NIH Shard (Dirichlet|   | Data: NIH Shard        |        | Data: NIH Shard        |            | \n"
        "|  | Quantity Skew, alpha=1.0) |  | Local MAE Training     |        | Local MAE Training     |            | \n"
        "|  | Loss: MAE + Proximal   |   | Loss: MAE + Proximal   |        | Loss: MAE + Proximal   |            | \n"
        "|  +------------------------+   +------------------------+        +------------------------+            | \n"
        "+-------------------------------------------------------------------------------------------------------+ \n"
        "            |                                                                                             \n"
        "            | Pretrained Global Encoder Weights (Stage 1 Completed)                                      \n"
        "            v                                                                                             \n"
        "+-------------------------------------------------------------------------------------------------------+ \n"
        "| STAGE 2: FEW-SHOT ADAPTATION : src/client/local_train.py                                              | \n"
        "| Support Set: K-shot Shenzhen Hospital X-rays (K Normal + K TB)                                         | \n"
        "| Module: PrototypicalHead (src/models/proto_head.py)                                                   | \n"
        "| - 2-Layer Projection MLP: Linear(192 -> 128) -> GELU -> Linear(128 -> 128)                            | \n"
        "| - Leave-One-Out Support Adaptation (AdamW, lr=1e-3, epochs=30-100)                                    | \n"
        "| - Compute Class Centroids: c_Normal, c_TB in 128-dim metric space                                     | \n"
        "+-------------------------------------------------------------------------------------------------------+ \n"
        "            |                                                                                             \n"
        "            | Class Centroids c_k + Adapted Projection Head                                               \n"
        "            v                                                                                             \n"
        "+-------------------------------------------------------------------------------------------------------+ \n"
        "| STAGE 3: EVALUATION & INFERENCE PIPELINE : src/utils/metrics.py & src/web/api.py                      | \n"
        "| - Held-Out Montgomery CXRs (Never seen in pretraining or support sampling)                            | \n"
        "| - Logits = 10.0 * CosineSimilarity(z_query, c_k)                                                      | \n"
        "| - Metrics: ROC-AUC, Accuracy, Sensitivity, Specificity, F1-Score                                      | \n"
        "| - FastAPI Service (src/web/api.py) & React Console (src/web/frontend/)                                | \n"
        "+-------------------------------------------------------------------------------------------------------+"
    )
    add_code_block(doc, arch_diagram)

    add_h2(doc, "1.4 The Three Project Lifecycles")
    add_body(doc, "The repository is engineered around a strict three-stage execution lifecycle:")

    add_body(doc, "Stage 1: Decentralized Pretraining (Pure SSL)", bold_prefix="Stage 1 — ")
    add_body(doc, "10 simulated hospital clients train the ViT-Tiny Masked Autoencoder exclusively on unlabeled NIH ChestX-ray14 images. In each communication round, the server broadcasts global encoder and decoder parameters. Clients perform local AdamW optimization with FedProx proximal regularization. Only parameter tensors are transmitted back to the server; patient chest radiographs remain strictly on local disk.")

    add_body(doc, "Stage 2: Few-Shot Adaptation (Metric Space Alignment)", bold_prefix="Stage 2 — ")
    add_body(doc, "Once federated pretraining converges, the encoder is frozen (or conditionally fine-tuned). A scarce support set of K labeled images per class is drawn from the Shenzhen Hospital dataset. The PrototypicalHead's 128-dimensional projection MLP is optimized using leave-one-out support episodes. Once adapted, the class prototypes (centroids) c_Normal and c_TB are computed and cached.")

    add_body(doc, "Stage 3: Out-of-Domain Generalization & Clinical Evaluation", bold_prefix="Stage 3 — ")
    add_body(doc, "The system is evaluated on the Montgomery County TB dataset. Montgomery images are strictly held-out: no Montgomery image is ever seen during Stage 1 pretraining or Stage 2 support adaptation. Queries are projected into metric space and classified against the Shenzhen prototypes, measuring true out-of-domain cross-hospital diagnostic generalization.")

    add_h2(doc, "1.5 Information Boundaries & Privacy Reality")
    add_callout(doc, "HONEST DEFENSE: WHAT IS AND IS NOT PRIVACY IN FEDSSL",
        "Viva Examination Key Point: Do NOT claim in your viva that this project implements Differential Privacy or mathematically uncrackable privacy.\n\n"
        "• What IS guaranteed: Raw patient chest radiographs, patient demographics, and hospital metadata never leave the client file system. Raw pixel data is never transmitted across the network.\n"
        "• What IS NOT guaranteed: The current implementation transmits raw neural network weights (state_dicts). Standard federated learning without Differential Privacy (DP-SGD) or Secure Aggregation (SecAgg) is theoretically vulnerable to Gradient Inversion Attacks (e.g., Deep Leakage from Gradients, Zhu et al., NeurIPS 2019), where an adversarial server could reconstruct approximations of training images from parameter updates.\n"
        "• Recommended viva response: 'Our architecture establishes an operational proof-of-concept for privacy-preserving federated pretraining that eliminates raw data pooling. The modular server-client boundary is explicitly engineered to support plug-and-play integration of DP-SGD noise addition and homomorphic encryption in clinical deployment.'",
        "PROFESSOR"
    )

    add_horizontal_rule(doc)
