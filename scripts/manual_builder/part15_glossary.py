"""
scripts/manual_builder/part15_glossary.py
-----------------------------------------
Part XV: Comprehensive Technical Glossary, Repository Source Code Index,
and Authoritative Academic Literature References.
"""

from scripts.manual_builder.styles import (
    add_h1, add_h2, add_h3, add_h4, add_body, add_bullet,
    add_code_block, add_callout, add_table, add_horizontal_rule
)


def build_part15(doc):
    add_h1(doc, "Part XV: Technical Glossary, Source Index & References")

    add_h2(doc, "15.1 Comprehensive Technical Glossary")
    add_body(doc, "Authoritative definitions for all machine learning, federated learning, and medical imaging terminology used throughout this manual:")

    glossary = [
        ("Vision Transformer (ViT)", "An attention-based deep neural network that processes 2D images as sequences of non-overlapping 16x16 patch tokens using Multi-Head Self-Attention rather than convolutional filters."),
        ("Masked Autoencoder (MAE)", "A self-supervised learning architecture that randomly hides a high percentage (e.g. 80%) of image patches and trains an encoder-decoder network to reconstruct the missing raw pixels."),
        ("Federated Learning (FL)", "A decentralized machine learning paradigm where multiple client institutions train local models on local datasets and transmit only parameter weights to a central server, preserving raw data locality."),
        ("FedAvg", "Federated Averaging. The baseline aggregation algorithm that computes a sample-size-weighted average of client model parameters to update the global consensus model."),
        ("FedProx", "An enhanced federated optimization framework that adds a proximal regularization penalty (mu/2)*||w - w_global||^2 to local client objectives, mitigating client drift caused by non-IID data heterogeneity."),
        ("Client Drift", "The phenomenon in federated optimization where local client models diverge toward distinct local minima due to heterogeneous (non-IID) local data distributions, destabilizing server aggregation."),
        ("Non-IID Data", "Non-Independently and Non-Identically Distributed data. In healthcare, patient volumes, disease prevalence, and scanner demographics vary widely across hospital sites."),
        ("Dirichlet Distribution", "A continuous multivariate probability distribution used in federated learning benchmarks to generate synthetic non-IID partition splits with controlled concentration parameter alpha."),
        ("Prototypical Networks", "A metric-space few-shot learning framework where an encoder embeds labeled support images, computes mean class centroids (prototypes), and classifies queries by distance to nearest prototypes."),
        ("Few-Shot Learning (K-Shot)", "A regime where a model must learn to classify categories given only K labeled support examples per class (e.g., K = 5 or 20 samples)."),
        ("Leave-One-Out (LOO) Adaptation", "An episodic training scheme where each support sample is sequentially held out as a query while prototypes are computed on the remaining support samples, adapting the projection head without test set leakage."),
        ("Temperature Scaling (tau)", "A hyperparameter dividing or multiplying distance logits before softmax (here tau=10.0), controlling the sharpness and entropy of posterior probability distributions."),
        ("ROC-AUC", "Receiver Operating Characteristic Area Under the Curve. The probability that a classifier ranks a randomly chosen positive disease case higher than a randomly chosen negative normal case."),
        ("Sensitivity (True Positive Rate)", "The proportion of actual tuberculosis patients correctly identified by the diagnostic model: TP / (TP + FN). Also known as recall."),
        ("Specificity (True Negative Rate)", "The proportion of healthy, non-TB individuals correctly identified as Normal by the diagnostic model: TN / (TN + FP)."),
        ("Balanced Accuracy", "The arithmetic mean of sensitivity and specificity: 0.5 * (TPR + TNR), providing an unbiased performance metric under severe class prevalence skew."),
        ("AdamW Optimizer", "An adaptive gradient descent optimizer that decouples L2 weight decay from gradient updates, stabilizing vision transformer parameter optimization."),
        ("Cosine Annealing LR", "A learning rate schedule that decays the step size smoothly from a maximum value down to a minimum value following a half-period cosine curve."),
        ("Gradient Norm Clipping", "A numerical stabilization technique that scales down parameter gradient vectors if their global L2 norm exceeds a maximum threshold (max_norm = 1.0)."),
        ("Cross-Site Domain Shift", "The performance degradation observed when a model trained on one hospital cohort (Shenzhen) is tested on an external hospital cohort (Montgomery) due to differences in scanner calibration, beam kVp, and patient demographics."),
        ("Gradient Inversion Attack", "A privacy attack in federated learning where an adversarial server reconstructs training images by optimizing pixel inputs to match observed parameter update tensors (e.g. DLG, Zhu et al.)."),
        ("Differential Privacy (DP-SGD)", "A rigorous mathematical privacy framework that adds calibrated Gaussian noise to clipped per-sample gradients, ensuring individual patient participation cannot be determined."),
        ("Patchify", "The tensor reshaping operation that decomposes an image of shape (B, C, H, W) into a 2D sequence of flattened spatial patches (B, N, P*P*C)."),
        ("Unpatchify", "The inverse tensor reshaping operation that reconstructs a full 2D spatial image (B, C, H, W) from a sequence of patch pixel predictions (B, N, P*P*C).")
    ]

    for term, definition in glossary:
        add_bullet(doc, definition, bold_prefix=f"{term}: ")

    add_h2(doc, "15.2 Comprehensive Repository Source Index")
    add_body(doc, "A definitive mapping of classes, functions, and configuration files to exact locations in the verified codebase:")

    idx_headers = ["Component Name", "Type", "Source File Path", "Primary Responsibility"]
    idx_rows = [
        ["ViTTinyEncoder", "Class", "src/models/encoder.py", "ViT-Tiny backbone returning 192-dim mean-pooled embeddings."],
        ["MAEDecoder", "Class", "src/models/decoder.py", "Lightweight 4-layer transformer decoder reconstructing 768-dim patches."],
        ["MaskedAutoencoder", "Class", "src/models/mae.py", "Full MAE model handling patchify, masking, visible encode, and loss."],
        ["PrototypicalHead", "Class", "src/models/proto_head.py", "Trainable 2-layer projection MLP and temperature-scaled cosine classifier."],
        ["ssl_local_train", "Function", "src/client/ssl_train.py", "Local hospital training loop executing MAE pass and FedProx penalty."],
        ["_fedprox_penalty", "Function", "src/client/ssl_train.py", "Computes (mu/2) * ||w_local - w_global||^2 on live parameter tensors."],
        ["finetune_local", "Function", "src/client/local_train.py", "Adapts projection head via leave-one-out support episodes on Shenzhen."],
        ["evaluate_on_montgomery", "Function", "src/client/local_train.py", "Evaluates prototypes on strictly held-out Montgomery test cohort."],
        ["fedavg", "Function", "src/server/aggregator.py", "Computes sample-weighted convex average across client state_dicts."],
        ["fedprox", "Function", "src/server/aggregator.py", "Server-side wrapper for FedProx aggregation."],
        ["FederatedServer", "Class", "src/server/server.py", "Central server managing global model, broadcasts, and checkpoints."],
        ["main", "Function", "src/federated/simulation.py", "Master orchestration script coordinating 30-100 round federated loop."],
        ["NIHDataset", "Class", "src/datasets/loader.py", "Scans and loads unlabeled NIH ChestX-ray14 pretraining cohort."],
        ["ShenzhenDataset", "Class", "src/datasets/loader.py", "Loads labeled Shenzhen TB images and parses binary folder labels."],
        ["MontgomeryDataset", "Class", "src/datasets/loader.py", "Loads held-out Montgomery County evaluation images."],
        ["split_nih_to_hospitals", "Function", "src/datasets/splitter.py", "Partitions NIH data into Dirichlet non-IID hospital shards."],
        ["split_cache_matches", "Function", "src/datasets/splitter.py", "Validates cached partition metadata against current YAML configuration."],
        ["load_config", "Function", "src/utils/config.py", "Parses YAML files into SimpleNamespace and applies CLI overrides."],
        ["evaluate", "Function", "src/utils/metrics.py", "Calculates ROC-AUC, accuracy, sensitivity, specificity, and F1."],
        ["FastAPI app", "Instance", "src/web/api.py", "Asynchronous web service exposing /predict, /metrics, and /status."],
        ["default.yaml", "Config", "configs/default.yaml", "10-hospital default configuration (alpha=1.0, 30 rounds)."],
        ["kaggle_suhas_100round.yaml", "Config", "configs/kaggle_suhas_100round.yaml", "100-round GPU production configuration for Kaggle."]
    ]
    add_table(doc, idx_headers, idx_rows, [1.6, 0.8, 1.8, 2.3])

    add_h2(doc, "15.3 Authoritative Academic Literature References")
    add_body(doc, "Key foundational research papers referenced throughout the architecture and defense of this project:")

    refs = [
        ("He et al. (2022)", "K. He, X. Chen, S. Xie, Y. Li, P. Dollar, and R. Girshick, 'Masked Autoencoders Are Scalable Vision Learners', in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 16000-16009."),
        ("Dosovitskiy et al. (2021)", "A. Dosovitskiy et al., 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale', in International Conference on Learning Representations (ICLR), 2021."),
        ("McMahan et al. (2017)", "B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, 'Communication-Efficient Learning of Deep Networks from Decentralized Data', in International Conference on Artificial Intelligence and Statistics (AISTATS), 2017, pp. 1273-1282."),
        ("Li et al. (2020)", "T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, 'Federated Optimization in Heterogeneous Networks', in Proceedings of Machine Learning and Systems (MLSys), vol. 2, 2020, pp. 429-450."),
        ("Snell et al. (2017)", "J. Snell, K. Swersky, and R. Zemel, 'Prototypical Networks for Few-Shot Learning', in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, 2017."),
        ("Zhu et al. (2019)", "L. Zhu, Z. Liu, and S. Han, 'Deep Leakage from Gradients', in Advances in Neural Information Processing Systems (NeurIPS), vol. 32, 2019."),
        ("Loshchilov & Hutter (2019)", "I. Loshchilov and F. Hutter, 'Decoupled Weight Decay Regularization', in International Conference on Learning Representations (ICLR), 2019."),
        ("Jaeger et al. (2014)", "S. Jaeger, P. Karargyris, S. Candemir, et al., 'Two public chest X-ray datasets for computer-aided detection of pulmonary diseases', Quantitative Imaging in Medicine and Surgery, vol. 4, no. 6, pp. 475-477, 2014. [Shenzhen & Montgomery]."),
        ("Wang et al. (2017)", "X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri, and R. M. Summers, 'ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases', in CVPR, 2017. [NIH ChestX-ray14].")
    ]

    for cite, full_ref in refs:
        add_bullet(doc, full_ref, bold_prefix=f"{cite}: ")

    add_horizontal_rule(doc)
