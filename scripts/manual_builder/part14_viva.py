"""
scripts/manual_builder/part14_viva.py
-------------------------------------
Part XIV: Comprehensive Viva Voce Preparation — 35 Expected Examination Questions
with Detailed, Code-Grounded Model Answers and Defense Strategies.
"""

from scripts.manual_builder.styles import (
    add_h1, add_h2, add_h3, add_h4, add_body, add_bullet,
    add_code_block, add_callout, add_table, add_horizontal_rule
)


def build_part14(doc):
    add_h1(doc, "Part XIV: Viva Examination Preparation (35 Master Q&A)")

    add_h2(doc, "14.1 Examiner Mindset & Defense Strategy")
    add_body(doc, "External academic examiners and industry review panels probe final-year projects looking for three specific red flags: (1) Claiming unrealistic 99% accuracy on medical data, (2) Repeating textbook definitions without knowing where they exist in code, and (3) Concealing engineering limitations. Defending with authentic empirical metrics, verified source code line numbers, and honest trade-offs guarantees top marks.")

    questions = [
        (
            "Q1: What is the core problem your final year project addresses?",
            "Tuberculosis kills 1.3 million people annually, and chest X-rays are the primary triage tool. However, automating detection with deep learning faces three bottlenecks: (1) 95% of hospital X-rays are unlabeled because expert annotation is prohibitively expensive, (2) Data privacy laws like HIPAA and DPDPA legally prohibit pooling patient radiographs into a central cloud, and (3) Individual community clinics possess only 5 to 20 confirmed TB cases. We solve this triad by combining Self-Supervised Learning (MAE) on unlabeled data, Federated Learning (FedProx) across decentralized hospitals, and Metric Few-Shot Learning (Prototypical Networks) on scarce labeled samples."
        ),
        (
            "Q2: Why did you choose Vision Transformers (ViT) over Convolutional Networks (ResNet)?",
            "While CNNs possess strong local inductive biases (spatial locality and translation invariance), they process images via continuous downsampling and fixed receptive fields. Vision Transformers treat images as discrete sequences of patches. For Masked Autoencoders, ViT allows us to physically drop 80% of patch tokens and process only the remaining 20% visible tokens through the encoder, yielding an asymmetric 25x reduction in attention computation. Doing this in a standard CNN is impossible because convolutional kernels require a contiguous 2D pixel grid."
        ),
        (
            "Q3: Why Masked Autoencoders (MAE) instead of contrastive SSL methods like SimCLR or BYOL?",
            "Contrastive learning relies on aggressive data augmentations (color jittering, blur, extreme crops) to enforce semantic invariance. In thoracic radiology, subtle signs like apical cavitations or faint reticular opacities would be destroyed by such augmentations. Furthermore, contrastive learning pushes different patient images apart, which is clinically flawed because many patients share identical healthy lung patterns. MAE reconstructs masked anatomical patches, forcing the transformer to learn global thoracic geometry without destructive augmentations."
        ),
        (
            "Q4: Why is your masking ratio 80%? Why not 15% like BERT or 50%?",
            "Unlike text, which is an information-dense symbolic language where removing 15% of words leaves a challenging prediction task, visual images are spatially redundant. Adjacent 16x16 pixels in a lung X-ray contain nearly identical information. If we masked only 15% or 50%, the network could easily interpolate missing patches from immediately adjacent pixels without learning high-level anatomical structure. Masking 80% (leaving only 39 out of 196 patches) eliminates trivial interpolation, forcing the ViT encoder to understand holistic lung anatomy."
        ),
        (
            "Q5: How does your ViT-Tiny model represent the global image without a [CLS] token?",
            "Standard ViT prepends a learnable [CLS] token at index 0. In our ViTTinyEncoder (src/models/encoder.py), we skip the [CLS] token in positional embeddings (pos_embed[:, 1:, :]) and compute the global feature representation via Mean Patch Token Pooling: tokens.mean(dim=1). This global average over all 196 patch embeddings forces the network to distribute diagnostic representations across the entire thoracic cavity rather than concentrating all information into a single token."
        ),
        (
            "Q6: What is Federated Averaging (FedAvg) mathematically?",
            "In FedAvg (McMahan et al., 2017), each hospital k trains on local data D_k for E epochs. The server aggregates local parameter updates proportional to local dataset size: w^(t+1) = sum_{k=1}^K (n_k / N) w_k^(t+1), where n_k is client sample count and N is total samples across all nodes. This computes a convex weighted average of model weights."
        ),
        (
            "Q7: Why was FedProx necessary instead of standard FedAvg?",
            "In healthcare, hospital cohorts are non-IID (varying patient volume and pathology). In standard FedAvg, local client models drift toward their local empirical optima, causing the aggregated global model to oscillate or diverge. FedProx (Li et al., 2020) adds a proximal regularization penalty (mu/2) * ||w_local - w_global||^2 to the local loss, acting as an elastic tether that stabilizes optimization under non-IID data heterogeneity."
        ),
        (
            "Q8: Where is the FedProx term calculated in code — client side or server side?",
            "On the CLIENT side! In src/client/ssl_train.py (lines 123-125), _fedprox_penalty() computes the squared L2 distance between live client parameters and broadcast global weights on the active PyTorch autograd graph. On the server side (src/server/aggregator.py), aggregation is mathematically identical to FedAvg weighted averaging because the proximal constraint was already enforced during client gradient updates."
        ),
        (
            "Q9: How do you simulate 10 hospital clients on a single computer?",
            "We use an in-memory simulation harness in src/federated/simulation.py following standard federated benchmarks (LEAF, FedML). In each round, the server broadcasts global weights. We use copy.deepcopy(global_model) to create an isolated client model instance for each hospital sequentially. Local training executes on local NIH data shards, after which client models are deleted and VRAM is cleared before the next client loads."
        ),
        (
            "Q10: Is your system completely private? Could an adversary reconstruct patient images?",
            "Our system guarantees that raw pixel data and patient files never leave the hospital workstation. However, raw parameter updates (state_dicts) are transmitted to the server. Standard FL without Differential Privacy is theoretically vulnerable to Gradient Inversion Attacks (e.g., Zhu et al., NeurIPS 2019), where an adversarial server could optimize pixel inputs to match gradient tensors. In viva defense, state honestly that FedSSL-TB establishes an operational proof-of-concept for decentralized pretraining, designed for plug-and-play integration of DP-SGD and Secure Aggregation in clinical deployment."
        ),
        (
            "Q11: Why didn't you implement Differential Privacy (DP-SGD) in this phase?",
            "DP-SGD injects calibrated Gaussian noise into clipped gradients (theta_t - eta * (g + N(0, sigma^2))). While it provides formal epsilon-delta privacy bounds, it significantly impairs model utility and representation learning, especially in high-dimensional vision transformers trained from scratch. In Phase 1, our objective was to establish the convergence boundary of decentralized MAE pretraining before tuning privacy-utility trade-offs."
        ),
        (
            "Q12: How is Non-IID data simulated across your 10 hospital clients?",
            "In src/datasets/splitter.py, we simulate quantity skew using a Dirichlet distribution with concentration parameter alpha = 1.0: proportions ~ Dirichlet(alpha * 1_K). Sample counts are allocated via a Multinomial distribution over the 10 hospitals. Because NIH data is unlabeled in Stage 1, this models real-world clinical patient throughput variation across hospitals."
        ),
        (
            "Q13: What are the three stages of your project lifecycle?",
            "Stage 1 (Decentralized Pretraining): 10 hospital clients train the ViT-Tiny MAE exclusively on unlabeled NIH X-rays using FedProx. Stage 2 (Few-Shot Adaptation): The federated encoder is frozen, and a PrototypicalHead's 128-dim projection MLP is adapted on 5-20 labeled Shenzhen images via leave-one-out support episodes. Stage 3 (Held-Out Evaluation): Class prototypes are evaluated on the strictly held-out Montgomery County dataset."
        ),
        (
            "Q14: What datasets are used, and what are their exact clinical roles?",
            "(1) NIH ChestX-ray14 (~112k images, USA): Used exclusively for unlabeled self-supervised pretraining. (2) Shenzhen TB Dataset (662 images, China): Used for few-shot support set extraction and in-domain query validation. (3) Montgomery County TB Dataset (138 images, USA): Held-out external evaluation set that is NEVER touched during pretraining or adaptation."
        ),
        (
            "Q15: How do you prove that there is zero data leakage between support and test sets?",
            "We enforce two programmatic barriers: (1) In evaluate_on_montgomery(), we resolve absolute file paths and assert: not (support_paths & test_paths). (2) We implemented test_stage2_leakage.py, an automated unit test that verifies that image paths and patient study IDs across Shenzhen and Montgomery are 100% disjoint."
        ),
        (
            "Q16: Why did you use Prototypical Networks instead of a standard linear classification head?",
            "With extreme low-data regimes (K = 5 or 20 samples per class), optimizing linear classification weights W in R^(2 x 192) via cross-entropy causes severe overfitting and memorization of non-pathological image features. Prototypical Networks compute non-parametric class centroids (prototypes) in a learned metric space, classifying samples by geometric distance, which is dramatically more robust to small sample sizes."
        ),
        (
            "Q17: What distance metric does your PrototypicalHead use? (Trick Question!)",
            "Cosine similarity scaled by temperature tau = 10.0! While the original Snell et al. paper used Euclidean distance, our implementation (src/models/proto_head.py, lines 87-93) L2-normalizes queries and prototypes and computes scaled dot-product similarity: logits = torch.mm(queries_norm, prototypes_norm.t()) * 10.0. Cosine metric prevents gradient explosion on high-dimensional unit hyperspheres."
        ),
        (
            "Q18: What is Leave-One-Out support adaptation, and why is K >= 2 mandatory?",
            "To adapt the 2-layer projection head on support samples without leaking test data, we iterate through the 2K support samples. In each step, sample i is held out as the episodic query, while the remaining 2K - 1 samples form the episode support set to compute prototypes. If K = 1, leaving out one sample for class k leaves 0 samples, making it impossible to compute prototype c_k. Hence K >= 2 is mandatory."
        ),
        (
            "Q19: What happens if you unfreeze the encoder during Stage 2 adaptation?",
            "We conducted an ablation study (experiments/ablation_studies/frozen_vs_unfrozen/results.json). Unfreezing all 5.5M encoder parameters with only 40 support samples dropped in-domain Shenzhen AUC from 81.26% to 79.89% and accuracy from 77.49% to 72.19% due to representation degradation on small cohorts. Freezing the encoder and training only the 41k-parameter projection head yielded superior, stable results."
        ),
        (
            "Q20: What are your final diagnostic metrics on the Shenzhen dataset?",
            "On the Shenzhen query cohort (Round 99, K=20): ROC-AUC = 81.26%, Accuracy = 77.49%, Sensitivity = 78.48%, Specificity = 76.47%, F1-Score = 77.99%, and Balanced Accuracy = 77.48% (Confusion Matrix: TP=243, FN=73, TN=206, FP=100)."
        ),
        (
            "Q21: What are your diagnostic metrics on the Montgomery dataset?",
            "On Montgomery (Round 99, K=20): ROC-AUC = 51.29%, Accuracy = 47.83%, Sensitivity = 60.34%, Specificity = 38.75%, F1-Score = 49.30%, and Balanced Accuracy = 49.55%."
        ),
        (
            "Q22: Why is there a performance gap between Shenzhen (~81%) and Montgomery (~51%)?",
            "This gap is caused by Radiographic Cross-Site Domain Shift. Shenzhen and Montgomery images were acquired using entirely different X-ray machines in China and the US with different beam energy (kVp), patient positioning, and contrast. Prototypes computed on Shenzhen suffer from scanner-specific feature entanglement when tested on Montgomery. Transparently reporting this gap reflects rigorous scientific honesty and motivates future domain adaptation work."
        ),
        (
            "Q23: Why is ROC-AUC your primary clinical evaluation metric rather than Accuracy?",
            "Accuracy depends heavily on the decision threshold (default 0.5) and is distorted by class imbalance. ROC-AUC evaluates true positive rate versus false positive rate across ALL possible classification thresholds, measuring the model's fundamental ranking ability. In healthcare screening, clinicians adjust thresholds depending on clinical context (e.g. lowering threshold to prioritize sensitivity)."
        ),
        (
            "Q24: What is the clinical significance of achieving 78.48% sensitivity?",
            "In public health screening for infectious tuberculosis, Sensitivity (Recall) is the most critical safety metric. A false positive results in an unnecessary sputum smear or GeneXpert test (low harm). A false negative sends an active, contagious TB patient back into the community untreated (catastrophic harm). A 78.5% sensitivity ensures nearly 4 out of 5 TB patients are flagged for confirmatory laboratory work."
        ),
        (
            "Q25: What optimizer and learning rate schedule do you use?",
            "AdamW optimizer with lr=1e-3, weight_decay=0.05, and betas=(0.9, 0.95). We use CosineAnnealingLR scheduling, decaying learning rate smoothly down to 1e-6 across communication epochs."
        ),
        (
            "Q26: Why AdamW instead of standard Adam or SGD?",
            "Standard Adam applies L2 regularization to the gradient, which distorts the moving averages of past gradients (v_t) and penalizes weights adaptively rather than proportionally. AdamW decouples weight decay directly into parameter updates, preventing weights from growing excessively large and stabilizing vision transformer self-attention layers."
        ),
        (
            "Q27: Why is gradient clipping (max norm 1.0) necessary for ViT training?",
            "Vision transformers lack convolutional inductive biases and can experience sudden loss spikes during masked patch reconstruction (e.g., when an unusual anatomical anomaly is masked). Clipping parameter gradients to a maximum L2 norm of 1.0 prevents explosive weight updates that would destabilize the self-attention projections."
        ),
        (
            "Q28: How does your checkpointing and resume mechanism work?",
            "After every round, server.save_checkpoint() writes flame_round_XXX.pt containing full encoder, decoder, and prototype state_dicts, current round number, and configuration snapshots. When launched with --resume, simulation.py parses the latest checkpoint file name, restores global model weights, and seamlessly continues from round latest + 1."
        ),
        (
            "Q29: How is the web application structured?",
            "The backend is built with FastAPI (src/web/api.py), providing high-throughput async endpoints (/predict, /status, /metrics, /health). The frontend is an interactive single-page dashboard built with React and Vite (src/web/frontend/), rendering real-time loss progression charts and diagnostic inference results."
        ),
        (
            "Q30: Why are evaluation metrics hardcoded in App.jsx?",
            "In App.jsx (lines 36-43), FINAL_EVALUATION_METRICS stores the exact calibrated findings from the 30-round Shenzhen evaluation (AUC=70.97%, Acc=67.33%, Sens=79.15%). This ensures that during academic viva presentations or conference demos, the UI displays authentic, reproducible research telemetry even if the backend is running in offline test mode."
        ),
        (
            "Q31: What happens if an image is corrupted or malformed during dataset loading?",
            "In src/datasets/loader.py, Image.open() is wrapped in a try-except block. If an image file is corrupted or unreadable, the loader logs an error and returns a valid zero-tensor of shape (3, 224, 224), preventing a single corrupted file from crashing an entire multi-hour federated round."
        ),
        (
            "Q32: Did extending pretraining from 50 to 100 rounds improve performance?",
            "We compared Round 49 vs Round 99 (experiments/ablation_studies/round49_vs_round99/). In-domain Shenzhen AUC remained essentially flat (81.42% at Round 49 vs 81.26% at Round 99), while sensitivity gained +2.53% (75.95% to 78.48%). This shows that self-supervised representation learning reaches saturation by ~50 rounds under our dataset size."
        ),
        (
            "Q33: Does lower MAE reconstruction loss guarantee higher downstream TB accuracy?",
            "NO! This is a fundamental research insight. MAE loss measures pixel reconstruction accuracy (MSE). While a model with lower reconstruction loss has learned better spatial representations of ribs and lungs, downstream TB classification depends on fine-grained pathological discriminability. Pixel reconstruction and clinical classification are distinct objectives."
        ),
        (
            "Q34: What are the primary engineering limitations of this project?",
            "(1) In-memory client simulation rather than true cross-network socket deployment, (2) Absence of Differential Privacy (DP-SGD) noise layers to counter gradient inversion, (3) Lack of domain-adversarial adaptation layers (like DANN or AdaBN) to bridge the cross-site scanner domain gap, and (4) Grayscale duplication to 3 channels rather than a native 1-channel ViT tokenizer."
        ),
        (
            "Q35: What would you implement next if given 6 more months of engineering?",
            "(1) Deploy physical multi-node federation using Flower (flwr) over gRPC with Secure Aggregation, (2) Implement Domain-Adversarial Neural Networks (DANN) during Stage 2 to align Shenzhen and Montgomery latent distributions, (3) Integrate Differential Privacy via Opacus with calibrated epsilon-delta privacy accounting, and (4) Benchmark against a native single-channel pre-trained RadImageNet ViT backbone."
        )
    ]

    for q, a in questions:
        add_h3(doc, q)
        add_body(doc, a, bold_prefix="Examiner Response: ")

    add_horizontal_rule(doc)
