"""
scripts/manual_builder/part8_fewshot.py
---------------------------------------
Part VIII: Few-Shot Metric Learning: Prototypical Networks, Leave-One-Out Adaptation,
and Clinical Diagnostic Metrics Formulation.
"""

from scripts.manual_builder.styles import (
    add_h1, add_h2, add_h3, add_h4, add_body, add_bullet,
    add_code_block, add_callout, add_table, add_horizontal_rule
)


def build_part8(doc):
    add_h1(doc, "Part VIII: Few-Shot Learning & Evaluation Pipeline")

    add_h2(doc, "8.1 Metric-Space Classification Theory")
    add_body(doc, "Standard supervised deep learning trains a linear classification layer W x + b via Cross-Entropy loss. In extreme low-data regimes (e.g., K = 5 labeled chest X-rays per class), optimizing weight matrices leads to catastrophic overfitting: the linear hyperplane fits idiosyncratic noise and memorizes individual patients rather than pathology.")

    add_body(doc, "Prototypical Networks (Snell et al., NeurIPS 2017) resolve this by operating in a metric space where classification is performed by computing distances to representative class centroids (prototypes):")

    add_bullet(doc, "Given a labeled support set S_k of K samples for class k (where k in {0: Normal, 1: TB}), the prototype c_k is the mean vector of support embeddings in projected space:", bold_prefix="1. Class Prototype: ")
    add_code_block(doc, "c_k = (1 / |S_k|) * sum_{x_i in S_k} f_phi(x_i)    in R^128")

    add_bullet(doc, "A query radiograph x_q is embedded as z_q = f_phi(x_q) and classified by its cosine proximity to each class centroid:", bold_prefix="2. Cosine Metric Logits: ")
    add_code_block(doc, "s(z_q, c_k) = (z_q . c_k) / (||z_q||_2 * ||c_k||_2)\n\nlogits_k = tau * s(z_q, c_k)    where tau = 10.0")

    add_bullet(doc, "The posterior probability that query x_q belongs to class k is given by softmax over the temperature-scaled cosine similarities:", bold_prefix="3. Posterior Probability: ")
    add_code_block(doc, "P(y = k | x_q) = exp(10.0 * s(z_q, c_k)) / sum_{j} exp(10.0 * s(z_q, c_j))")

    add_h2(doc, "8.2 Leave-One-Out Support Adaptation (Stage 2)")
    add_body(doc, "In our verified codebase (src/client/local_train.py), the prototypical projection head is NOT static. It contains a 2-layer MLP projection: Linear(192 -> 128) -> GELU -> Linear(128 -> 128). To adapt this projection head without leaking information from the held-out query or test sets, the algorithm constructs Leave-One-Out support episodes:")

    loo_code = (
        "# Leave-One-Out Support Episode Loop in src/client/local_train.py:\n"
        "for support_position in range(len(support_labels)):\n"
        "    # 1. Leave out the i-th support sample to serve as the episodic query\n"
        "    support_mask = torch.ones(len(support_labels), dtype=torch.bool, device=device)\n"
        "    support_mask[support_position] = False\n"
        "\n"
        "    episode_support_embeddings = support_embeddings[support_mask]     # (2K - 1) samples\n"
        "    episode_support_labels     = support_labels[support_mask]\n"
        "    query_embedding            = support_embeddings[support_position : support_position + 1]\n"
        "    query_label                = support_labels[support_position : support_position + 1]\n"
        "\n"
        "    # 2. Compute prototypes on remaining 2K - 1 support samples\n"
        "    prototypes = head.get_learnable_prototypes(episode_support_embeddings, episode_support_labels)\n"
        "\n"
        "    # 3. Compute loss on the left-out support sample\n"
        "    loss, _ = head.prototypical_loss(query_embedding, query_label, prototypes)\n"
        "    episode_losses.append(loss)\n"
        "\n"
        "# 4. Backpropagate mean episodic loss to adapt projection weights\n"
        "loss = torch.stack(episode_losses).mean()\n"
        "loss.backward()\n"
        "nn.utils.clip_grad_norm_(train_parameters, 1.0)\n"
        "optimizer.step()"
    )
    add_code_block(doc, loo_code)

    add_body(doc, "Why K >= 2 is Mandatory:", bold_prefix="Algorithmic Requirement: ")
    add_body(doc, "If K = 1, leaving out one sample for class k leaves exactly zero support samples for that class, making it impossible to compute class prototype c_k. The code explicitly checks: if k < 2: raise ValueError('Few-shot adaptation requires K >= 2').")

    add_h2(doc, "8.3 Clinical Diagnostic Metrics Formulation")
    add_body(doc, "In medical imaging, classification accuracy is often misleading due to prevalence imbalance. In src/utils/metrics.py, evaluation is grounded in six clinical metrics computed from the binary confusion matrix:")

    metric_headers = ["Metric Name", "Mathematical Formula", "Clinical Interpretation in TB Diagnosis"]
    metric_rows = [
        [
            "ROC-AUC (Area Under ROC Curve)",
            "AUC = integral_0^1 TPR(FPR^-1(t)) dt",
            "Measures the probability that a randomly chosen TB patient is assigned a higher predicted risk than a healthy patient. Primary metric for regulatory submission."
        ],
        [
            "Sensitivity (Recall / True Positive Rate)",
            "TPR = TP / (TP + FN)",
            "Critical clinical safety metric. Measures the proportion of actual TB patients correctly identified. A high sensitivity minimizes deadly false negatives."
        ],
        [
            "Specificity (True Negative Rate)",
            "TNR = TN / (TN + FP)",
            "Measures the proportion of healthy individuals correctly classified as Normal. A low specificity leads to unnecessary sputum tests and patient anxiety."
        ],
        [
            "Accuracy",
            "Acc = (TP + TN) / (TP + TN + FP + FN)",
            "Overall proportion of correct classifications across both disease and normal cohorts."
        ],
        [
            "F1-Score",
            "F1 = 2 * (Precision * Recall) / (Precision + Recall)",
            "Harmonic mean of precision and recall. Useful balance metric under imbalanced class testing."
        ],
        [
            "Balanced Accuracy",
            "B-Acc = 0.5 * (Sensitivity + Specificity)",
            "Unweighted mean of sensitivity and specificity, preventing dominant class bias."
        ]
    ]
    add_table(doc, metric_headers, metric_rows, [1.6, 2.0, 2.9])

    add_horizontal_rule(doc)
