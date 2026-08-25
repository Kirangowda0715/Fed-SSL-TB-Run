"""
src/utils/metrics.py
--------------------
Evaluation metrics for binary TB detection.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
)


def evaluate(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """
    Compute all evaluation metrics for binary TB classification.

    Args:
        y_true        : 1-D array of ground-truth labels (0=Normal, 1=TB)
        y_pred_proba  : 1-D array of predicted probabilities for class 1 (TB)

    Returns:
        dict with keys:
            auc, accuracy, sensitivity, specificity, f1, confusion_matrix
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # AUC — requires at least one positive and one negative sample
    try:
        auc = float(roc_auc_score(y_true, y_pred_proba))
    except ValueError:
        auc = float("nan")

    accuracy = float(accuracy_score(y_true, y_pred))

    # Sensitivity = Recall for TB class (label=1)
    sensitivity = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))

    # Specificity = Recall for Normal class (label=0)
    specificity = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))

    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)

    return {
        "auc": auc,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm,
    }


def format_metrics(metrics: dict) -> str:
    """Pretty-print a metrics dict as a one-line summary string."""
    return (
        f"AUC={metrics['auc']:.4f}  "
        f"Acc={metrics['accuracy']:.4f}  "
        f"Sens={metrics['sensitivity']:.4f}  "
        f"Spec={metrics['specificity']:.4f}  "
        f"F1={metrics['f1']:.4f}"
    )
