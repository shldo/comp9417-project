from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def compute_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_prob: Any,
) -> dict[str, float]:
    """
    Compute core binary classification metrics.

    Expects:
    - y_pred: predicted class labels
    - y_prob: positive-class probabilities or scores
    """
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    y_prob_array = np.asarray(y_prob)

    # If probabilities come from predict_proba with shape (n_samples, 2),
    # use the positive-class probability column.
    if y_prob_array.ndim == 2 and y_prob_array.shape[1] >= 2:
        y_prob_array = y_prob_array[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "roc_auc": float(roc_auc_score(y_true_array, y_prob_array)),
    }
    return metrics
