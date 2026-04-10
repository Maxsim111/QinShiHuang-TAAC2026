from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def compute_classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = labels.astype(np.float32, copy=False)
    probabilities = probabilities.astype(np.float32, copy=False)
    predictions = (probabilities >= 0.5).astype(np.int32)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "logloss": float(log_loss(labels, probabilities, labels=[0, 1])),
    }
    metrics["auc"] = float(roc_auc_score(labels, probabilities)) if np.unique(labels).size > 1 else float("nan")
    return metrics


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    return (
        f"{prefix} "
        f"AUC={metrics['auc']:.6f} "
        f"LogLoss={metrics['logloss']:.6f} "
        f"Accuracy={metrics['accuracy']:.6f}"
    )
