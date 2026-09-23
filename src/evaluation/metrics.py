from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray
    logit_min: float | None = None
    logit_max: float | None = None
    logit_mean: float | None = None
    prob_min: float | None = None
    prob_max: float | None = None


def compute_metrics(
    y_true,
    y_pred,
    num_classes: int,
    logits: list[float] | None = None,
    probs: list[float] | None = None,
) -> ClassificationMetrics:
    logit_min = float(np.min(logits)) if logits and len(logits) > 0 else None
    logit_max = float(np.max(logits)) if logits and len(logits) > 0 else None
    logit_mean = float(np.mean(logits)) if logits and len(logits) > 0 else None
    prob_min = float(np.min(probs)) if probs and len(probs) > 0 else None
    prob_max = float(np.max(probs)) if probs and len(probs) > 0 else None

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        confusion=confusion_matrix(y_true, y_pred, labels=list(range(num_classes))),
        logit_min=logit_min,
        logit_max=logit_max,
        logit_mean=logit_mean,
        prob_min=prob_min,
        prob_max=prob_max,
    )
