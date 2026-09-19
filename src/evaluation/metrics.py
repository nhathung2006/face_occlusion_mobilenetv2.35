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


def compute_metrics(y_true, y_pred, num_classes: int) -> ClassificationMetrics:
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        confusion=confusion_matrix(y_true, y_pred, labels=list(range(num_classes))),
    )
