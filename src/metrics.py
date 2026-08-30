"""Classification metrics shared by training, evaluation and reporting."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from . import config


def compute_classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    label_names: Sequence[str] = config.LABEL_NAMES,
) -> dict[str, object]:
    """Return accuracy, macro/weighted F1, confusion matrix and per-class report.

    ``logits`` has shape ``[N, C]`` and holds raw scores (no softmax applied).
    """
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [N, C], got {logits.shape}.")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError(
            f"labels must have shape [N] matching logits, got {labels.shape} vs {logits.shape}."
        )

    predictions = np.argmax(logits, axis=-1)
    class_ids = list(range(len(label_names)))
    report = classification_report(
        labels,
        predictions,
        labels=class_ids,
        target_names=list(label_names),
        zero_division=0,
        output_dict=True,
    )
    return {
        "num_examples": int(labels.shape[0]),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=class_ids).tolist(),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in label_names
        },
    }


def macro_f1(logits: np.ndarray, labels: np.ndarray) -> float:
    """Return macro F1 from raw logits."""
    return float(
        f1_score(labels, np.argmax(np.asarray(logits), axis=-1), average="macro", zero_division=0)
    )


def mcnemar_exact(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Exact McNemar test on paired predictions over the same examples.

    Returns the discordant counts and the two-sided exact binomial p-value.
    ``b`` = A correct and B wrong; ``c`` = A wrong and B correct.
    """
    from scipy.stats import binomtest

    correct_a = np.asarray(predictions_a) == np.asarray(labels)
    correct_b = np.asarray(predictions_b) == np.asarray(labels)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return {"b": 0.0, "c": 0.0, "p_value": 1.0}
    result = binomtest(b, n=b + c, p=0.5, alternative="two-sided")
    return {"b": float(b), "c": float(c), "p_value": float(result.pvalue)}


def bootstrap_accuracy_difference(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    labels: np.ndarray,
    *,
    num_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap CI for accuracy(A) - accuracy(B) on identical examples."""
    correct_a = (np.asarray(predictions_a) == np.asarray(labels)).astype(np.float64)
    correct_b = (np.asarray(predictions_b) == np.asarray(labels)).astype(np.float64)
    difference = correct_a - correct_b
    rng = np.random.default_rng(seed)
    n = difference.shape[0]
    samples = difference[rng.integers(0, n, size=(num_resamples, n))].mean(axis=1)
    return {
        "mean_difference": float(difference.mean()),
        "ci_lower_95": float(np.percentile(samples, 2.5)),
        "ci_upper_95": float(np.percentile(samples, 97.5)),
    }
