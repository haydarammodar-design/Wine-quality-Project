"""Metrics, decision threshold and plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Turn probabilities into 0/1 decisions.
    Args:
        scores: Probabilities of class 1.
        threshold: A wine is predicted good when its score is >= threshold.
    Returns:
        An array of 0 and 1.
    """
    return (np.asarray(scores) >= threshold).astype(int)


def choose_threshold(
    y_true: np.ndarray, scores: np.ndarray, min_recall: float
) -> float:
    """Pick the threshold with the best precision among those reaching a recall.
    Args:
        y_true: True labels (0 or 1).
        scores: Out-of-fold probabilities of class 1.
        min_recall: The recall the threshold must reach, in (0, 1].
    Returns:
        The chosen threshold.
    Raises:
        ValueError: If ``min_recall`` is not in (0, 1].
    """
    if not 0 < min_recall <= 1:
        raise ValueError(f"min_recall must be in (0, 1], got {min_recall}")
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    precision, recall = precision[:-1], recall[:-1]
    valid = recall >= min_recall
    best = int(np.argmax(np.where(valid, precision, -1.0)))
    return float(thresholds[best])


def compute_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    """Compute threshold-free and threshold-based metrics.
    Args:
        y_true: True labels (0 or 1).
        scores: Probabilities of class 1.
        threshold: The decision threshold.
    Returns:
        A dictionary metric name -> value.
    """
    y_pred = apply_threshold(scores, threshold)
    return {
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "threshold": float(threshold),
    }


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, path: Path | None = None
) -> Figure:
    """Draw the confusion matrix and optionally save it.
    Args:
        y_true: True labels.
        y_pred: Predicted labels (after the threshold).
        path: Where to save the PNG file; nothing is saved when None.
    Returns:
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["not good", "good"], ax=ax, colorbar=False
    )
    ax.set_title("Confusion matrix (test set)")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    return fig


def plot_threshold_curve(
    y_true: np.ndarray, scores: np.ndarray, threshold: float, path: Path | None = None
) -> Figure:
    """Plot precision and recall against the threshold.
    Args:
        y_true: True labels.
        scores: Out-of-fold probabilities of class 1.
        threshold: The chosen threshold, drawn as a vertical line.
        path: Where to save the PNG file; nothing is saved when None.
    Returns:
        The matplotlib figure.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, precision[:-1], label="precision")
    ax.plot(thresholds, recall[:-1], label="recall")
    ax.axvline(
        threshold, color="grey", linestyle="--", label=f"chosen = {threshold:.2f}"
    )
    ax.set_xlabel("threshold on P(good)")
    ax.set_ylabel("score")
    ax.legend()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    return fig
